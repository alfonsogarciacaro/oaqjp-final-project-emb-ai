import sqlite3
import hashlib
import time
import json
from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np # Required for vec0 float32 conversion

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
EMBEDDING_DIMENSION = 768 # IMPORTANT: Set this to your model's output dimension

# Placeholder for the PyTorch model embedding generation function.
# Replace this with your actual PyTorch model.
def generate_embedding(text: str) -> List[float]:
    """
    Placeholder function to generate embeddings from text.
    Replace this with your actual PyTorch model.

    Args:
        text (str): The text to generate an embedding for.

    Returns:
        List[float]: A dummy embedding (ensure dimension matches EMBEDDING_DIMENSION).
    """
    # logging.debug(f"Generating dummy embedding for text length: {len(text)}")
    # Ensure the dummy embedding matches the specified dimension
    return [0.0] * EMBEDDING_DIMENSION

# Helper function to convert list of floats to bytes suitable for vec0
def embedding_to_blob(embedding: List[float]) -> bytes:
    """Converts a list of floats to a numpy float32 byte blob."""
    return np.array(embedding, dtype=np.float32).tobytes()

# Helper function to convert bytes from vec0 back to list of floats (if needed)
def blob_to_embedding(blob: bytes) -> List[float]:
    """Converts a numpy float32 byte blob back to a list of floats."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


class DBManager:
    """
    Manages the database connection and operations using SQLite with vec0 extension.
    Handles the revised schema with file_id, chunk_id, and vec0 virtual table.
    """
    def __init__(self, db_type: str, db_name: str = "file_chunks_v3.db", embedding_dim: int = 768):
        self.db_type = db_type
        self.db_name = db_name
        self.embedding_dim = embedding_dim
        self.extension_loaded = False

    def _connect(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Establishes DB connection, enables WAL, loads vec extension."""
        if self.db_type == "sqlite":
            try:
                # isolation_level=None enables autocommit mode, but we'll manage transactions manually
                conn = sqlite3.connect(self.db_name, isolation_level=None) # Changed isolation level
                cursor = conn.cursor()

                # Enable WAL mode FIRST (best practice)
                cursor.execute("PRAGMA journal_mode=WAL;")
                # Verify WAL mode
                current_journal_mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
                if current_journal_mode.lower() != 'wal':
                     logging.warning(f"Failed to set WAL mode. Current mode: {current_journal_mode}")
                else:
                    logging.info("SQLite WAL mode enabled.")

                # Load the vector extension
                if not self.extension_loaded:
                    try:
                        conn.enable_load_extension(True)
                        cursor.execute("SELECT load_extension('vec.so')")
                        logging.info("SQLite vector extension 'vec.so' loaded.")
                        self.extension_loaded = True
                    except sqlite3.Error as e:
                        logging.error(f"Error loading SQLite vector extension 'vec.so': {e}. Vector search WILL fail.")
                        # Decide if this is fatal
                        raise RuntimeError("Failed to load vec.so extension") from e
                    finally:
                        # Security practice: disable extension loading after use
                        conn.enable_load_extension(False)

                return conn, cursor
            except sqlite3.Error as e:
                logging.error(f"SQLite connection/setup error: {e}")
                raise # Re-raise the exception
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _disconnect(self, conn: sqlite3.Connection):
        """Closes the database connection."""
        if conn:
            try:
                conn.close()
            except sqlite3.Error as e:
                logging.error(f"Error closing SQLite connection: {e}")


    def create_tables(self):
        """Creates the necessary database tables based on the revised schema."""
        conn, cursor = None, None
        try:
            conn, cursor = self._connect()
            logging.info("Creating/verifying database tables...")

            # Begin transaction for DDL
            cursor.execute("BEGIN")

            # files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    last_access_time INTEGER NOT NULL,
                    UNIQUE(filepath, file_hash)
                )
            """)

            # chunk_info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_info (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_hash TEXT NOT NULL UNIQUE
                )
            """)
            # Index on chunk_hash for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_info_hash ON chunk_info(chunk_hash)")

            # chunk_embeddings virtual table using vec0
            # IMPORTANT: Dimension must match your model!
            # Storing chunk_id as a hidden column to link back
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                    embedding VECTOR[{self.embedding_dim}],
                    chunk_id INTEGER HIDDEN
                )
            """)
            # Note: vec0 manages its own indexing internally.

            # chunk_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunk_info(chunk_id),
                    start_offset INTEGER NOT NULL,
                    length INTEGER NOT NULL
                )
            """)
            # Indexes for faster joins/filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_metadata_file_id ON chunk_metadata(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_metadata_chunk_id ON chunk_metadata(chunk_id)")

            # Commit transaction
            cursor.execute("COMMIT") # Use execute for COMMIT/ROLLBACK with isolation_level=None
            logging.info("Database tables created or verified successfully.")

        except (sqlite3.Error, ValueError, RuntimeError) as e:
            logging.error(f"Error creating tables: {e}")
            if conn and cursor:
                try:
                    cursor.execute("ROLLBACK")
                    logging.warning("Transaction rolled back due to table creation error.")
                except sqlite3.Error as rb_err:
                    logging.error(f"Error during rollback: {rb_err}")
            raise # Re-raise the error
        finally:
            if conn:
                self._disconnect(conn)

    def get_file_id(self, cursor: sqlite3.Cursor, filepath: str, file_hash: str) -> Optional[int]:
        """Looks up file_id using filepath and file_hash."""
        cursor.execute("SELECT file_id FROM files WHERE filepath = ? AND file_hash = ?", (filepath, file_hash))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_or_create_chunk_id(self, cursor: sqlite3.Cursor, chunk_hash: str, chunk_content: str) -> int:
        """Gets chunk_id for a hash, or creates embedding and info record if new."""
        # 1. Check if chunk_hash exists
        cursor.execute("SELECT chunk_id FROM chunk_info WHERE chunk_hash = ?", (chunk_hash,))
        result = cursor.fetchone()

        if result:
            # logging.debug(f"Chunk hash {chunk_hash[:8]}... found, ID: {result[0]}")
            return result[0] # Return existing chunk_id
        else:
            # logging.debug(f"Chunk hash {chunk_hash[:8]}... not found. Creating new entry.")
            # 2. Insert into chunk_info
            cursor.execute("INSERT INTO chunk_info (chunk_hash) VALUES (?)", (chunk_hash,))
            chunk_id = cursor.lastrowid
            # logging.debug(f"Inserted into chunk_info, new chunk_id: {chunk_id}")

            # 3. Generate embedding
            embedding = generate_embedding(chunk_content)
            embedding_blob = embedding_to_blob(embedding)

            # 4. Insert into chunk_embeddings virtual table
            # Syntax for vec0 insert often involves specifying columns including hidden ones
            cursor.execute(
                "INSERT INTO chunk_embeddings (rowid, embedding, chunk_id) VALUES (?, ?, ?)",
                (chunk_id, embedding_blob, chunk_id) # Using chunk_id as rowid for simplicity if allowed, otherwise let vec0 handle rowid
            )
            # Check vec0 docs: rowid might be managed automatically. Let's try default rowid management
            # cursor.execute(
            #     "INSERT INTO chunk_embeddings (embedding, chunk_id) VALUES (?, ?)",
            #     (embedding_blob, chunk_id)
            # )
            # embedding_rowid = cursor.lastrowid # If rowid is automatic

            # logging.debug(f"Inserted embedding for chunk_id {chunk_id} into vec0 table.")
            return chunk_id


    # --- Transactional Insertion ---
    def insert_file_and_chunks_transactional(self, filepath: str, file_hash: str, chunks: List[Dict]):
        """
        Inserts file and chunk data within a single database transaction using the new schema.
        """
        conn, cursor = None, None
        try:
            conn, cursor = self._connect()
            # Start transaction
            cursor.execute("BEGIN")
            logging.info(f"Starting transaction to insert file: {filepath}")

            # 1. Check if file version already exists
            existing_file_id = self.get_file_id(cursor, filepath, file_hash)
            if existing_file_id:
                logging.info(f"File version already exists (file_id: {existing_file_id}), skipping insertion. Rolling back.")
                cursor.execute("ROLLBACK") # Nothing to insert, rollback immediately
                return True # Indicate operation handled (skipped)

            # 2. Insert file record
            last_access_time = int(time.time())
            cursor.execute(
                "INSERT INTO files (filepath, file_hash, last_access_time) VALUES (?, ?, ?)",
                (filepath, file_hash, last_access_time)
            )
            file_id = cursor.lastrowid
            logging.debug(f"Inserted file record, new file_id: {file_id}")

            # 3. Process chunks
            processed_chunks = 0
            for chunk in chunks:
                chunk_content = chunk["content"]
                chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()

                # Get existing chunk_id or create new chunk_info and embedding entry
                chunk_id = self.get_or_create_chunk_id(cursor, chunk_hash, chunk_content)

                # Insert chunk metadata
                cursor.execute(
                    """
                    INSERT INTO chunk_metadata (file_id, chunk_id, start_offset, length)
                    VALUES (?, ?, ?, ?)
                    """,
                    (file_id, chunk_id, chunk["start"], chunk["length"])
                )
                processed_chunks += 1

            # Commit transaction
            cursor.execute("COMMIT")
            logging.info(f"Successfully committed transaction for file: {filepath} (file_id: {file_id}). "
                         f"Processed {processed_chunks} chunks.")
            return True

        except (sqlite3.Error, ValueError, KeyError, RuntimeError) as e:
            logging.error(f"Error during file/chunk insertion transaction for {filepath}: {e}", exc_info=True)
            if conn and cursor:
                try:
                    cursor.execute("ROLLBACK")
                    logging.warning(f"Transaction rolled back for file: {filepath}")
                except sqlite3.Error as rb_err:
                    logging.error(f"Error during transaction rollback for {filepath}: {rb_err}")
            return False
        finally:
            if conn:
                self._disconnect(conn)

    def update_last_access_time_by_id(self, file_ids: List[int]):
        """Updates the last access time for a list of file IDs."""
        if not file_ids:
            logging.warning("update_last_access_time_by_id called with empty list.")
            return

        conn, cursor = None, None
        try:
            conn, cursor = self._connect()
            placeholders = ', '.join(['?'] * len(file_ids))
            current_time = int(time.time())
            # Use transaction for potentially multiple updates
            cursor.execute("BEGIN")
            cursor.execute(
                f"UPDATE files SET last_access_time = ? WHERE file_id IN ({placeholders})",
                (current_time, *file_ids)
            )
            updated_count = cursor.rowcount
            cursor.execute("COMMIT")
            logging.info(f"Updated last access time for {updated_count} file IDs.")
        except (sqlite3.Error, ValueError) as e:
            logging.error(f"Error updating last access time by ID: {e}")
            if conn and cursor:
                try: cursor.execute("ROLLBACK")
                except: pass # Ignore rollback errors
        finally:
            if conn:
                self._disconnect(conn)

    def find_closest_chunks(self, target_file_ids: List[int], query_embedding: List[float], num_results: int) -> List[Tuple[str, int, int, float]]:
        """
        Finds the closest N chunks using vec0 MATCH, filtering by target file_ids.
        """
        if not self.extension_loaded:
            logging.error("Vector extension not loaded. Cannot perform similarity search.")
            return []
        if not target_file_ids:
            logging.warning("find_closest_chunks called with empty target_file_ids list.")
            return []

        conn, cursor = None, None
        try:
            conn, cursor = self._connect()
            query_embedding_blob = embedding_to_blob(query_embedding)

            # 1. Perform ANN search using vec0 MATCH
            # The result gives rowid (which we assume is chunk_id) and distance
            # We might need to adjust the query based on how vec0 handles hidden columns and joins
            # This query assumes vec0 returns the hidden `chunk_id` along with distance
            search_sql = f"""
                SELECT
                    ce.chunk_id, -- Retrieve the hidden chunk_id
                    ce.distance
                FROM chunk_embeddings AS ce
                WHERE ce.embedding MATCH ?
                ORDER BY ce.distance
                LIMIT ?
            """
            # The 'vector' needs to be passed as parameter to MATCH
            cursor.execute(search_sql, (query_embedding_blob, num_results * 5)) # Fetch more initially for filtering
            initial_matches = cursor.fetchall() # List of (chunk_id, distance)

            if not initial_matches:
                logging.info("Initial vector search returned no matches.")
                return []

            # Extract chunk_ids and distances from initial matches
            candidate_chunk_ids = [match[0] for match in initial_matches]
            distances = {match[0]: match[1] for match in initial_matches} # Map chunk_id to distance

            # 2. Filter candidates by target file_ids and retrieve metadata
            placeholders = ', '.join(['?'] * len(target_file_ids))
            chunk_id_placeholders = ', '.join(['?'] * len(candidate_chunk_ids))

            filter_sql = f"""
                SELECT
                    f.filepath,
                    cm.start_offset,
                    cm.length,
                    cm.chunk_id -- Include chunk_id to look up distance
                FROM chunk_metadata AS cm
                JOIN files AS f ON cm.file_id = f.file_id
                WHERE cm.chunk_id IN ({chunk_id_placeholders})
                  AND cm.file_id IN ({placeholders})
            """

            params = (*candidate_chunk_ids, *target_file_ids)
            cursor.execute(filter_sql, params)
            filtered_results = cursor.fetchall()

            # 3. Combine with distances and limit
            final_results = []
            for filepath, start, length, chunk_id in filtered_results:
                distance = distances.get(chunk_id)
                if distance is not None: # Should always be found if logic is correct
                    final_results.append((filepath, start, length, distance))

            # Sort by distance again as filtering might change order, then take top N
            final_results.sort(key=lambda x: x[3])
            limited_results = final_results[:num_results]

            logging.info(f"Found {len(limited_results)} similar chunks after filtering for {len(target_file_ids)} files.")
            return limited_results

        except sqlite3.Error as e:
            if "no such function: vector_to_blob" in str(e) or "no such function: vec_search" in str(e) or "MATCH" in str(e):
                 logging.error(f"Vector search function error (is vec.so loaded correctly and schema correct?): {e}")
            else:
                logging.error(f"Error finding closest chunks: {e}", exc_info=True)
            return []
        except (ValueError, RuntimeError) as e:
             logging.error(f"Error during closest chunk search: {e}", exc_info=True)
             return []
        finally:
            if conn:
                self._disconnect(conn)


class App:
    """Main application class."""
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        # Ensure tables exist at startup - crucial!
        self.db_manager.create_tables()

    def process_file_data(self, file_data: Dict):
        """Processes incoming file data using transactional insertion."""
        try:
            filepath = file_data["filepath"]
            file_hash = file_data["file_hash"]
            chunks = file_data["chunks"] # Assuming chunks have 'start', 'length', 'content'

            # The transactional method now handles the check internally
            success = self.db_manager.insert_file_and_chunks_transactional(
                filepath, file_hash, chunks
            )
            if not success:
                logging.error(f"Failed to process and insert file data for: {filepath}")

        except KeyError as e:
            logging.error(f"Missing key in file_data: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in process_file_data: {e}", exc_info=True)

    def find_similar_chunks(self, file_identifiers: List[Dict[str, str]], query_chunk_content: str, num_results: int) -> List[Tuple[str, int, int, float]]:
        """Finds similar chunks based on query content across specified files."""
        if not file_identifiers:
             logging.warning("find_similar_chunks called with empty file list.")
             return []
        if not query_chunk_content:
             logging.warning("find_similar_chunks called with empty query chunk content.")
             return []

        conn, cursor = None, None
        target_file_ids = []
        try:
            # 1. Resolve file identifiers to file_ids
            conn, cursor = self.db_manager._connect() # Use internal connect for lookup
            cursor.execute("BEGIN") # Transaction for lookups + updates
            
            filepaths_hashes = [(item['filepath'], item['file_hash']) for item in file_identifiers]
            placeholders = ', '.join(['(?, ?)'] * len(filepaths_hashes))
            
            # Build query dynamically to fetch file_ids for multiple (path, hash) pairs
            # This is a bit tricky in SQLite, might need multiple queries or specific syntax
            # Simpler approach: loop and fetch individually (less efficient for many files)
            for fp, fh in filepaths_hashes:
                 file_id = self.db_manager.get_file_id(cursor, fp, fh)
                 if file_id:
                      target_file_ids.append(file_id)
                 else:
                      logging.warning(f"File not found in DB, skipping search for: {fp} ({fh[:8]}...)")
            
            if not target_file_ids:
                 logging.warning("No valid file_ids found for the given identifiers.")
                 cursor.execute("ROLLBACK") # Rollback as nothing was updated
                 return []

            # 2. Update access time
            self.db_manager.update_last_access_time_by_id(target_file_ids) # This handles its own transaction/connection

            cursor.execute("COMMIT") # Commit the lookup transaction if needed (though reads don't strictly need it)

            # 3. Generate query embedding
            logging.debug(f"Generating embedding for query chunk (length: {len(query_chunk_content)})...")
            query_embedding = generate_embedding(query_chunk_content)

            # 4. Perform search using file_ids
            logging.info(f"Searching for {num_results} chunks similar to query in {len(target_file_ids)} files.")
            results = self.db_manager.find_closest_chunks(target_file_ids, query_embedding, num_results)
            return results

        except KeyError as e:
            logging.error(f"Missing 'filepath' or 'file_hash' key in file_identifiers item: {e}")
            if conn and cursor:
                try:
                    cursor.execute("ROLLBACK")
                except:
                    pass
            return []
        except (sqlite3.Error, ValueError, RuntimeError) as e:
             logging.error(f"An unexpected error occurred in find_similar_chunks: {e}", exc_info=True)
             if conn and cursor: try: cursor.execute("ROLLBACK") catch: pass
             return []
        finally:
            # Ensure connection used for lookup is closed
            if conn:
                self.db_manager._disconnect(conn)


def main():
    """Main function to demonstrate running the application."""
    logging.info("Application starting...")
    # Initialize the database manager
    db_manager = DBManager(db_type="sqlite", db_name="file_chunks_v3_vec0.db", embedding_dim=EMBEDDING_DIMENSION)
    app = App(db_manager)

    # --- Example Data ---
    file_data_1 = {
        "filepath": "/path/to/my/document_a.txt", "file_hash": hashlib.sha256(b"content_a v1").hexdigest(),
        "chunks": [
            {"start": 0, "length": 35, "content": "This is the first part of document A."},
            {"start": 35, "length": 40, "content": "This is the second part, containing unique words."},
            {"start": 75, "length": 25, "content": "The final section of A."} ]}
    file_data_1_v2 = { # Same path, different hash
        "filepath": "/path/to/my/document_a.txt", "file_hash": hashlib.sha256(b"content_a v2").hexdigest(),
        "chunks": [
            {"start": 0, "length": 35, "content": "This is the initial part of document A version 2."}, # Changed content
            {"start": 35, "length": 40, "content": "This is the second part, containing unique words."}, # Same chunk content as v1
            {"start": 75, "length": 30, "content": "The last section of A v2."} ]} # Changed content
    file_data_2 = {
        "filepath": "/path/to/another/document_b.log", "file_hash": hashlib.sha256(b"content_b").hexdigest(),
        "chunks": [
            {"start": 0, "length": 40, "content": "Log entry: System started successfully."},
            {"start": 40, "length": 45, "content": "Log entry: User 'admin' logged in."},
            {"start": 85, "length": 50, "content": "Warning: Disk space low on /var/log."},
            {"start": 135, "length": 30, "content": "Another part of document B."} ]}

    # --- Process Files ---
    logging.info("\n--- Processing File 1 (v1) ---")
    app.process_file_data(file_data_1)
    logging.info("\n--- Processing File 1 (v1) Again (should skip) ---")
    app.process_file_data(file_data_1) # Should be skipped
    logging.info("\n--- Processing File 1 (v2) ---")
    app.process_file_data(file_data_1_v2) # Should be added
    logging.info("\n--- Processing File 2 ---")
    app.process_file_data(file_data_2)

    # --- Search Examples ---
    logging.info("\n--- Searching for 'second part unique' in File 1 v1 ---")
    query1 = "second part unique"
    files_to_search_1 = [{"filepath": file_data_1["filepath"], "file_hash": file_data_1["file_hash"]}]
    results1 = app.find_similar_chunks(files_to_search_1, query1, 3)
    print("Results (File 1 v1):")
    for fp, start, length, dist in results1: print(f"  File: {fp}, Start: {start}, Len: {length}, Dist: {dist:.4f}")

    logging.info("\n--- Searching for 'second part unique' in File 1 v2 ---")
    files_to_search_1v2 = [{"filepath": file_data_1_v2["filepath"], "file_hash": file_data_1_v2["file_hash"]}]
    results1v2 = app.find_similar_chunks(files_to_search_1v2, query1, 3)
    print("Results (File 1 v2):") # Should find the identical chunk
    for fp, start, length, dist in results1v2: print(f"  File: {fp}, Start: {start}, Len: {length}, Dist: {dist:.4f}")

    logging.info("\n--- Searching for 'document A version' in all versions of File 1 ---")
    query2 = "document A version"
    files_to_search_all_v1 = [
        {"filepath": file_data_1["filepath"], "file_hash": file_data_1["file_hash"]},
        {"filepath": file_data_1_v2["filepath"], "file_hash": file_data_1_v2["file_hash"]}
    ]
    results2 = app.find_similar_chunks(files_to_search_all_v1, query2, 5)
    print("Results (All File 1 versions):")
    for fp, start, length, dist in results2: print(f"  File: {fp}, Start: {start}, Len: {length}, Dist: {dist:.4f}")


    logging.info("\n--- Searching for 'admin log' in File 2 ---")
    query3 = "admin log"
    files_to_search_2 = [{"filepath": file_data_2["filepath"], "file_hash": file_data_2["file_hash"]}]
    results3 = app.find_similar_chunks(files_to_search_2, query3, 3)
    print("Results (File 2):")
    for fp, start, length, dist in results3: print(f"  File: {fp}, Start: {start}, Len: {length}, Dist: {dist:.4f}")


    logging.info("\nApplication finished.")


if __name__ == "__main__":
    main()
