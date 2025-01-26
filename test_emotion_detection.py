from EmotionDetection import  emotion_detection
import unittest

class TestEmotionDetector(unittest.TestCase):
    def test_sentiment_analyzer(self):
        tests = [
            ("I am glad this happened", "joy"),
            ("I am really mad about this", "anger"),
            ("I feel disgusted just hearing about this", "disgust"),
            ("I am so sad about this", "sadness"),
            ("I am really afraid that this will happen", "fear"),
            ("", None)
        ]
        for testCase in tests:
            (msg, expected) = testCase
            res = emotion_detection.emotion_detector(msg)
            self.assertEqual(res["dominant_emotion"], expected)

unittest.main()
    