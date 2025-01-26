"""Web UI for EmotionDetection"""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detection

app = Flask(__name__)

@app.get('/emotionDetector')
def detect_emotion():
    """Calls EmotionDetection API"""
    text = request.args["textToAnalyze"]
    res = emotion_detection.emotion_detector(text)
    if res['dominant_emotion'] is None:
        return "Invalid text! Please try again!"
    return res

@app.get("/")
def home():
    """Renders the home page"""
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
