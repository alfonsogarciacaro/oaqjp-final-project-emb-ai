"""Sentiment analysis using Watson NLP Library"""

import json
import requests

def emotion_detector(text_to_analyze):
    """
    Use Watson NLP library to analize the sentiment
    of the argument text_to_analyze
    """

    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    json_body = {"raw_document": {"text": text_to_analyze } }
    res = requests.post(url, headers=headers, json=json_body)
    formatted = json.loads(res.text)
    # Just take the first prediction for now
    prediction = formatted["emotionPredictions"][0]
    emotions = ["anger", "disgust", "fear", "joy", "sadness"]
    scores = [prediction["emotion"][emotion] for emotion in emotions]
    final_res = { emotion: scores[i] for i, emotion in enumerate(emotions)}
    final_res['dominant_emotion'] = emotions[scores.index(max(scores))]
    return final_res
