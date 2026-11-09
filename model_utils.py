import os
import joblib

def load_models(model_dir):
    emotion_path = os.path.join(model_dir, "emotion_model.joblib")
    sentiment_path = os.path.join(model_dir, "sentiment_model.joblib")

    if not (os.path.exists(emotion_path) and os.path.exists(sentiment_path)):
        return None, None, "Models not trained"

    try:
        emotion_model = joblib.load(emotion_path)
        sentiment_model = joblib.load(sentiment_path)
        return emotion_model, sentiment_model, None
    except Exception as exc:
        return None, None, str(exc)


def predict_text(text, emotion_model, sentiment_model):
    emotion_probs = emotion_model.predict_proba([text])[0]
    emotion_classes = emotion_model.classes_
    emotion_index = emotion_probs.argmax()
    emotion = str(emotion_classes[emotion_index])

    sentiment_probs = sentiment_model.predict_proba([text])[0]
    sentiment_classes = sentiment_model.classes_
    sentiment_index = sentiment_probs.argmax()
    sentiment = str(sentiment_classes[sentiment_index])

    return {
        "emotion": emotion,
        "emotion_confidence": float(emotion_probs[emotion_index]),
        "emotion_scores": {
            str(label): float(score)
            for label, score in zip(emotion_classes, emotion_probs)
        },
        "sentiment": sentiment,
        "sentiment_confidence": float(sentiment_probs[sentiment_index]),
    }
