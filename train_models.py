
import os
import joblib

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- CONFIGURATION ----------------

DATASET_NAME = "dair-ai/emotion"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

EMOTION_NAMES = [
    "sadness", "joy", "love",
    "anger", "fear", "surprise"
]

SENTIMENT_NAMES = ["negative", "neutral", "positive"]


# ---------------- SENTIMENT MAPPING ----------------

def sentiment_from_emotion(label):

    if label in ("joy", "love"):
        return "positive"

    elif label in ("sadness", "anger", "fear"):
        return "negative"

    else:
        return "neutral"


# ---------------- MODEL CREATION ----------------

def create_pipeline():

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True
        )),

        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])


# ---------------- MODEL TRAINING ----------------

def train_model(texts, labels, model_name):

    print(f"\nTraining {model_name} model...")

    model = create_pipeline()

    parameters = {
        "classifier__C": [0.5, 1.0, 2.0],
        "classifier__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(texts, labels)

    print(f"\nBest parameters for {model_name}:")
    print(search.best_params_)

    print(f"Best cross-validation F1-score: {search.best_score_:.4f}")

    return search.best_estimator_


# ---------------- EVALUATION ----------------

def evaluate_model(model, texts, labels, class_names, model_name):

    predictions = model.predict(texts)

    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=class_names
    )

    print(f"\n========== {model_name} EVALUATION ==========")

    print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    return accuracy, report, matrix


# ---------------- MAIN PROGRAM ----------------

def main():

    print("Downloading the public emotion dataset...")

    dataset = load_dataset(DATASET_NAME)

    train_texts = dataset["train"]["text"]

    train_labels = [
        EMOTION_NAMES[int(x)]
        for x in dataset["train"]["label"]
    ]

    test_texts = dataset["test"]["text"]

    test_labels = [
        EMOTION_NAMES[int(x)]
        for x in dataset["test"]["label"]
    ]

    print(f"\nTraining examples: {len(train_texts)}")
    print(f"Testing examples: {len(test_texts)}")

    # Train emotion classifier

    emotion_model = train_model(
        train_texts,
        train_labels,
        "Emotion"
    )

    # Convert emotion labels into sentiment labels

    train_sentiments = [
        sentiment_from_emotion(x)
        for x in train_labels
    ]

    test_sentiments = [
        sentiment_from_emotion(x)
        for x in test_labels
    ]

    # Train sentiment classifier

    sentiment_model = train_model(
        train_texts,
        train_sentiments,
        "Sentiment"
    )

    # Evaluate both models

    emotion_accuracy, emotion_report, emotion_matrix = evaluate_model(
        emotion_model,
        test_texts,
        test_labels,
        EMOTION_NAMES,
        "EMOTION"
    )

    sentiment_accuracy, sentiment_report, sentiment_matrix = evaluate_model(
        sentiment_model,
        test_texts,
        test_sentiments,
        SENTIMENT_NAMES,
        "SENTIMENT"
    )

    # Save trained models

    joblib.dump(
        emotion_model,
        os.path.join(MODEL_DIR, "emotion_model.joblib")
    )

    joblib.dump(
        sentiment_model,
        os.path.join(MODEL_DIR, "sentiment_model.joblib")
    )

    # Save evaluation results

    evaluation_path = os.path.join(BASE_DIR, "evaluation.txt")

    with open(evaluation_path, "w", encoding="utf-8") as f:

        f.write("MODEL EVALUATION RESULTS\n")
        f.write("========================\n\n")

        f.write("Dataset: dair-ai/emotion\n")
        f.write(f"Training examples: {len(train_texts)}\n")
        f.write(f"Testing examples: {len(test_texts)}\n\n")

        f.write("EMOTION MODEL\n")
        f.write(f"Accuracy: {emotion_accuracy:.4f}\n\n")
        f.write(emotion_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(emotion_matrix))

        f.write("\n\nSENTIMENT MODEL\n")
        f.write(f"Accuracy: {sentiment_accuracy:.4f}\n\n")
        f.write(sentiment_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(sentiment_matrix))

    print("\n================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("================================")

    print("\nModels saved in the models/ folder.")
    print("Evaluation saved in evaluation.txt")


if __name__ == "__main__":
    main()
