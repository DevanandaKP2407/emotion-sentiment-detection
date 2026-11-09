# Beyond Sentiment — Emotion Detection in Text

A small NLP application that predicts the sentiment and likely emotion expressed in a sentence.

## Features

- Sentiment classification: Positive, Negative and Neutral.
- Emotion classification: Sadness, Joy, Love, Anger, Fear and Surprise.
- Confidence scores for both predictions.
- Evaluation using accuracy and classification reports.
- Streamlit interface.

## Technology

- Python
- Scikit-learn
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Logistic Regression
- Hugging Face `dair-ai/emotion` public dataset
- Streamlit

## Project structure

```text
beyond_sentiment/
├── app.py
├── model_utils.py
├── train_models.py
├── requirements.txt
├── README.md
├── evaluation.txt          # generated after training
└── models/
    ├── emotion_model.joblib  # generated after training
    └── sentiment_model.joblib
```

## Setup and run

Python 3.10 or newer is recommended.

### 1. Open the project folder

```bash
cd beyond_sentiment
```

### 2. Create and activate a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and evaluate the models

```bash
python train_models.py
```

This downloads the public dataset, trains both classifiers, evaluates them on the test split, and saves the models in `models/`. Internet access is needed for the first training run. Training time depends on your computer and connection.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal. Open it in your browser.

## How it works

### 1. TF-IDF

TF-IDF converts text into numerical features. It gives higher importance to words that are useful in a document while reducing the importance of words that appear very frequently across documents.

### 2. Logistic Regression

Logistic Regression learns patterns from labelled examples and predicts a class. `predict_proba()` provides a score for each possible class.

### 3. Two classifiers

The emotion classifier predicts one of six emotion labels. The sentiment classifier predicts positive, negative or neutral.

### 4. Sentiment label mapping

The source dataset contains emotion labels, not separate sentiment labels. For this project, sentiment is derived using this simple mapping:

| Emotion labels | Derived sentiment |
|---|---|
| Joy, Love | Positive |
| Sadness, Anger, Fear | Negative |
| Surprise | Neutral |

This is a project-level simplification. Surprise can be positive or negative depending on context, and love is not always equivalent to positive sentiment.

## Evaluation

The training script evaluates both classifiers on the dataset's test split and reports accuracy and per-class precision, recall and F1-score. The actual measured values are written to `evaluation.txt` after you run the training script.

Do not claim an accuracy value before running the evaluation on your machine.

## Limitations

- The dataset contains short text examples and may not represent every writing style.
- Sarcasm can confuse the model. For example, "That's just wonderful..." may express disappointment.
- The model does not understand the full context of a conversation.
- A sentence can contain more than one emotion, but this model predicts one main emotion.
- Confidence is the model's estimated probability, not a guarantee that the prediction is correct.
- The sentiment labels are derived from emotion labels, so the sentiment evaluation measures this simplified mapping rather than an independently annotated sentiment dataset.

## Viva preparation

**What is NLP?**  
Natural Language Processing is a field of AI that helps computers process and analyse human language.

**What is TF-IDF?**  
A method that converts text into numerical values based on how important words are in a document compared with the dataset.

**Why Logistic Regression?**  
It is relatively simple, fast to train, works well with sparse TF-IDF features and can provide class probabilities.

**What is accuracy?**  
The number of correct predictions divided by the total number of predictions.

**What is a confidence score?**  
The probability-like score assigned by the classifier to a predicted class. It should not be treated as certainty.

**Why can emotion detection fail?**  
Sarcasm, missing context, ambiguous wording, mixed emotions and dataset limitations can lead to incorrect predictions.

## Dataset

`dair-ai/emotion` on Hugging Face Datasets. The dataset provides text examples labelled with six emotions. See the dataset page for its licence and usage terms before redistribution.
