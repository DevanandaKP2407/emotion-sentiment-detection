# Emotion & Sentiment Detection System

A Machine Learning-based application that analyses text and predicts the underlying emotion and sentiment using **TF-IDF and Logistic Regression**.

The application identifies six emotions — Joy, Sadness, Love, Anger, Fear and Surprise — and classifies the overall sentiment as Positive, Negative or Neutral.

## Overview

Understanding emotions and sentiment in text is useful in applications such as customer feedback analysis, social media monitoring and opinion analysis.

This project uses Natural Language Processing (NLP) and Machine Learning to analyse user-provided text and identify its emotional tone.

The application provides:
- Emotion classification across six categories.
- Sentiment classification into Positive, Negative and Neutral.
- Probability scores for each emotion.
- A simple interactive interface built using Streamlit.
- Model evaluation using standard classification metrics.

## Features

- **Emotion Detection:** Identifies Joy, Sadness, Love, Anger, Fear and Surprise.
- **Sentiment Analysis:** Classifies text as Positive, Negative or Neutral.
- **Probability Scores:** Displays the model's predicted probability for each emotion.
- **Model Evaluation:** Includes accuracy, precision, recall, F1-score and confusion matrices.
- **Interactive Interface:** Allows users to enter text and view predictions instantly.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Scikit-learn | Machine Learning algorithms |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification |
| Hugging Face Datasets | Dataset loading |
| Joblib | Saving and loading trained models |

## Machine Learning Approach

The project uses a TF-IDF-based text classification approach with Logistic Regression.

### 1. Dataset

The model is trained using the publicly available [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from Hugging Face.

The dataset contains text samples labelled with six emotions:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

The provided training and testing splits are used for model development and evaluation.

### 2. Text Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical features that can be processed by a machine learning model.

The implementation uses:
- Unigrams and bigrams.
- Lowercase text processing.
- A maximum of 30,000 features.
- Sublinear term frequency scaling.

### 3. Classification Algorithm

**Logistic Regression** is used for both emotion and sentiment classification.

Hyperparameter tuning is performed using GridSearchCV with three-fold stratified cross-validation. The best model is selected based on macro F1-score.

### 4. Sentiment Classification

The original dataset contains emotion labels rather than independent sentiment labels. Therefore, sentiment categories are derived using the following mapping:

| Emotion | Sentiment |
|---|---|
| Joy | Positive |
| Love | Positive |
| Sadness | Negative |
| Anger | Negative |
| Fear | Negative |
| Surprise | Neutral |

This is a simplified mapping used for the project and does not represent every possible interpretation of sentiment.

## Project Structure

```text
emotion-sentiment-detection/
│
├── app.py
├── model_utils.py
├── train_models.py
├── requirements.txt
├── README.md
├── evaluation.txt
│
└── models/
    ├── emotion_model.joblib
    └── sentiment_model.joblib
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Internet connection for downloading the dataset during training

### 1. Clone the Repository

```bash
git clone https://github.com/DevanandaKP2407/emotion-sentiment-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd emotion-sentiment-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Models

If the trained model files are not already present in the `models/` directory, execute:

```bash
python train_models.py
```

This downloads the dataset, trains both classifiers, performs hyperparameter tuning, evaluates the models and saves the trained models.

The evaluation results are saved in `evaluation.txt`.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified samples.
- **Precision:** Measures how many predicted samples belong to the correct class.
- **Recall:** Measures how many actual samples are correctly identified.
- **F1-score:** Provides a balance between precision and recall.
- **Confusion Matrix:** Shows the correct and incorrect predictions across different classes.

Actual evaluation results are generated after training and are available in `evaluation.txt`.

## Limitations

- TF-IDF does not fully understand context, sarcasm or complex sentence structures.
- Emotion predictions depend on the patterns and labels present in the training dataset.
- Some emotions may have overlapping language patterns.
- Sentiment labels are derived from emotion labels, which may oversimplify the relationship between emotion and sentiment.
- Predicted probabilities are model estimates and should not be interpreted as guaranteed measures of a person's emotional state.

## Future Improvements

- Experiment with transformer-based models such as BERT or DistilBERT.
- Explore larger and more diverse datasets.
- Improve sentiment classification using independently labelled sentiment data.
- Enhance contextual understanding and handling of sarcasm.
- Improve prediction reliability through probability calibration.
- Add support for multilingual text analysis.

## Author

**Devananda K P**

B.Tech Computer Science and Engineering

College of Engineering Trivandrum (CET)

[GitHub Profile](https://github.com/DevanandaKP2407)

---

*Developed as a Machine Learning project exploring Natural Language Processing, text classification and emotion detection.*
