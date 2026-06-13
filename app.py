import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECT_PATH = BASE_DIR / "vectorizer.pkl"
DATA_PATH = BASE_DIR / "email.csv"

st.set_page_config(page_title="Spam/Ham Email Classifier", layout="centered")

@st.cache_resource
def load_saved_artifacts():
    """Try to load saved model and vectorizer from the app directory (script location).
    Returns (model, vectorizer) or (None, None)."""
    # Use absolute paths (Path objects) and convert to str for joblib
    model_path = str(MODEL_PATH)
    vect_path = str(VECT_PATH)
    # Show minimal debug in the app via return values; UI will display messages
    if Path(model_path).exists() and Path(vect_path).exists():
        try:
            model = joblib.load(model_path)
            vect = joblib.load(vect_path)
            return model, vect
        except Exception as e:
            # Save load error to session_state so UI can display it
            st.session_state.setdefault('load_error', str(e))
            return None, None
    else:
        st.session_state.setdefault('load_error', None)
        return None, None


def train_and_save_model(verbose=True):
    """Train model from DATA_PATH and save model & vectorizer to disk.
    Returns (model, vectorizer, stats_dict)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file '{DATA_PATH}' not found in working directory.")

    df = pd.read_csv(DATA_PATH)
    # Keep original rows but normalize missing
    mail_data = df.where(pd.notnull(df), "")

    # Map labels safely and drop invalids
    mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
    mail_data = mail_data[mail_data['Category'].notnull()].copy()
    mail_data['Category'] = mail_data['Category'].astype(int)

    X = mail_data['Message'].fillna('').astype('str')
    Y = mail_data['Category']

    # Stratified split to keep class proportions
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3, stratify=Y)

    vect = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vect.fit_transform(X_train)
    X_test_features = vect.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, Y_train)

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)

    stats = {
        'train_accuracy': float(accuracy_score(Y_train, model.predict(X_train_features))),
        'test_accuracy': float(accuracy_score(Y_test, model.predict(X_test_features)))
    }

    if verbose:
        st.success('Model trained and saved to disk.')
        st.write('Train accuracy:', stats['train_accuracy'])
        st.write('Test accuracy:', stats['test_accuracy'])

    return model, vect, stats


def predict_text(model, vect, texts):
    """Return list of (label, prob) for each text. label: 'Spam' or 'Ham'."""
    X = vect.transform(texts)
    preds = model.predict(X)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
    results = []
    for i, p in enumerate(preds):
        label = 'Spam' if p == 0 else 'Ham'
        prob = None
        if proba is not None:
            # Use probability of predicted class
            prob = float(proba[i, p])
        results.append((label, prob))
    return results


def main():
    st.title('Spam Mail Prediction')

    # Load saved artifacts (model & vectorizer)
    model, vect = load_saved_artifacts()

    default_sample = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
    input_text = st.text_area('Email message', value=default_sample, height=150)

    # Single Predict button only
    if st.button('Predict'):
        if model is None or vect is None:
            st.error('Model not available. Place model.pkl and vectorizer.pkl in the application folder or train the model first.')
        else:
            with st.spinner('Predicting...'):
                try:
                    label, prob = predict_text(model, vect, [input_text])[0]
                    if label == 'Spam':
                        st.error(f'Prediction: {label} {f"(Confidence: {prob:.3f})" if prob is not None else ""}')
                    else:
                        st.success(f'Prediction: {label} {f"(Confidence: {prob:.3f})" if prob is not None else ""}')
                except Exception as e:
                    st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
