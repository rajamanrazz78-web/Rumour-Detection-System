from flask import Flask, render_template, request
import pickle
import re
import string
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

FACT_CHECK_API_KEY = "YOUR_API_KEY_HERE"
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

app = Flask(__name__)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    punctuation_to_remove = string.punctuation.replace("-", "")
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    words = word_tokenize(text)
    english_stopwords = set(stopwords.words("english"))
    preserved_words = {"5g", "covid-19", "nasa", "rover", "perseverance"}
    filtered_words = [
        word for word in words
        if word not in english_stopwords or word in preserved_words
    ]
    return " ".join(filtered_words)


def fact_check_api(query):
    try:
        params = {"query": query, "key": FACT_CHECK_API_KEY}
        response = requests.get(FACT_CHECK_API_URL, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "claims" in data and len(data["claims"]) > 0:
                fact_results = []
                for claim in data["claims"]:
                    fact_results.append({
                        "text": claim["text"],
                        "claimant": claim.get("claimant", "Unknown"),
                        "rating": claim["claimReview"][0]["textualRating"]
                    })
                return fact_results
    except Exception:
        pass
    return [{"text": "No fact-check results found", "claimant": "N/A", "rating": "N/A"}]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["text"]

    if not user_text or len(user_text.strip()) < 5:
        return render_template(
            "result.html",
            input_text=user_text,
            model_prediction="Invalid Input ❌",
            confidence="0%",
            fact_check_results=[],
            warning="",
            final_decision="Please enter meaningful text."
        )

    if len(user_text.strip().split()) < 30:
        warning = "⚠️ For best results please enter at least 30 words."
    else:
        warning = ""

    cleaned_text = clean_text(user_text)

    if not cleaned_text.strip():
        return render_template(
            "result.html",
            input_text=user_text,
            model_prediction="Invalid Input ❌",
            confidence="0%",
            fact_check_results=[],
            warning="",
            final_decision="Input not meaningful after processing."
        )

    transformed_text = tfidf_vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(transformed_text)
    rumor_prob = probabilities[0][1]
    confidence = round(max(probabilities[0]) * 100, 2)
    model_result = "Rumor 🚫" if rumor_prob >= 0.80 else "True News ✅"

    fact_results = fact_check_api(user_text)

    if fact_results and fact_results[0]["rating"].lower() != "n/a":
        rating = fact_results[0]["rating"].lower()
    else:
        rating = "n/a"

    if rating in ["false", "misleading"]:
        final_decision = "Verified Rumor 🚫"
    elif rating in ["true", "mostly true"]:
        final_decision = "Verified True News ✅"
    elif rating == "n/a":
        final_decision = (
            f"{model_result} (Confidence: {confidence}%) - Manual review recommended"
        )
    else:
        final_decision = (
            f"{model_result} (Confidence: {confidence}%) - Further review needed"
        )

    return render_template(
        "result.html",
        input_text=user_text,
        model_prediction=model_result,
        confidence=f"{confidence}%",
        fact_check_results=fact_results,
        warning=warning,
        final_decision=final_decision
    )


if __name__ == "__main__":
    app.run(debug=True)
