from flask import Flask, render_template, request, redirect, url_for, session
from transformers import TFRobertaForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # required for session

# Load model and tokenizer
MODEL_PATH = "./sentiment_model"
model = TFRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    logits = model(inputs).logits
    prob = tf.nn.sigmoid(logits)[0].numpy()[0]
    return "Positive" if prob > 0.5 else "Negative"

# Translation function
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text  # fallback: return original

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        original_review = request.form["review"]
        translated_review = translate_to_english(original_review)
        sentiment = predict_sentiment(translated_review)

        session["review"] = original_review
        session["sentiment"] = sentiment

        return redirect(url_for("index"))

    return render_template(
        "index.html",
        review=session.pop("review", ""),
        sentiment=session.pop("sentiment", None)
    )

if __name__ == "__main__":
    app.run(debug=True)
