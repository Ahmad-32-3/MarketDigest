import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT sentiment analysis pipeline once
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)


df = pd.read_json("selenium_yahoo_finance.jsonl", lines=True)

def classify_article(row):
    # Prefer full content, fallback to summary, then title
    text = row.get("content_full") or row.get("summary") or row.get("title")
    # Limit length for performance if needed
    if not text or len(text.strip()) < 20:
        return "neutral"

    # Run sentiment model
    try:
        result = sentiment_pipeline(text[:512])[0]  # FinBERT is best with up to 512 tokens
        label = result["label"].lower()  # 'positive', 'neutral', or 'negative'
        score = result["score"]

        # Refine with some rule-based logic for “Best to Buy/Avoid”
        text_lower = text.lower()
        if label == "positive" and ("upgrade" in text_lower or "beats" in text_lower or "raises" in text_lower or "price target" in text_lower or "record high" in text_lower or "outperform" in text_lower):
            return "best_to_buy"
        elif label == "negative" and ("downgrade" in text_lower or "warns" in text_lower or "misses" in text_lower or "cuts" in text_lower or "recall" in text_lower or "underperform" in text_lower):
            return "best_to_avoid"
        elif label == "positive":
            return "best_to_buy"
        elif label == "negative":
            return "best_to_avoid"
        else:
            return "neutral"
    except Exception as e:
        print("Error classifying article:", e)
        return "neutral"

df["classification"] = df.apply(classify_article, axis=1)

df.to_csv("sentiment.csv", index=False)