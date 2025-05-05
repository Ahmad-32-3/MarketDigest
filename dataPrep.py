import pandas as pd

df = pd.read_json("selenium_yahoo_finance.jsonl", lines=True)
df["sentiment"] = df.apply(
    lambda r: "positive" if r.pos_count > r.neg_count
              else "negative" if r.neg_count > r.pos_count
              else "neutral",
    axis=1
)
df.to_csv("sentiment.csv", index=False)