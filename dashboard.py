# dashboard.py

import streamlit as st
st.set_page_config(page_title="24h Stock News Sentiment", layout="wide")

import pandas as pd
from typing import Optional

# ────────────────────────────────────────────────────────────
# IMPORT YOUR RAG FUNCTION
# ────────────────────────────────────────────────────────────
from rag import answer_query

# ────────────────────────────────────────────────────────────
# 1) LOAD & CLASSIFY DATA
# ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_classify(path: str = "selenium_yahoo_finance.jsonl") -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    def classify(row) -> str:
        low = row.title.lower()
        # force positive if “Best ... to Buy” appears
        if "best" in low and "to buy" in low:
            return "positive"
        # otherwise use your pos/neg counts
        if row.pos_count > row.neg_count:
            return "positive"
        elif row.neg_count > row.pos_count:
            return "negative"
        else:
            return "neutral"

    df["sentiment"] = df.apply(classify, axis=1)
    return df

df = load_and_classify()

# ────────────────────────────────────────────────────────────
# 2) DASHBOARD LAYOUT
# ────────────────────────────────────────────────────────────
st.title("24h Stock News Sentiment Dashboard")
st.markdown("Positive vs Negative stories from the last 24h")

col1, col2 = st.columns(2)

with col1:
    st.header("🟢 Best to Buy")
    pos_df = df[df.sentiment == "positive"]
    if pos_df.empty:
        st.write("No positive stories right now.")
    else:
        for _, row in pos_df.iterrows():
            with st.expander(row.title):
                st.markdown(f"[Read on Yahoo ▶]({row.url})")
                text = getattr(row, "summary", None) or getattr(row, "content_full", "")
                st.write(text)

with col2:
    st.header("🔴 Probably should sell")
    neg_df = df[df.sentiment == "negative"]
    if neg_df.empty:
        st.write("No negative stories right now.")
    else:
        for _, row in neg_df.iterrows():
            with st.expander(row.title):
                st.markdown(f"[Read on Yahoo ▶]({row.url})")
                text = getattr(row, "summary", None) or getattr(row, "content_full", "")
                st.write(text)

# ────────────────────────────────────────────────────────────
# 3) FREE-FORM RAG QUESTION
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Ask a question about today’s news")

query: Optional[str] = st.text_input("Enter your question")
if query:
    with st.spinner("Retrieving and answering…"):
        answer, sources = answer_query(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source Articles")
    for meta in sources:
        title = meta.get("title", "No title")
        url   = meta.get("url", "#")
        st.markdown(f"- [{title}]({url})")
