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
def load_classification(path: str = "sentiment.csv") -> pd.DataFrame:
    return pd.read_csv(path)

df = load_classification()


# ────────────────────────────────────────────────────────────
# 2) DASHBOARD LAYOUT
# ────────────────────────────────────────────────────────────
st.title("24h Stock News Sentiment Dashboard")
st.markdown("Best to Buy vs Best to Avoid stories from the last 24h")

col1, col2 = st.columns(2)

with col1:
    st.header("🟢 Best to Buy")
    pos_df = df[df.classification == "best_to_buy"]
    if pos_df.empty:
        st.write("No positive stories right now.")
    else:
        for _, row in pos_df.iterrows():
            with st.expander(row.title):
                st.markdown(f"[Read on Yahoo ▶]({row.url})")
                text = getattr(row, "summary", None) or getattr(row, "content_full", "")
                st.write(text)

with col2:
    st.header("🔴 Best to Avoid")
    neg_df = df[df.classification == "best_to_avoid"]
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
