MarketDigest: Daily Stock News Dashboard

## Overview

MarketDigest is a locally hosted tool that turns the latest stock news into actionable insights you can review each morning. It scrapes fresh articles from Yahoo Finance, summarizes and classifies their sentiment, and presents them in a clean dashboard. You can also ask free‑form questions about the news to get focused answers with source links. No paid APIs are required—everything runs on your machine.

---
![image](https://github.com/user-attachments/assets/427d118f-8f9f-4dfa-b610-b9d56cbd6cbb)
![image](https://github.com/user-attachments/assets/238e367e-95fd-4546-875f-b663265b3489)


## Features

- Automatic scraping of Yahoo Finance news via Selenium
- Article summaries generated with a local text‑generation model
- Sentiment classification (positive, negative, neutral) based on simple word counts and recommendations
- FAISS‑backed vector index for fast retrieval
- Interactive Streamlit dashboard with two‑column view and expandable summaries
- Retrieval‑Augmented Generation (RAG) Q&A powered by a local FLAN‑T5 model

---

## Requirements

- Python 3.8 or higher
- Google Chrome or Chromium browser
- ChromeDriver matching your browser version
- Git for version control
- pip for installing dependencies

---

## Setup Steps

### 1. Clone the repository

```bash
git clone git@github.com:YOUR_USERNAME/MarketDigest.git
cd MarketDigest
```

Replace `YOUR_USERNAME` with your GitHub username (or use the HTTPS URL).

---

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows use `.venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
selenium
streamlit
pandas
sentence-transformers
transformers
huggingface-hub
faiss-cpu
langchain-community
langchain-huggingface
```

---

### 3. Download ChromeDriver

1. Visit https://chromedriver.chromium.org/downloads  
2. Download the version that matches your browser  
3. Place the `chromedriver` (or `chromedriver.exe`) file in the project root or note its full path

---

### 4. Scrape the latest news

```bash
python selenium_financial.py \
  --driver ./chromedriver.exe \
  --output selenium_yahoo_finance.jsonl \
  --timestamp \
  --format jsonl
```

This command collects articles from the past 24 hours and saves them as a JSONL file.

---

### 5. Ingest articles into FAISS

```bash
python ingest.py
```

This step computes embeddings, builds a FAISS index in the `faiss_index/` folder, and saves article metadata to `faiss_meta.pkl`.

---

### 6. Verify search functionality

```bash
python test_search.py
```

Run this to confirm that similar articles are retrieved correctly for a sample query.

---

### 7. Launch the dashboard

```bash
streamlit run dashboard.py
```

- The left column displays positive stories  
- The right column displays negative stories  
- Click on any headline to expand its summary and link  
- Type a question at the bottom to get an AI‑generated answer and source list

---

## How It Works

1. **Scraper** (`selenium_financial.py`): Scrolls the Yahoo Finance news page, extracts titles, URLs, timestamps, summaries, and sentiment counts.  
2. **Filtering**: Keeps only the articles published within the last 24 hours.  
3. **Ingestion** (`ingest.py`): Loads the JSONL data, embeds text with MiniLM, and builds a FAISS vector index.  
4. **Search Test** (`test_search.py`): Reloads the FAISS index and runs a similarity query to verify retrieval accuracy.  
5. **Dashboard** (`dashboard.py`): Reads the JSONL into a DataFrame, applies refined sentiment classification, and displays an interactive dashboard in Streamlit.  
6. **RAG QA** (`rag.py`): Uses the FAISS retriever to fetch top‑k articles, builds a prompt including their content, and generates answers with a local FLAN‑T5 model.

Feel free to fork, customize, and share this tool to help teams get a quick, actionable snapshot of the market before the trading day begins.
