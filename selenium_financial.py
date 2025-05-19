import argparse
import csv
import json
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta

import dateparser
from datetime import datetime, timedelta, timezone
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def parse_timestamp_text(text: str) -> datetime:
    """
    Convert relative text like '3 hours ago' or 'yesterday' into a UTC datetime.
    """
    dt = dateparser.parse(
        text,
        settings={'RELATIVE_BASE': datetime.utcnow(), 'RETURN_AS_TIMEZONE_AWARE': False}
    )
    return dt

def extract_detail(driver: webdriver.Chrome, url: str):
    """
    Open article in a new tab, scrape <time> and full <article><p> text.
    Returns (timestamp_iso, full_text).
    """
    original = driver.current_window_handle
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(url)

    ts, full = None, ""
    try:
        # wait for the <time> element
        elem = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'time'))
        )
        raw = elem.get_attribute('datetime') or elem.text
        # ISO? or relative?
        if raw and raw[0:4].isdigit():
            ts = raw  # ISO8601
        else:
            dt = parse_timestamp_text(raw)
            ts = dt.isoformat() + 'Z'
        # get full article paragraphs
        paras = driver.find_elements(By.CSS_SELECTOR, 'article p')
        full = "\n\n".join(p.text for p in paras)
    except Exception as e:
        logging.warning(f"Detail scrape failed for {url}: {e}")
    finally:
        driver.close()
        driver.switch_to.window(original)

    return ts, full

def main(args):
    setup_logging(args.log)
    logging.info("Starting scraper")

    # Chrome webdriver setup
    chrome_opts = Options()
    if not args.debug:
        chrome_opts.add_argument("--headless")
        chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--ignore-certificate-errors")
    service = Service(args.driver)
    driver = webdriver.Chrome(service=service, options=chrome_opts)

    # Load page and scroll
    driver.get(args.url)
    logging.info(f"Navigated to {args.url}")
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(args.scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(args.pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    logging.info("Completed scrolling")

    # Wait for articles
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li.stream-item.story-item'))
        )
    except Exception as e:
        logging.error(f"Timeout waiting for articles: {e}")
        driver.save_screenshot("timeout_error.png")
        driver.quit()
        return

    elems = driver.find_elements(By.CSS_SELECTOR, 'li.stream-item.story-item')
    logging.info(f"Found {len(elems)} article elements")

    data = []
    pos_words = ["up", "rise", "gain", "bull", "positive"]
    neg_words = ["down", "fall", "lose", "bear", "negative"]
    ticker_pattern = re.compile(r'\(([A-Z]{1,5})\)')

    for idx, el in enumerate(elems, start=1):
        try:
            title   = el.find_element(By.CSS_SELECTOR, 'h3').text.strip()
            url     = el.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
            summary = el.find_element(By.CSS_SELECTOR, 'p').text.strip() if el.find_elements(By.CSS_SELECTOR, 'p') else ""
            timestamp, content_full = extract_detail(driver, url)


            # Unique stable ID from URL
            uid = hashlib.sha1(url.encode('utf-8')).hexdigest()

            # Extract ticker if present
            match  = ticker_pattern.search(title)
            ticker = match.group(1) if match else None

            # Basic sentiment scaffolding
            low       = title.lower()
            pos_count = sum(low.count(w) for w in pos_words)
            neg_count = sum(low.count(w) for w in neg_words)

            record = {
                "id": uid,
                "title": title,
                "url": url,
                "ticker": ticker,
                "timestamp": timestamp,
                "summary": summary,
                "content_full": content_full,
                "pos_count": pos_count,
                "neg_count": neg_count,
            }
            data.append(record)
            logging.info(f"Scraped [{idx}]: {title}")
        except Exception as e:
            logging.error(f"Error on article #{idx}: {e}")

    driver.quit()
    logging.info(f"Collected {len(data)} total articles")

    # ——— 24-Hour or Custom Date-Range Filter ———
    if args.timestamp and data:
        # use a timezone-aware “now”
        now = datetime.now(timezone.utc)
        # if user passed --start-date, use that; otherwise go back 24h
        if args.start_date:
            cutoff = datetime.fromisoformat(
                args.start_date.replace('Z', '+00:00')
            )
        else:
            cutoff = now - timedelta(hours=24)

        before = len(data)
        filtered = []
        for rec in data:
            ts = rec.get('timestamp')
            if not ts:
                continue
            # parse ISO8601 into UTC-aware datetime
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if dt >= cutoff:
                filtered.append(rec)
        data = filtered
        logging.info(
            f"Filtered articles: {len(data)} of {before} "
            f"kept since {cutoff.isoformat()}Z"
        )
    elif not args.timestamp:
        logging.warning(
            "Timestamp flag not set; skipping date filter"
        )

    # ——— Write out JSONL or CSV ———
    if not data:
        logging.warning("No articles to write, exiting")
        return

    if args.format == 'jsonl':
        with open(args.output, 'w', encoding='utf-8') as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(data)} articles to {args.output} (JSONL)")
    else:
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        logging.info(f"Saved {len(data)} articles to {args.output} (CSV)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Yahoo Finance News Scraper")
    parser.add_argument("--driver", default="./chromedriver.exe", help="Path to chromedriver")
    parser.add_argument("--url", default="https://finance.yahoo.com/news/", help="News URL")
    parser.add_argument("--output", default="selenium_yahoo_finance.csv", help="Output filename")
    parser.add_argument("--log", default="scrape.log", help="Log file path")
    parser.add_argument("--scrolls", type=int, default=3, help="Number of scrolls")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause between scrolls")
    parser.add_argument("--timestamp", action="store_true",
                        help="Visit each article for timestamp & full text")
    parser.add_argument("--start-date", type=str, default=None,
                        help="ISO8601 lower bound filter, e.g. '2025-05-03T00:00:00Z'")
    parser.add_argument("--format", choices=['csv','jsonl'], default='csv',
                        help="Output format: csv or jsonl")
    parser.add_argument("--debug", action="store_true", help="Disable headless mode")
    args = parser.parse_args()
    main(args)
