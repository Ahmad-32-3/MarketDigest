import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

MODEL_NAME        = "facebook/bart-large-cnn"
MAX_CHUNK_TOKENS  = 512    # max tokens per chunk
DEFAULT_MAX_SUM   = 128    # default summary length
DEFAULT_MIN_SUM   = 30     # default minimum summary length

# ────────────────────────────────────────────────────────────
# INITIALIZE: tokenizer + model + pipeline
# ────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading `{MODEL_NAME}` on {device} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if device == "cuda":
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe_device = 0
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    pipe_device = -1

# give the pipeline a default max_length so truncation has something to work against
_summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=pipe_device,
    truncation=True,
    max_length=DEFAULT_MAX_SUM,    # <-- default truncate length
    min_length=DEFAULT_MIN_SUM,    # <-- default minimum length
)


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """
    Naively split on sentence boundaries into chunks of <= max_tokens.
    """
    sents = re.split(r'(?<=[\.\!\?]) +', text)
    chunks, current, curr_len = [], [], 0

    for sent in sents:
        length = len(tokenizer.tokenize(sent))
        if curr_len + length > max_tokens:
            if current:
                chunks.append(" ".join(current))
            current = [sent]
            curr_len = length
        else:
            current.append(sent)
            curr_len += length

    if current:
        chunks.append(" ".join(current))

    return chunks

def summarize(text: str,
              max_length: int = DEFAULT_MAX_SUM,
              min_length: int = DEFAULT_MIN_SUM) -> str:
    """
    Summarize the text, automatically chunking long inputs.
    Returns a single string with each chunk's summary joined by blank lines.
    """
    if not text:
        return ""

    chunks    = _chunk_text(text)
    summaries = []

    for chunk in chunks:
        tok_count  = len(tokenizer.tokenize(chunk))
        # never ask for more than half your input length
        clamp_max  = min(max_length, tok_count // 2) or 1
        clamp_min  = min(min_length, clamp_max // 2) or 1

        out = _summarizer(
            chunk,
            max_length=clamp_max,
            min_length=clamp_min,
            do_sample=False,
            truncation=True,
        )
        summaries.append(out[0]["summary_text"].strip())

    return "\n\n".join(summaries)
