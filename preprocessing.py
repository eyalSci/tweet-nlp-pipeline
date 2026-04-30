"""
preprocessing.py
Tweet cleaning and target extraction utilities shared across the pipeline.
"""

import re
import pandas as pd

IMPORTANT_USERS = {"@realdonaldtrump", "@joebiden"}
POLITICIANS = ["Trump", "Biden"]
_POLITICIAN_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, POLITICIANS)) + r")\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _replace_mention(match: re.Match) -> str:
    mention = match.group()
    return mention if mention.lower() in IMPORTANT_USERS else ""


def clean_mentions(text: str) -> str:
    """Remove @mentions except for the two main candidates."""
    return re.sub(r"@\w+", _replace_mention, text)


def normalize_poi_names(text: str) -> str:
    """Normalise popular nicknames/aliases to canonical candidate names."""
    text = re.sub(r"sleepy joe", "bad Biden ", text, flags=re.IGNORECASE)
    text = re.sub(r"\borange\b", "bad Trump ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Joe Biden|#JoeBiden|#Biden|joebiden|@joebiden|\bjoe\b|\bbiden\b",
        " Biden ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Donald Trump|Donald J\. Trump|#Trump|#DonaldTrump|donaldtrump"
        r"|donaldjtrump|realdonaldtrump|@realdonaldtrump|\b45\b|the donald|\btrump\b",
        " Trump ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def clean_tweet(text: str) -> str:
    """Full cleaning pipeline for sentiment analysis."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = clean_mentions(text)
    text = re.sub(r"#(\w+)", r"\1", text)          # keep hashtag words
    text = normalize_poi_names(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def extract_targets(tweet: str) -> list[str]:
    """Return unique politician names mentioned in a tweet."""
    return list(set(_POLITICIAN_PATTERN.findall(tweet)))


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def load_and_filter_tweets(path: str) -> pd.DataFrame:
    """Load tweets CSV, drop unnamed index column, keep English tweets."""
    df = pd.read_csv(path)
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)
    df = df[df["language"] == "en"].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_for_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned tweet and target columns."""
    out = df[["tweet_id", "tweet"]].copy()
    out["tweet_original"] = out["tweet"]
    out["tweet"] = out["tweet"].apply(clean_tweet)
    out["targets"] = out["tweet"].apply(extract_targets)
    return out
