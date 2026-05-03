"""
absa_sentiment.py
Aspect-Based Sentiment Analysis pipeline using PyABSA.

Run directly:
    python src/absa_sentiment.py --input tweets.csv --output sentiment.csv
"""

import argparse
import time
import warnings

import pandas as pd
import torch
from pyabsa import AspectPolarityClassification as APC
from tqdm import tqdm

from tweet_preprocessing import load_and_filter_tweets, prepare_for_sentiment

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_SIZE = 2048       # rows per batch — tune to your GPU memory
BATCH_SIZE = 2048       # internal PyABSA batch size
MAX_SEQ_LEN = 80        # tweets rarely exceed this


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_tagged_inputs(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Explode one row per (tweet, target) pair and wrap each target in
    the [B-ASP]...[E-ASP] markers that PyABSA expects.

    Returns
    -------
    tagged_tweets : list of strings ready for the classifier
    metadata      : parallel list of {tweet_id, target} dicts
    """
    df_filtered = df[df["targets"].str.len() > 0].copy()
    df_exploded = df_filtered.explode("targets").reset_index(drop=True)
    df_exploded["tagged_tweet"] = (
        "[B-ASP]" + df_exploded["targets"] + "[E-ASP] " + df_exploded["tweet"]
    )
    tagged_tweets = df_exploded["tagged_tweet"].tolist()
    metadata = df_exploded[["tweet_id", "targets"]].to_dict(orient="records")
    return tagged_tweets, metadata


def run_absa(
    tagged_tweets: list[str],
    metadata: list[dict],
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Run PyABSA on all tagged tweets and return a tidy results DataFrame."""
    classifier = APC.SentimentClassifier("english", auto_device=True, tokenizer="fast")
    classifier.config.use_amp = True
    classifier.config.batch_size = BATCH_SIZE
    classifier.config.max_seq_len = MAX_SEQ_LEN

    all_results: list = []
    t0 = time.time()

    for i in tqdm(range(0, len(tagged_tweets), chunk_size), desc="ABSA batches"):
        chunk = tagged_tweets[i : i + chunk_size]
        results = classifier.predict(chunk, print_result=False, ignore_error=True)
        all_results.extend(results)

    elapsed = time.time() - t0
    print(f"Classified {len(tagged_tweets)} (tweet, target) pairs in {elapsed:.1f}s")

    records = [
        {
            "tweet_id": meta["tweet_id"],
            "target": meta["targets"],
            "sentiment": result["sentiment"][0] if result else "Not_Mentioned",
            "confidence": result["confidence"][0] if result else 0.0,
        }
        for meta, result in zip(metadata, all_results)
    ]
    return pd.DataFrame(records)


def pivot_results(results_df: pd.DataFrame, tweets_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (tweet_id, target) rows to one row per tweet with per-candidate columns."""
    pivot = results_df.pivot(
        index="tweet_id", columns="target", values=["sentiment", "confidence"]
    )
    pivot.columns = [f"{col[1]}.{col[0]}" for col in pivot.columns]
    return tweets_df.merge(pivot, left_on="tweet_id", right_index=True, how="left")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABSA on election tweets")
    parser.add_argument("--input", default="tweets.csv", help="Path to raw tweets CSV")
    parser.add_argument("--output", default="sentiment.csv", help="Output CSV path")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    print("Loading tweets …")
    raw_df = load_and_filter_tweets(args.input)
    tweets_df = prepare_for_sentiment(raw_df)

    print("Building tagged inputs …")
    tagged_tweets, metadata = build_tagged_inputs(tweets_df)

    print(f"Running ABSA on {len(tagged_tweets)} (tweet, target) pairs …")
    results_df = run_absa(tagged_tweets, metadata, chunk_size=args.chunk_size)

    print("Pivoting and saving …")
    final_df = pivot_results(results_df, tweets_df)
    final_df.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
