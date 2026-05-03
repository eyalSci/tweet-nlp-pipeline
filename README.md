# Twitter NLP Pipeline - 2020 US Presidential Election

A full NLP pipeline applied to ~1.72M tweets from the 2020 US presidential election, demonstrating end-to-end text preprocessing, keyword-based feature extraction, document classification, and structured visualisation of results.

Built as a university project at the University of Twente.

---

## What this project demonstrates

This project maps closely to applied NLP work in industrial settings: turning large, noisy collections of unstructured text into structured, searchable insights.

**Text preprocessing pipeline** - a reusable cleaning module handles URL removal, mention normalisation, hashtag stripping, and alias resolution (e.g. mapping informal names to canonical entities). The pipeline is parameterised and documented so it can be applied to new text corpora without modification.

**Keyword and aspect extraction** - two complementary approaches are implemented and compared:
- *TF-IDF*: identifies the most discriminative terms per document class, used both for popularity prediction and for surfacing the vocabulary that drives engagement.
- *Aspect-Based Sentiment Analysis (ABSA)*: extracts sentiment towards specific named entities within a document using a transformer model, enabling per-entity classification even when multiple targets appear in the same text.

**Document linking via shared features** - tweets are connected to geographic units (US states) through extracted keywords and candidate mentions, then cross-referenced against an external dataset (official election results) to evaluate whether shared vocabulary patterns correlate with real-world outcomes.

**Visualisation of relationships** - results are presented as time-series plots, choropleth maps, log-odds bar charts (word importance), and a confusion matrix, making findings accessible to both technical and non-technical audiences.

---

## Project structure

```
tweet-election-sentiment/
├── data/                         # Place downloaded CSVs here (git-ignored due to size)
├── notebooks/
│   └── analysis.ipynb            # Full pipeline: tweet_preprocessing → VADER → ABSA → prediction → visualisation
├── src/
│   ├── tweet_preprocessing.py    # Reusable tweet cleaning and target extraction module
│   └── absa_sentiment.py         # ABSA inference pipeline (CLI script)
├── requirements.txt
└── README.md
```

---

## Dataset

Both datasets are publicly available on Kaggle:

- [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets) - 1.72M tweets containing `#JoeBiden`, `#Biden`, `#DonaldTrump`, or `#Trump`, collected October 15 – November 8, 2020
- [2020 US Presidential Election Results by State](https://www.kaggle.com/datasets/callummacpherson14/2020-us-presidential-election-results-by-state)

Download both and place the CSVs in the `data/` directory.

---

## Setup

```bash
git clone https://github.com/<your-username>/tweet-election-sentiment.git
cd tweet-election-sentiment

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Running the pipeline

**Step 1 - ABSA sentiment extraction** (GPU recommended; produces `data/sentiment.csv`):

```bash
python src/absa_sentiment.py --input data/tweets.csv --output data/sentiment.csv
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `tweets.csv` | Raw tweets CSV |
| `--output` | `sentiment.csv` | Output path |
| `--chunk-size` | `2048` | Rows per inference batch (tune to available memory) |

**Step 2 - Analysis notebook**:

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Key results

### Keyword extraction - what drives engagement

TF-IDF log-odds analysis reveals that tweets containing real-time, action-oriented terms (*electionnight*, *electionresults2020*, *votes*, *won*) are significantly more likely to receive engagement, while generic political vocabulary (*democrats*, *america*, *election*) is not. This directly demonstrates how term-level features can be used to classify and rank documents.

### Aspect-based extraction - per-entity sentiment

ABSA correctly disambiguates sentiment within tweets mentioning both candidates simultaneously. A tweet can yield negative sentiment towards one entity and positive towards another - something a document-level classifier cannot capture. Each aspect is tagged with a confidence-weighted score, enabling fine-grained cross-document comparison.

### Document–external data linking via shared features

State-level sentiment scores (derived from aggregated keywords and ABSA outputs) are linked to official election results, showing moderate Pearson correlation:

| Method | Biden r | Trump r |
|---|---|---|
| VADER (lexicon-based) | 0.38 | 0.32 |
| ABSA (transformer, zero-shot) | 0.33 | 0.05 |

VADER outperforms zero-shot ABSA here, consistent with the known limitation of applying general-purpose transformer models to specialised domains without fine-tuning - a relevant consideration for industrial keyword extraction tasks.

### Popularity prediction

A Multinomial Naïve Bayes classifier trained on TF-IDF features achieves **61% accuracy** on held-out data, with interpretable per-class word importance plots.

---

## Technical skills demonstrated

| Skill | Where |
|---|---|
| Python (pandas, scikit-learn, NLTK, spaCy) | Throughout |
| Text preprocessing pipeline design | `src/tweet_preprocessing.py` |
| TF-IDF vectorisation and keyword selection | `notebooks/analysis.ipynb` §4 |
| Transformer-based aspect extraction (PyABSA) | `src/absa_sentiment.py` |
| Document classification (Naïve Bayes) | `notebooks/analysis.ipynb` §4 |
| Relationship visualisation (time series, maps, word charts) | `notebooks/analysis.ipynb` §2–3 |
| Structured documentation and reproducible code | All files |

---

## Authors

Eyal Gavrielov & Cristina Man - University of Twente, Enschede
