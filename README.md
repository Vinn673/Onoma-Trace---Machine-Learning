# 🌍 Onoma Trace

Predict the most likely country of origin for a romanized personal name, across ~104
countries, using character-level machine learning. The project is a complete, reproducible
ML pipeline — from raw name-frequency data to a live demo — built as a university group
project.

> ⚠️ **Ethics note:** a name is a weak, probabilistic signal. This project produces a
> statistical guess based on character patterns, not a factual statement about anyone's
> nationality or ethnicity. It should not be used for profiling, hiring, credit,
> immigration, or law-enforcement decisions.

## Overview

Names carry statistical patterns (letter combinations, spelling conventions, common
suffixes) that vary by country. This project explores whether those patterns alone are
enough to predict where a name is likely to come from, without using any additional
personal information.

**What it does:** given a romanized full name (e.g. `Joko Widodo`), the model returns a
ranked list of candidate countries with confidence scores, out of 104 possible countries.

**How it works at a high level:**
1. The name is normalized — decomposed, stripped of accents, lowercased, and reduced to
   `a–z` characters and single spaces. The same normalization logic is shared between
   training and inference to guarantee consistency.
2. The normalized name is vectorized using character n-gram TF-IDF.
3. A linear classifier scores the vector against all 104 countries and returns the
   top-k predictions.

**Main purpose:** to build and evaluate an end-to-end, reproducible NLP/ML pipeline —
covering data preparation, exploratory analysis, model training, evaluation, and
deployment — rather than to produce a production-grade identification tool.

## Features

- **Five-stage ML pipeline**, each stage runnable independently or end-to-end via `main.py`:
  1. **EDA** — generates plots and a written report on class balance, name length,
     character/token distributions, and multi-country name ambiguity.
  2. **Preprocess** — synthesizes realistic full names from aggregated forename/surname
     frequency tables, splits the data (stratified 70/15/15 train/val/test), and
     featurizes names into arrays.
  3. **Train (baseline)** — trains three linear classifiers on character-level TF-IDF
     features and saves the trained models and vectorizer.
  4. **Evaluate** — computes held-out test metrics (including an ambiguity-aware,
     "set-lenient" metric that accounts for names spanning multiple countries) and
     generates comparison plots.
  5. **Predict** — command-line inference that returns the top-k most likely countries
     for a given name.
- **Shared preprocessing module** (`pipeline/helper/common.py`) ensures identical
  normalization and featurization logic between training and inference.
- **Interactive demo app** (Gradio) that walks through every pipeline stage as a separate
  tab, replaying pre-generated logs, plots, and metrics, with a live "Predict" tab for:
  - single-name predictions with confidence scores, and
  - batch predictions from an uploaded CSV, with results downloadable as CSV.
- **Model backend toggle** in the demo — switch between Logistic Regression, Linear SVM,
  and SGD at inference time.
- **One-shot deployment script** (`pipeline/helper/deploy_space.py`) to publish the `app/`
  demo to a Hugging Face Space.

## Technologies

- **Language:** Python
- **Machine learning:** scikit-learn (Logistic Regression, Linear SVM, SGD classifiers,
  TF-IDF vectorization)
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib / seaborn (for EDA and evaluation plots)
- **Demo / UI:** Gradio (deployed as a Hugging Face Space)
- **Serialization:** pickle (for saved models and vectorizer)

## Project Structure

```
.
├── main.py                 # Runs the full pipeline end-to-end
├── pipeline/
│   ├── eda01.py             # 1. Exploratory data analysis
│   ├── preprocess02.py      # 2. Build dataset + train/val/test splits
│   ├── baseline03.py        # 3. Train linear models
│   ├── evaluation04.py      # 4. Evaluate on held-out test set
│   ├── predict05.py         # 5. CLI inference
│   └── helper/
│       ├── common.py        # Shared normalization + featurization logic
│       ├── eda_style.py      # Shared plot styling
│       └── deploy_space.py   # Deploys app/ to a Hugging Face Space
├── app/                     # Gradio demo application (Hugging Face Space)
├── archive/                 # Raw source data (forename/surname/country tables)
├── data/                    # Generated splits, featurized arrays, and label maps
├── eda/                     # Generated EDA plots and report
└── models/                  # Trained models, vectorizer, metrics, and plots
```

## Installation / Setup

**Requirements:** Python 3, pip.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd onoma-trace

# 2. Install dependencies
pip install -r requirements.txt
```

The raw source data is not included in the repository. It is provided as a Kaggle
dataset; see [`data/download_link.txt`](data/download_link.txt) for the link, and place
the files under `archive/` before running the pipeline.

## Usage

**Run the full pipeline (EDA → preprocess → train → evaluate → predict):**

```bash
python main.py
```

**Run a single stage:**

```bash
python -m pipeline.eda01
python -m pipeline.baseline03
```

> Run pipeline modules from the project root using `python -m pipeline.<module>`
> (not `python pipeline/<module>.py`) so the `pipeline` package resolves correctly.

**Predict from the command line:**

```bash
python -m pipeline.predict05 --name "Joko Widodo" --top_k 5
python -m pipeline.predict05 --name "Tanaka" --model baseline_linsvm.pkl --json
```

**Run the interactive demo locally:**

```bash
cd app
pip install -r requirements.txt
python app.py
```

The demo presents the project as one tab per pipeline stage, with a live "Predict" tab
for single or batch (CSV) predictions and a toggle between model backends.

## Team Members

**Group 03**
- Brian Nicholas Tedjo — 2802403183
- Jason Budiharjo — 2802419446
- Marvin Adriano Rusdianto — 2802402275

## Notes

- **Data source:** the raw data consists of aggregated forename and surname frequency
  counts per country (not pre-joined full names). Since there are no ready-made full-name
  records, the preprocessing stage synthesizes full names by sampling a forename and
  surname weighted by their real per-country frequencies.
- **Data license:** the dataset license is included at
  [`archive/LICENSE.txt`](archive/LICENSE.txt) (Apache License 2.0).
- **Limitations:** name-based origin prediction is inherently uncertain — many names are
  common across multiple countries, and the underlying data is romanized and unevenly
  distributed across countries. Predictions are statistical estimates, not factual
  claims about any individual.
- **Reproducibility:** normalization and featurization logic is centralized in
  `pipeline/helper/common.py` and reused at inference time, so predictions are generated
  the same way as during training.
- **Screenshots:** the `eda/`, `models/plots/`, and `app/assets/` folders contain
  generated plots (class balance, EDA charts, confusion matrix, model comparison) that
  could be added here as screenshots to illustrate results.
