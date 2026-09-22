# 🌍 Onoma Trace — Name Origin & Nationality Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.16-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/britod/name-origins-checker)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicting country of origin from romanized personal names across 104 countries using character-level machine learning.**

---

## 1. Project Title

**Onoma Trace — End-to-End Machine Learning Pipeline for Personal Name Origin & Nationality Prediction**

---

## 2. Description

**Onoma Trace** is an end-to-end, reproducible Machine Learning and Natural Language Processing (NLP) system designed to predict the probable country of origin for romanized personal names across **104 countries**. 

Personal names carry distinctive sub-word orthographic patterns, affixes, and phonotactic conventions reflecting cultural and linguistic heritage. However, predicting origin from names is challenging due to cross-border name ambiguity, migration, and raw data imbalance. Onoma Trace addresses this problem by streaming and processing over **33.5 million** raw name-frequency records, synthesizing a balanced dataset of **5.18 million** full names, extracting character-level $n$-gram TF-IDF representations, and training lightweight linear classifiers optimized for balanced macro-performance.

The primary purpose of this project is to build and evaluate a rigorous, production-grade, and CPU-efficient ML pipeline—spanning chunked exploratory data analysis (EDA), synthetic dataset generation, model benchmarking, ambiguity-aware evaluation metrics, CLI inference, and a live web application.

> ⚠️ **Ethics & Responsible AI Note:** A personal name is a probabilistic signal, not an identity verification tool. Predictions are statistical approximations based on character distributions and do not represent factual statements about an individual's nationality, citizenship, or ethnicity. This system must **not** be used for profiling, hiring, credit scoring, immigration, or law enforcement decisions.

---

## 3. Technology Used

The project is built entirely in Python using modern, standard scientific computing and machine learning libraries:

- **Python**: Core programming language for data pipelines, modeling, and application backend.
- **scikit-learn**: Model training, TF-IDF vectorization, multiclass classification, and comprehensive performance metrics.
- **pandas & NumPy**: Memory-efficient chunked CSV streaming, dataset synthesis, array operations, and label mapping.
- **Gradio**: Interactive web application featuring pipeline execution replay, live single-name inference, and batch CSV processing.
- **Matplotlib & Seaborn**: Clean visual styling for exploratory data analysis charts, confusion matrices, and model comparison figures.
- **Hugging Face Hub**: Automated cloud deployment of the web demo to Hugging Face Spaces.
- **Pickle**: High-performance serialization of fitted vectorizers and trained classification models.

---

## 4. Overview

### What the Project Does
Given a romanized personal name (such as `"Joko Widodo"`, `"Tanaka"`, or `"Alexandre Dumas"`), Onoma Trace normalizes the text, extracts character $n$-gram features, and outputs a ranked distribution of the most probable candidate countries out of 104 nations with associated confidence probabilities.

### Its Purpose
Many international name classification tasks face significant bottlenecks: massive dataset sizes, heavy class imbalances, and high computational costs from deep neural architectures. Onoma Trace demonstrates that an optimized, character-level linear pipeline can achieve competitive classification performance (~0.501 macro-F1 across 104 classes) with CPU-only training in under 3 minutes and sub-millisecond inference latency.

### The General Concept
Rather than relying on dictionary lookups or static origin databases, the model learns subtle sub-word structural clues. By decomposing names into character boundary $n$-grams (ranges of 2 to 4 characters), the system identifies language-specific affixes (e.g., *-opoulos*, *-ski*, *van*, *al-*, *-ov*, *-ez*, *-chen*) regardless of whether the specific full name was seen during training.

### What Users Can Do With It
- **Interactive Single-Name Query**: Enter any romanized full name and receive the top-$k$ most likely countries with calibrated probability scores.
- **Batch CSV Classification**: Upload a CSV file containing hundreds or thousands of names and export predictions and probability rankings as a downloadable CSV.
- **Model Backend Switching**: Dynamically toggle between three distinct model backends: Multinomial Logistic Regression, Linear Support Vector Machine (LinearSVC), and Stochastic Gradient Descent (SGD).
- **Pipeline Replay**: Walk through the five modular pipeline stages (EDA, Preprocessing, Baselines, Evaluation, Prediction) with interactive console execution logs and pre-rendered analytical charts.

---

## 5. Live Demo

Experience a recorded walkthrough of the pipeline execution and user interface demonstration:

[![Live Demo](https://img.shields.io/badge/Google_Drive-Live_Demo_Video-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/u/0/folders/11vR_7o_P7wB18QjTrMKqmnOwl6ep9aYj)

🔗 **Direct Link:** [https://drive.google.com/drive/u/0/folders/11vR_7o_P7wB18QjTrMKqmnOwl6ep9aYj](https://drive.google.com/drive/u/0/folders/11vR_7o_P7wB18QjTrMKqmnOwl6ep9aYj)

---

## 6. Try the App

Access and test the live deployed Gradio application directly in your web browser:

[![Try the App](https://img.shields.io/badge/HuggingFace-Try_the_App_Live-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/britod/name-origins-checker)

🔗 **Direct Link:** [https://huggingface.co/spaces/britod/name-origins-checker](https://huggingface.co/spaces/britod/name-origins-checker)

---

## 7. Key Features

- **Chunked Big-Data Processing**: Scans 33,522,823 raw records across 104 countries using chunked iterator streams (`chunksize=1,000,000`), keeping memory overhead bounded regardless of dataset scale.
- **Realistic Full-Name Synthesis**: Bridges the gap between disjoint forename and surname frequency tables by sampling combinations weighted by real-world country frequency distributions, generating 5,178,552 realistic full names.
- **Class-Balanced Stratified Splitting**: Controls class imbalance from severe raw disparities down to an even **1.75×** ratio across all 104 countries (3,624,986 train / 776,782 validation / 776,784 test).
- **Shared Normalization Contract (`common.py`)**: Ensures strict byte-identical preprocessing (Unicode NFD decomposition, accent removal, lowercase sanitization, and whitespace reduction) between offline training and online inference.
- **Fast CPU-Optimized Model Suite**: Trains three linear classifiers (`LogisticRegression` with SAGA solver, `LinearSVC`, and `SGDClassifier` with modified Huber loss) using character-level word-boundary TF-IDF vectorization.
- **Ambiguity-Aware Evaluation Metrics**: In addition to standard strict accuracy and macro-F1, calculates **Set-Lenient Accuracy** (`lenient@1` and `lenient@3`) leveraging a pre-computed map of 174,564 multi-country names to fairly credit names legitimately shared across borders (e.g., *Ali*, *Smith*, *Garcia*).
- **Dual Inference Modes**: Full support for both a scriptable CLI interface (`pipeline/predict05.py`) with JSON/ASCII output and an interactive Gradio web application (`app/app.py`).
- **One-Shot Cloud Deployment**: Includes `pipeline/helper/deploy_space.py` to programmatically package assets and publish the Gradio app directly to Hugging Face Spaces.

---

## 8. Tech Stack

| Technology / Library / Framework | Category | Purpose / Usage in the Project |
|---|---|---|
| **Python** (3.10+) | Programming Language | Core execution environment for all scripts, pipeline stages, and application servers |
| **scikit-learn** | Machine Learning | Implements `LogisticRegression`, `LinearSVC`, `SGDClassifier`, `TfidfVectorizer`, and evaluation metrics |
| **pandas** | Data Manipulation | Memory-bounded chunked processing of raw 33M CSV rows, dataset split management, and batch CSV inference |
| **NumPy** | Numerical Computing | Matrix operations, vectorized index representations, probability transformations, and array storage |
| **Gradio** | Web Framework | Powers the interactive web application, pipeline walkthrough tabs, and live prediction interfaces |
| **Matplotlib** | Data Visualization | Generates publication-quality charts for class balance, token distributions, and model performance |
| **Seaborn** | Statistical Visualization | Produces row-normalized confusion matrices for top multiclass classifications |
| **unicodedata** | Text Processing | Standard library module used for Unicode NFD decomposition and diacritic stripping |
| **Pickle** | Model Persistence | Serializes and deserializes trained linear models and the 50,000-feature TF-IDF vectorizer |
| **Hugging Face Hub** | Deployment & Hosting | Automates remote repository creation and file synchronization to Hugging Face Spaces |

---

## 9. Installation & Running

Follow these step-by-step instructions to set up the project and run it on your local machine.

### Prerequisites
- **Python**: Version `3.10` or higher (verified on Python 3.13)
- **Git**: For cloning the repository
- **RAM**: Minimum 8 GB recommended for running the full preprocessing and training stages

### Step 1: Clone the Repository
```bash
git clone https://github.com/Vinn673/Onoma-Trace---Machine-Learning.git
cd Onoma-Trace---Machine-Learning
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell):
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Obtain Dataset (Required for Full Pipeline Run)
The raw name frequency tables are sourced from Kaggle:
1. Download the dataset from [Kaggle: Forenames and Surnames with Gender and Country](https://www.kaggle.com/datasets/erpel1/forenames-and-surnames-with-gender-and-country).
2. Place the extracted files into the `archive/` directory:
   - `archive/forenames.csv`
   - `archive/surnames.csv`
   - `archive/country_codes.csv`

*(Note: If you only wish to run the Gradio app or CLI inference with pre-trained models, raw dataset download is not required if pre-generated models exist under `models/` or assets under `app/assets/`.)*

### Step 5: Run the Project

#### Option A: Run the End-to-End Pipeline
Execute all 5 stages sequentially (EDA $\rightarrow$ Preprocessing $\rightarrow$ Baseline Training $\rightarrow$ Evaluation $\rightarrow$ Inference):
```bash
python main.py
```

#### Option B: Run Individual Pipeline Stages
Execute any stage as a standalone Python module from the project root:
```bash
# Stage 1: Exploratory Data Analysis
python -m pipeline.eda01

# Stage 2: Data Synthesis & Stratified Splitting
python -m pipeline.preprocess02

# Stage 3: Feature Extraction & Model Training
python -m pipeline.baseline03

# Stage 4: Test Set Evaluation & Metric Calculation
python -m pipeline.evaluation04

# Stage 5: Command-Line Prediction
python -m pipeline.predict05 --name "Joko Widodo" --top_k 5
python -m pipeline.predict05 --name "Tanaka" --model baseline_linsvm.pkl --json
```

#### Option C: Run the Interactive Gradio Web App Locally
```bash
cd app
pip install -r requirements.txt
python app.py
```
After launch, open your browser and navigate to the local server address (default: `http://127.0.0.1:7860`).

---

## 10. Project Structure

```
onoma-trace/
├── main.py                     # Entry point orchestrating stages 01 through 05
├── requirements.txt            # Root dependencies for full pipeline and training
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── pipeline/                   # Modular machine learning pipeline stages
│   ├── __init__.py
│   ├── eda01.py                # Stage 1: Chunked raw table scanning & EDA visualization
│   ├── preprocess02.py         # Stage 2: Full-name synthesis, stratified 70/15/15 split
│   ├── baseline03.py           # Stage 3: TF-IDF feature extraction & linear model training
│   ├── evaluation04.py         # Stage 4: Strict and ambiguity-aware test set evaluation
│   ├── predict05.py            # Stage 5: CLI inference module (interactive or arguments)
│   └── helper/                 # Shared pipeline utilities
│       ├── __init__.py
│       ├── common.py           # Shared Unicode normalization & featurization logic
│       ├── eda_style.py        # Centralized Matplotlib visualization styling
│       └── deploy_space.py     # Script to publish app/ to Hugging Face Spaces
├── app/                        # Standalone Gradio web application (Hugging Face Space)
│   ├── app.py                  # Multi-tab Gradio UI with live predict & walkthrough
│   ├── common.py               # Identical normalization logic for web deployment
│   ├── country_codes.csv       # ISO country code to country name dictionary
│   ├── requirements.txt        # Lightweight inference dependencies for Gradio
│   ├── README.md               # Hugging Face Space configuration and metadata
│   └── assets/                 # Pre-generated artifacts for pipeline walkthrough replay
│       ├── baseline/           # Training runtime metrics (baseline_metrics.json)
│       ├── eda/                # Distribution figures and EDA summary report
│       ├── evaluation/         # Test evaluation metrics, per-country F1, confusion matrix
│       └── preprocess/         # Class balance and split counts (preprocess_stats.json)
├── archive/                    # Raw source data directory (Kaggle dataset destination)
│   └── LICENSE.txt             # Apache License 2.0 for raw dataset
├── data/                       # Processed datasets, split arrays, and lookup mappings
│   └── download_link.txt       # URL pointer to the raw Kaggle source dataset
├── eda/                        # Generated EDA plots, summary JSON, and report markdown
└── models/                     # Saved model artifacts, TF-IDF vectorizer, and test plots
    └── plots/baselines/        # Evaluation charts (confusion matrix, comparison bar charts)
```

### Purpose of Important Files & Folders
- **`main.py`**: High-level runner that executes stages 1 to 5 sequentially in a single invocation.
- **`pipeline/helper/common.py`**: The single source of truth for text normalization (`normalize_name`), character mapping (`ALPHABET`), and integer index featurization (`name_to_indices`). Guarantees zero train-inference skew.
- **`pipeline/eda01.py`**: Streams raw tables without exhausting RAM to analyze token counts, distinct name distributions, and multi-country overlaps.
- **`pipeline/preprocess02.py`**: Samples forenames and surnames according to country-specific probability weights to synthesize full names, applies a 70/15/15 stratified split, and exports `.csv` and `.npy` arrays.
- **`pipeline/baseline03.py`**: Fits character $n$-gram TF-IDF vectorizers (`char_wb`, range 2–4, 50,000 features) and trains Logistic Regression, LinearSVC, and SGDClassifier models.
- **`pipeline/evaluation04.py`**: Computes strict metrics (macro-F1, accuracy, top-$k$, MRR, NLL) and ambiguity-aware set-lenient accuracy on 776,784 unseen test samples.
- **`pipeline/predict05.py`**: Standalone CLI inference tool with interactive prompts or argument flags, outputting probability bars or JSON.
- **`app/`**: Self-contained web deployment folder designed to be hosted on Hugging Face Spaces or run locally via `python app/app.py`.

---

## 11. How It Works

The complete machine learning workflow operates across six well-defined steps:

```
[Raw Input Name]
       │
       ▼
1. Unicode Normalization (NFD decompose, strip accents, lowercase, filter [a-z ])
       │
       ▼
2. Dataset Generation (Monte Carlo weighted forename × surname pairing + 70/15/15 split)
       │
       ▼
3. Feature Extraction (Character word-boundary TF-IDF n-grams: 2 to 4 chars, 50k features)
       │
       ▼
4. Model Classification (Logistic Regression / Linear SVM / SGD with balanced weights)
       │
       ▼
5. Probability Scoring & Ranking (Softmax / calibrated margins → Top-K candidate sorting)
       │
       ▼
6. Evaluation & Output (Strict Macro-F1 + Ambiguity-Aware Set-Lenient Top-1/Top-3 metrics)
```

### Step 1: Input & Unicode Normalization
When a user provides a raw input string (e.g., `"Émilie François"`):
- The string passes through `normalize_name()` in `pipeline/helper/common.py`.
- **Unicode NFD Decomposition**: Deconstructs accented characters into base letters and combining diacritical marks.
- **Mark Stripping & Lowercasing**: Discards non-spacing marks and converts characters to lowercase.
- **Character Filtering**: Retains only English alphabetic characters `a–z` and spaces; multiple consecutive spaces are collapsed into single spaces.
- **Normalized Output**: `"emilie francois"`.

### Step 2: Preprocessing & Synthetic Data Construction
Because raw international data exists only as separate frequency lists of forenames and surnames per country rather than pre-joined full names:
- `pipeline/preprocess02.py` streams forename counts (12.4M rows) and surname counts (21.1M rows).
- Calculates country-specific token probability distributions: $P(\text{token}) = \frac{\text{count}}{\sum \text{counts}}$.
- Performs frequency-weighted Monte Carlo sampling of forenames and surnames to synthesize realistic full names matching real-world naming practices.
- Enforces a cap of 50,000 synthetic names per country, compressing severe real-world imbalance into an even **1.75×** ratio across 104 countries.
- Performs a stratified **70% train (3,624,986)** / **15% val (776,782)** / **15% test (776,784)** split.
- Indexes 174,564 multi-country ambiguous names into `name_country_map.json`.

### Step 3: Feature Extraction (Character $n$-gram TF-IDF)
- `TfidfVectorizer` is configured with:
  - **Analyzer**: `char_wb` (character $n$-grams generated only from text within word boundaries, padded by whitespace).
  - **$N$-gram Range**: `(2, 4)` (captures bi-grams, tri-grams, and 4-grams such as `" vi"`, `"wid"`, `"ido"`, `"do "`).
  - **Max Features**: `50,000` highest-frequency vocabulary tokens.
  - **Sublinear Term Frequency**: Applied (`sublinear_tf=True`) to temper the influence of repetitive character patterns.

### Step 4: Model Training & Benchmarking
Three linear classifiers are trained on the extracted TF-IDF matrix using balanced class weighting:
1. **Multinomial Logistic Regression (`LogisticRegression`)**:
   - Solver: `saga` (handles large sparse matrices efficiently)
   - Regularization: $C=5.0$, `max_iter=200`
   - Training time: **~169.1 seconds** on CPU.
2. **Linear Support Vector Classifier (`LinearSVC`)**:
   - Regularization: $C=1.0$, `dual=False`
   - Training time: **~485.6 seconds** on CPU.
3. **Stochastic Gradient Descent (`SGDClassifier`)**:
   - Loss: `modified_huber` (smooth, calibrated probabilistic loss)
   - Penalty: $\alpha = 10^{-5}$, `max_iter=30`
   - Training time: **~9.5 seconds** on CPU.

### Step 5: Prediction & Ranking
- For a query name, the saved TF-IDF vectorizer projects the normalized text into a 50,000-dimensional sparse feature vector.
- The chosen classifier evaluates the vector against all 104 country hyperplanes:
  - **Logistic Regression & SGD**: Direct calculation of class posterior probabilities via `predict_proba`.
  - **LinearSVC**: Softmax transformation over decision function margins ($e^{s_i - \max(s)} / \sum e^{s_j - \max(s)}$).
- The resulting probability vector is sorted in descending order to extract the Top-$k$ candidate countries, mapped to full country names via `country_codes.csv`.

### Step 6: Evaluation & Ambiguity-Aware Performance
Models are scored on the 776,784 unseen test samples:
- **Strict Macro-F1**: Evaluates performance treating all 104 classes equally regardless of frequency.
  - **Logistic Regression**: **0.501** Macro-F1 | **51.1%** Accuracy | **69.3%** Top-3 | **77.1%** Top-5 | **0.629** MRR
  - **Linear SVM**: **0.495** Macro-F1 | **51.3%** Accuracy | **67.9%** Top-3 | **74.9%** Top-5 | **0.622** MRR
  - **SGD Classifier**: **0.478** Macro-F1 | **50.0%** Accuracy | **67.3%** Top-3 | **74.6%** Top-5 | **0.611** MRR
- **Ambiguity-Aware (Set-Lenient) Metric**: Many names legitimately span multiple countries (e.g., *Ali* spans 30 countries). The evaluation script checks if the top prediction belongs to *any* verified country associated with that name:
  - Logistic Regression Set-Lenient Top-1 rises to **55.1%** (+4.1% over strict accuracy), and Set-Lenient Top-3 reaches **72.8%**.

---

## 12. Group Members

This project was developed by **Group 03**:

| Name | Student ID |
|---|---|
| **Brian Nicholas Tedjo** | 2802403183 |
| **Jason Budiharjo** | 2802419446 |
| **Marvin Adriano Rusdianto** | 2802402275 |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.  
The underlying name frequency dataset is licensed under the [Apache License 2.0](archive/LICENSE.txt).
