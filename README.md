# AdClassifier Pro 🚀

> **Intelligent Digital Advertisement Classification System**  
> Automatically categorises job ads, house listings, apartment rentals, retail postings, and banking ads using multiple machine learning algorithms.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Categories](#categories)
3. [Dataset](#dataset)
4. [Algorithm Comparison](#algorithm-comparison)
5. [Why Each Algorithm?](#why-each-algorithm)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Running the App](#running-the-app)
9. [Retraining the Model](#retraining-the-model)
10. [Results Summary](#results-summary)

---

## Project Overview

AdClassifier Pro is a full-stack NLP classification application that:

- **Classifies** digital advertisements into 5 categories in real-time
- **Compares** 6 different ML algorithms with metrics, charts, and confusion matrices
- **Provides** a Streamlit web dashboard for single-input classification, batch CSV/Excel processing, and algorithm comparison
- **Handles class imbalance** via synthetic data augmentation and `class_weight="balanced"`

The best-performing model (LinearSVC) is automatically saved and used in production.

---

## Categories

| Category | Description |
|---|---|
| `Jobs – IT` | Software engineering, DevOps, cloud, sysadmin, cybersecurity roles |
| `Jobs – Retail` | Cashier, sales associate, store manager, stock associate roles |
| `Banking` | Loan officer, financial advisor, credit analyst, accountant roles |
| `Sell – House` | Property listings with bedrooms, bathrooms, price, features |
| `Rent – Apartment` | Apartment rental ads with rent price, amenities, availability |

---

## Dataset

| Source | Samples |
|---|---|
| Original (`ConcatenatedDigitalAdData.xlsx`) | ~1,541 |
| Synthetic (generated via `retrain_model.py`) | ~1,270 |
| **Total after merging** | **2,805** |

**Class distribution after balancing:**

```
Sell – House        607
Jobs – Retail       552
Rent – Apartment    550
Jobs – IT           549
Banking             547
```

Train / Test split: **80% / 20%** (stratified)

---

## Algorithm Comparison

Six algorithms were trained and evaluated on identical data splits. Results are ranked by **Test Accuracy**:

| Rank | Algorithm | Test Accuracy | Macro F1 | Weighted F1 | CV Mean (5-fold) | CV Std | Train Time |
|---|---|---|---|---|---|---|---|
| 🥇 1 | **LinearSVC** *(Production)* | **97.33%** | **97.28%** | **97.32%** | **97.54%** | ±0.35% | 0.8s |
| 🥈 2 | Logistic Regression | 96.97% | 96.91% | 96.97% | 97.54% | ±0.58% | 1.5s |
| 🥈 2 | Random Forest | 96.97% | 96.94% | 96.98% | 96.47% | ±0.67% | 1.4s |
| 4 | Multinomial Naive Bayes | 96.79% | 96.73% | 96.79% | 97.08% | ±0.49% | **0.3s** |
| 5 | Gradient Boosting | 95.37% | 95.35% | 95.40% | 96.04% | ±0.73% | 65.9s |
| 6 | K-Nearest Neighbors | 94.65% | 94.54% | 94.63% | 95.22% | ±0.86% | 0.5s |

> All algorithms use TF-IDF vectorization (bigrams, 20k max features) as input features.

---

## Why Each Algorithm?

### ⚔️ LinearSVC *(Current Production Model)*
**Why chosen:** Linear SVM is the gold standard for high-dimensional sparse text classification. TF-IDF produces very large sparse feature vectors where linear boundaries separate classes cleanly. `CalibratedClassifierCV` wraps it to produce probability estimates. It achieves the best accuracy (97.33%) with the second-fastest training time (0.8s), making it ideal for production.

### 📈 Logistic Regression
**Why included:** A natural probabilistic alternative to SVM. Outputs well-calibrated class probabilities without wrapping. Tied for 2nd place at 96.97% with similar CV performance to LinearSVC (97.54%). Highly interpretable — coefficients directly indicate which words drive each classification.

### 🧮 Multinomial Naive Bayes
**Why included:** The classic NLP baseline algorithm. Assumes conditional independence of features given the class. Extremely fast (0.3s training) and achieved 96.79% — impressively close to the top models. Best choice when training resources are severely limited or real-time retraining is needed.

### 🌲 Random Forest
**Why included:** A bagging ensemble of 200 decision trees. Handles non-linear feature interactions and is naturally resistant to overfitting. Tied for 2nd place (96.97%) but has a lower CV score (96.47%), suggesting it generalizes slightly less consistently than linear models on text data.

### 🚀 Gradient Boosting
**Why included:** A powerful sequential boosting ensemble that iteratively corrects prior model errors. Strong on structured tabular data. Demonstrated here to show the accuracy–speed trade-off: only 95.37% accuracy while taking 65.9 seconds to train — far slower than linear models on text.

### 📍 K-Nearest Neighbors
**Why included:** A non-parametric, instance-based learner. Classifies new samples by voting from the k=7 nearest training examples using cosine similarity. Included as a simple distance-based baseline. Performs weakest (94.65%) due to the curse of dimensionality in high-dimensional TF-IDF space.

---

## Project Structure

```
Digital-Advertisement-Classification/
│
├── streamlit_app.py          # Main Streamlit web application
├── retrain_model.py          # Generates synthetic data & retrains LinearSVC
├── compare_algorithms.py     # Trains & compares all 6 algorithms
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── ConcatenatedDigitalAdData.xlsx  # Original labelled dataset
│   ├── synthetic_data.csv              # Augmented synthetic training samples
│   └── comparison_report/             # Auto-generated by compare_algorithms.py
│       ├── algorithm_comparison.csv    # Numerical results table
│       ├── algorithm_reasons.txt       # Text explanations per algorithm
│       ├── accuracy_comparison.png     # Bar chart: accuracy vs CV
│       ├── f1_comparison.png           # Bar chart: Macro F1 vs Weighted F1
│       ├── training_time.png           # Horizontal bar: training speed
│       ├── radar_comparison.png        # Radar chart: multi-metric overview
│       ├── cm_LinearSVC_Current.png    # Confusion matrix per algorithm
│       └── cm_*.png                   # (one per algorithm)
│
└── notebook/
    └── model/
        ├── adv_model.sav         # Active production model (best algorithm)
        └── adv_model_backup.sav  # Previous model backup
```

---

## Setup & Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/kuxall/Digital-Advertisement-Classification.git
cd Digital-Advertisement-Classification

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:
```
streamlit
scikit-learn
pandas
numpy
openpyxl
matplotlib
seaborn
```

---

## Running the App

```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` with 4 pages:

| Page | Description |
|---|---|
| **Dashboard** | Data distribution, total samples, category counts |
| **Classifier** | Real-time single-ad classification with confidence scores |
| **Batch Processor** | Upload CSV/Excel for bulk classification and download |
| **Model Comparison** | Interactive charts and rankings for all 6 algorithms |

---

## Retraining the Model

### Retrain with synthetic data augmentation (LinearSVC only):
```bash
python retrain_model.py
```

### Run full multi-algorithm comparison + save best model:
```bash
python compare_algorithms.py
```

This will:
1. Train all 6 algorithms on the same data split
2. Print a full comparison table to the console
3. Save charts and CSVs to `data/comparison_report/`
4. Automatically save the **best model** to `notebook/model/adv_model.sav`
5. Back up the previous model to `adv_model_backup.sav`

After running, restart the Streamlit app — it will load the new best model automatically.

---

## Results Summary

The system achieves production-grade accuracy with a highly balanced dataset:

- **Best Model:** LinearSVC + TF-IDF (bigrams)
- **Test Accuracy:** 97.33%
- **5-Fold CV Accuracy:** 97.54% (±0.35%)
- **Training Samples:** 2,805 (real + synthetic)
- **Categories:** 5
- **Training Time:** < 1 second

> The linear models (LinearSVC, Logistic Regression) consistently outperform ensemble and distance-based methods on TF-IDF text features, confirming well-established NLP research findings. Naive Bayes is the recommended lightweight alternative if inference speed is critical.

---

*Built with Python · scikit-learn · Streamlit · TF-IDF · LinearSVC*
