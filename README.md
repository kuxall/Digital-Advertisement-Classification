# 🚀 AdClassifier Pro

> **Paste any ad text → get an instant AI-powered category prediction.**
> No coding knowledge needed to run this app!

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Accuracy](https://img.shields.io/badge/Model_Accuracy-97.3%25-brightgreen)
![Version](https://img.shields.io/badge/Version-v3.0-blueviolet)

---

## 🟢 Quick Start — Run the App in 5 Steps

> **Never used Python before? Start here. Follow each step exactly.**

---

### ✅ Step 1 — Install Python

1. Go to **https://www.python.org/downloads**
2. Click the big yellow **"Download Python"** button
3. Run the installer
4. ⚠️ **IMPORTANT:** On the first screen, check the box that says **"Add Python to PATH"** before clicking Install

To verify it worked, open a terminal (search `cmd` in your Start Menu) and type:
```
python --version
```
You should see something like `Python 3.11.0`. If you do, move to Step 2.

---

### ✅ Step 2 — Download This Project

**Option A — If you have Git installed:**
```bash
git clone <repository-url>
cd Digital-Advertisement-Classification
```

**Option B — No Git? Download as ZIP:**
1. Click the green **"Code"** button on GitHub → **"Download ZIP"**
2. Extract the ZIP to a folder (e.g. `C:\Projects\AdClassifier`)
3. Open that folder

---

### ✅ Step 3 — Open a Terminal in the Project Folder

**Windows:**
1. Open File Explorer and navigate to the project folder
2. Click the address bar at the top, type `cmd`, press Enter
3. A black terminal window will open — you're in the right place ✅

**Mac / Linux:**
1. Open Terminal
2. Type `cd ` (with a space), then drag the project folder into the terminal window, press Enter

---

### ✅ Step 4 — Install the Requirements

Copy and paste this command into your terminal, then press Enter:

```bash
pip install -r requirements.txt
```

This will automatically install everything the app needs. It may take 1–2 minutes.

> 💡 If you see a `pip not found` error, try `python -m pip install -r requirements.txt` instead.

---

### ✅ Step 5 — Run the App

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically at **http://localhost:8501** 🎉

> 💡 If the browser doesn't open, manually go to **http://localhost:8501** in Chrome or Firefox.

---

### 🛑 How to Stop the App

Go back to the terminal and press **`Ctrl + C`**.

---

## 🖥️ Using the App

Once it's running, you'll see 3 pages in the left sidebar:

### 📊 Dashboard
An overview of the model — total ads, accuracy, and a chart of the training data categories.

### 🔍 Classifier ← *Start here*
1. Click **"Classifier"** in the sidebar
2. Paste any advertisement or job description text into the box
3. Click **"🚀 Analyze Now"**
4. The app will show which category it belongs to and a confidence score

**Example input you can try:**
```
We are looking for a Software Engineer with experience in Python and AWS.
The ideal candidate will troubleshoot and maintain computer systems and networks.
```

### 📂 Batch Processor
Upload a `.csv` or `.xlsx` file with ad text in one column — the app classifies every row at once and lets you download the results.

---

## 🏷️ Categories the App Recognizes

| Category | Examples |
|---|---|
| **Jobs – IT** | Software engineers, network admins, developers, IT support |
| **Jobs – Retail** | Cashiers, store associates, retail managers |
| **Banking** | Loan officers, financial advisors, bank tellers |
| **Sell – House** | Houses for sale, real estate listings |
| **Rent – Apartment** | Apartments for rent, studio listings |

---

## 🤖 How It Works (Plain English)

1. Your text gets cleaned (lowercased, punctuation removed)
2. It's converted into numbers using a technique called **TF-IDF** (counts how often important words appear)
3. A **Support Vector Machine** model — trained on 2,805 ad samples — predicts which category it belongs to
4. A confidence score (0–100%) is shown. If it's below 60%, the app says "unclear" instead of guessing wrong

---

## 📁 What's in This Project

```
AdClassifier Pro/
│
├── streamlit_app.py        ← The app you run
├── retrain_model.py        ← Script to regenerate + retrain the model
├── requirements.txt        ← List of Python packages needed
├── README.md               ← This guide
│
├── data/
│   ├── ConcatenatedDigitalAdData.xlsx   ← Original training data
│   └── synthetic_data.csv              ← Extra generated training samples
│
└── notebook/
    └── model/
        └── adv_model.sav   ← The trained AI model file
```

---

## ⚙️ Optional: Retrain the Model

> You only need this if you want to improve or update the model. **Skip this if you just want to use the app.**

```bash
python retrain_model.py
```

This will:
1. Generate 1,270 new synthetic training samples
2. Merge them with the original dataset
3. Retrain the classifier
4. Save the new model (the old one is backed up automatically)

---

## 🛠️ Tech Stack

| What | Tool |
|---|---|
| Web App | Streamlit |
| ML Model | Scikit-learn (LinearSVC + TF-IDF) |
| Data | Pandas, NumPy |
| Styling | Custom CSS, Google Fonts (Inter) |

---

## 🤖 Model Performance (v3.0)

| Category | F1-Score |
|---|---|
| Jobs – IT | 0.97 |
| Jobs – Retail | 0.96 |
| Banking | 0.97 |
| Rent – Apartment | 0.97 |
| Sell – House | 1.00 |
| **Overall Accuracy** | **97.3%** |

---

## ❓ Troubleshooting

**"streamlit is not recognized"**
> Your virtual environment isn't active, or Streamlit wasn't installed. Re-run:
> `pip install -r requirements.txt`

**"No module named streamlit"**
> Same fix — run `pip install -r requirements.txt`

**"FileNotFoundError" when the app starts**
> Make sure your terminal is inside the project folder (the one containing `streamlit_app.py`), not a subfolder.

**The classifier keeps saying "Input unclear"**
> Your text might be too short or too vague. Try pasting a longer, more detailed description with specific keywords related to the job or listing.

**The browser didn't open automatically**
> Manually go to **http://localhost:8501** in your browser.
