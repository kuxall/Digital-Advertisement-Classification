# 🚀 AdClassifier Pro

**AdClassifier Pro** is an enterprise-grade Machine Learning application designed to intelligently categorize digital advertisement text. Built with Streamlit and Scikit-learn, it offers a modern SaaS-like experience for both single-text analysis and bulk data processing.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)

## ✨ Key Features

### 📊 Executive Dashboard
- **Real-time Metrics**: View total ads, unique categories, model accuracy, and versioning.
- **Data Visualization**: Interactive charts showing the distribution of ad categories in the training dataset.

### 🔍 Intelligent Classifier
- **Real-time Analysis**: Paste any job description or ad text to get an instant classification.
- **Smart Logic**:
    - **High Confidence**: Displays the predicted category with a visual confidence bar.
    - **Low Confidence Handling**: If the model's confidence is below **60%**, the system flags the input as "Unclear/Random" rather than making a potentially incorrect guess. This ensures reliability for end-users.

### 📂 Batch Processor
- **Bulk Operations**: Upload `.csv` or `.xlsx` files containing thousands of ad texts.
- **Automated Labeling**: Processing engine classifies every row.
- **Uncertainty Detection**: Rows with low confidence scores are explicitly marked as `Uncertain/Random`.
- **Analytics**: Auto-generated reports on the distributed categories within your batch.
- **Export**: Download the fully labeled dataset in one click.

### 🎨 Premium UI/UX
- **Dark Mode**: Sleek, high-contrast design for reduced eye strain.
- **Responsive**: Fully functional on desktop and tablet sizes.
- **Interactive Outcomes**: clear, card-based result displays.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Visualization**: Streamlit native charts

---

## 🚀 Installation & Setup

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.8 or higher

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Digital-Advertisement-Classification
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run streamlit_app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration

The application allows for some configuration directly in the code logic.

### Confidence Threshold
In `streamlit_app.py`, you can adjust the sensitivity of the "Smart Logic" by modifying the `CONFIDENCE_THRESHOLD` constant at the top of the file.

```python
# Default is 60.0%
CONFIDENCE_THRESHOLD = 60.0
```

---

## 📂 Project Structure

```
├── data/
│   └── ConcatenatedDigitalAdData.xlsx  # Training dataset
├── model/
│   └── adv_model.sav                   # Pre-trained ML model
├── notebook/
│   └── ...                             # Research notebooks
├── streamlit_app.py                    # Main Application
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## ❓ FAQ & Troubleshooting

**Q: The Classifier keeps saying "Input unclear".**
A: This happens when the text provided is too short, gibberish, or completely unrelated to known ad categories. The model confidence is below 60%. Try providing more descriptive text.

**Q: "FileNotFoundError" on startup.**
A: Ensure you are running the `streamlit run` command from the root directory of the project, so it can find `model/adv_model.sav` and `data/`.

**Q: Can I train my own model?**
A: Yes, check the `notebook/` directory for the training scripts used to generate the `.sav` file.
