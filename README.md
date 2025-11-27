# 🚀 AdClassifier Pro

**AdClassifier Pro** is an enterprise-grade Machine Learning application designed to intelligently categorize digital advertisement text. Built with Streamlit and Scikit-learn, it offers a modern SaaS-like experience for both single-text analysis and bulk data processing.

## ✨ Key Features

- **📊 Executive Dashboard**: Get a high-level overview of your data distribution and model performance metrics.
- **🔍 Intelligent Classifier**:
    - Real-time classification of job descriptions.
    - **Confidence Scores**: Visual indicators of model certainty.
- **📂 Batch Processor**:
    - Bulk upload support for **CSV** and **Excel** files.
    - Automatic column detection.
    - **Instant Analytics**: Auto-generated charts for category distribution and confidence spread.
    - Export results to CSV.
- **🎨 Customization**:
    - Toggle between **Midnight Dark** and **Professional Light** themes.
    - Responsive, mobile-friendly UI.

## 🛠️ Installation

1.  **Install Python**: Ensure you have Python 3.8+ installed.
2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the application locally:

```bash
streamlit run streamlit_app.py
```

### How to Use
1.  **Dashboard**: View the model's training stats.
2.  **Classifier**: Go to the "Classifier" tab, paste an ad, and click "Analyze Now".
3.  **Batch Processing**:
    - Go to "Batch Processor".
    - Upload a CSV/Excel file (e.g., `test_ads.csv`).
    - Select the text column.
    - Click "Process Batch" to see charts and download results.

## 📂 Project Structure

- `streamlit_app.py`: Main application logic and UI.
- `model/`: Contains the pre-trained machine learning model (`adv_model.sav`).
- `data/`: Contains the dataset used for training/visualization.
- `notebook/`: Jupyter notebooks used for model development.
