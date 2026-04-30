import numpy as np
import pandas as pd
import pickle
import streamlit as st
import warnings
import time
import re
import os

warnings.filterwarnings("ignore")

CONFIDENCE_THRESHOLD = 60.0

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdClassifier Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ─────────────────────────────────────────────────────────────────────
def apply_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #f8fafc;
        }
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #334155;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc;
            font-weight: 700;
        }
        .stButton button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -1px rgba(59, 130, 246, 0.6);
        }
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .result-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .preview-card {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .job-tag {
            display: inline-block;
            background-color: #334155;
            color: #f8fafc;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            margin: 0.25rem;
            border: 1px solid #334155;
        }
    </style>
    """, unsafe_allow_html=True)

# ── DATA & MODEL LOADING ──────────────────────────────────────────────────────
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

@st.cache_resource
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

# ── HELPERS ───────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_with_proba(model, text):
    """Return (prediction, confidence%, proba_array, classes)."""
    cleaned = clean_text(text)
    classes = model.classes_
    proba = model.predict_proba([cleaned])[0]
    best_idx = int(np.argmax(proba))
    return classes[best_idx], float(proba[best_idx]) * 100, proba, classes

# ── PAGES ─────────────────────────────────────────────────────────────────────
def dashboard_page(df):
    st.title("📊 Executive Dashboard")
    st.markdown("Overview of model performance and data distribution.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Ads in DB</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{df["JobType"].nunique()}</h3><p>Categories</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>97.3%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>v3.0</h3><p>Model Version</p></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Category Distribution")
    st.bar_chart(df['JobType'].value_counts())


def classifier_page(model, job_types):
    st.title("🔍 Intelligent Classifier")
    st.markdown("Analyze single job descriptions in real-time.")

    col1, col2 = st.columns([2, 1])

    with col1:
        input_text = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste ad text here...",
            label_visibility="collapsed"
        )

        if st.button("🚀 Analyze Now", use_container_width=True):
            if input_text.strip():
                with st.spinner("Processing with AI..."):
                    time.sleep(0.5)
                    prediction, confidence, proba, classes = predict_with_proba(model, input_text)

                if confidence < CONFIDENCE_THRESHOLD:
                    st.error("⚠️ Input unclear. Please provide a valid job description.")
                else:
                    # Main result card
                    st.markdown(f"""
                    <div class="result-card">
                        <p style="color:#94a3b8;font-size:0.9rem;text-transform:uppercase;letter-spacing:1px;">Identified Category</p>
                        <h1 style="color:#3b82f6;font-size:3rem;margin:0.5rem 0;">{prediction}</h1>
                        <div style="margin-top:1.5rem;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                                <span>Confidence Score</span>
                                <span>{confidence:.1f}%</span>
                            </div>
                            <div style="background-color:#334155;border-radius:9999px;height:8px;width:100%;">
                                <div style="background-color:#3b82f6;width:{min(confidence,100):.1f}%;height:100%;border-radius:9999px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # All-categories breakdown — each row its own st.markdown to avoid escaping
                    st.markdown(
                        '<p style="color:#94a3b8;font-size:0.8rem;text-transform:uppercase;'
                        'letter-spacing:1px;margin-top:1.4rem;margin-bottom:0.6rem;">All Categories</p>',
                        unsafe_allow_html=True
                    )
                    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                        pct = p * 100
                        bar_color = "#3b82f6" if cls == prediction else "#475569"
                        label_color = "#f8fafc" if cls == prediction else "#94a3b8"
                        pct_color = "#3b82f6" if cls == prediction else "#94a3b8"
                        st.markdown(
                            f'<div style="margin-bottom:0.6rem;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:3px;">'
                            f'<span style="color:{label_color}">{cls}</span>'
                            f'<span style="color:{pct_color}">{pct:.1f}%</span>'
                            f'</div>'
                            f'<div style="background-color:#1e293b;border-radius:9999px;height:6px;">'
                            f'<div style="background-color:{bar_color};width:{pct:.1f}%;height:100%;border-radius:9999px;"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )
            else:
                st.warning("⚠️ Please enter text to analyze.")

    with col2:
        st.markdown("### 🏷️ Supported Categories")
        st.markdown('<div style="display:flex;flex-wrap:wrap;">', unsafe_allow_html=True)
        for job in job_types:
            st.markdown(f'<span class="job-tag">{job}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def batch_page(model):
    st.title("📂 Batch Processor")
    st.markdown("Bulk classify thousands of ads via CSV/Excel.")

    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            with st.expander("📄 View Raw Input Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

            col_name = st.selectbox("Select Text Column", df.columns)

            if st.button("⚡ Process Batch", use_container_width=True):
                with st.spinner("Crunching numbers..."):
                    def predict_row(text):
                        try:
                            pred, conf, _, _ = predict_with_proba(model, str(text))
                            return pred, conf
                        except:
                            return "Error", 0.0

                    results = df[col_name].apply(predict_row)
                    df['Predicted_Category'] = results.apply(lambda x: x[0])
                    df['Confidence'] = results.apply(lambda x: x[1])
                    df.loc[df['Confidence'] < CONFIDENCE_THRESHOLD, 'Predicted_Category'] = "Uncertain/Random"

                st.success("Processing Complete!")

                st.markdown("### ✨ Results Preview")
                cols = st.columns(3)
                for i, row in df.head(3).iterrows():
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="preview-card">
                            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">Ad #{i+1}</div>
                            <div style="font-weight:bold;color:#3b82f6;font-size:1.1rem;margin-bottom:0.5rem;">{row['Predicted_Category']}</div>
                            <div style="font-size:0.85rem;color:#cbd5e1;overflow:hidden;height:4.5em;">
                                {str(row[col_name])[:100]}...
                            </div>
                            <div style="margin-top:0.8rem;font-size:0.8rem;color:#22c55e;">
                                Confidence: {row['Confidence']:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### 📊 Batch Insights")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Category Distribution**")
                    st.bar_chart(df['Predicted_Category'].value_counts())
                with c2:
                    st.markdown("**Confidence Spread**")
                    st.line_chart(df['Confidence'])

                with st.expander("📋 View Full Results Table", expanded=True):
                    st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Report CSV",
                    csv,
                    "classified_batch.csv",
                    "text/csv",
                    key='download-csv',
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error: {e}")

# ── MODEL COMPARISON PAGE ────────────────────────────────────────────────────
ALGO_REASONS = {
    "LinearSVC (Current)": (
        "⚔️ LinearSVC",
        "High-dimensional linear classifier. Excellent for TF-IDF sparse text vectors. "
        "Fast training and inference. Class-weight balancing handles imbalanced data. "
        "This is the current production model."
    ),
    "Logistic Regression": (
        "📈 Logistic Regression",
        "Probabilistic linear model with naturally calibrated probability outputs. "
        "Highly interpretable, extremely effective on TF-IDF features, and stable. "
        "Strong alternative when confidence scores are critical."
    ),
    "Multinomial Naive Bayes": (
        "🧮 Naive Bayes",
        "Classic Bayesian text classifier. Assumes feature independence. Extremely fast "
        "and surprisingly competitive for bag-of-words/TF-IDF features. Best when "
        "training data is limited or speed is a priority."
    ),
    "Random Forest": (
        "🌲 Random Forest",
        "Ensemble of 200 decision trees using bagging. Handles non-linear patterns "
        "and feature interactions naturally. Less optimal on very high-dimensional "
        "sparse text but robust and resistant to overfitting."
    ),
    "Gradient Boosting": (
        "🚀 Gradient Boosting",
        "Sequential ensemble that corrects prior model errors. Powerful on structured "
        "data. Demonstrates the accuracy vs. training-time trade-off — typically much "
        "slower on text than linear models."
    ),
    "K-Nearest Neighbors": (
        "📍 KNN",
        "Instance-based learner that classifies by similarity to k nearest neighbours. "
        "Included as a simple non-parametric baseline. Weaker on high-dimensional "
        "TF-IDF vectors due to the curse of dimensionality."
    ),
}

CHART_LABELS = {
    "accuracy_comparison.png": "📊 Test Accuracy vs CV Accuracy",
    "f1_comparison.png": "🎯 Macro F1 vs Weighted F1",
    "training_time.png": "⏱️ Training Time per Algorithm",
    "radar_comparison.png": "🕸️ Multi-Metric Radar Chart",
}

def comparison_page():
    st.title("🔬 Model Comparison")
    st.markdown("Compare all trained algorithms across accuracy, F1 score, and training speed.")

    report_dir = "data/comparison_report"
    csv_path   = os.path.join(report_dir, "algorithm_comparison.csv")

    if not os.path.exists(csv_path):
        st.warning(
            "⚠️ No comparison report found. "
            "Run `python compare_algorithms.py` first to generate results."
        )
        st.code("python compare_algorithms.py", language="bash")
        return

    df = pd.read_csv(csv_path)
    best = df.iloc[0]

    # Banner
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#1e3a5f,#1e293b);border:1px solid #3b82f6;
                border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
        <p style="color:#94a3b8;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;
                  margin:0 0 0.5rem 0;">🏆 Best Performing Algorithm</p>
        <h2 style="color:#3b82f6;margin:0;">{best['Algorithm']}</h2>
        <p style="color:#f8fafc;margin:0.5rem 0 0 0;">
            Test Accuracy: <b style="color:#22c55e;">{best['Test Accuracy']:.2f}%</b> &nbsp;|&nbsp;
            Macro F1: <b style="color:#f59e0b;">{best['Macro F1']:.2f}%</b> &nbsp;|&nbsp;
            CV Mean: <b style="color:#a78bfa;">{best['CV Mean']:.2f}% ± {best['CV Std']:.2f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ranking table
    st.markdown("### 📋 Full Algorithm Rankings")
    styled = df.copy()
    styled.index = styled.index + 1  # 1-based rank
    styled.insert(0, "Rank", styled.index)
    styled = styled.drop(columns=["Rank"])
    st.dataframe(
        styled.style.format({
            "Test Accuracy": "{:.2f}%",
            "Macro F1": "{:.2f}%",
            "Weighted F1": "{:.2f}%",
            "CV Mean": "{:.2f}%",
            "CV Std": "{:.2f}%",
            "Train Time (s)": "{:.1f}s",
        }).background_gradient(subset=["Test Accuracy"], cmap="Blues"),
        use_container_width=True,
        hide_index=False,
    )

    # Charts
    st.markdown("### 📈 Visual Comparisons")
    chart_files = list(CHART_LABELS.keys())
    col1, col2 = st.columns(2)
    for i, fname in enumerate(chart_files):
        fpath = os.path.join(report_dir, fname)
        if os.path.exists(fpath):
            with (col1 if i % 2 == 0 else col2):
                st.markdown(f"**{CHART_LABELS[fname]}**")
                st.image(fpath, use_column_width=True)

    # Confusion matrices
    st.markdown("### 🗂️ Per-Algorithm Confusion Matrices")
    cm_files = sorted([f for f in os.listdir(report_dir) if f.startswith("cm_")])
    if cm_files:
        cm_cols = st.columns(2)
        for i, fname in enumerate(cm_files):
            fpath = os.path.join(report_dir, fname)
            algo_name = fname.replace("cm_", "").replace("_", " ").replace(".png", "")
            with cm_cols[i % 2]:
                st.markdown(f"**{algo_name}**")
                st.image(fpath, use_column_width=True)

    # Why each algorithm section
    st.markdown("### 💡 Why Each Algorithm?")
    for algo_key, (label, reason) in ALGO_REASONS.items():
        match = df[df["Algorithm"] == algo_key]
        acc_str = f"{match.iloc[0]['Test Accuracy']:.2f}%" if not match.empty else "N/A"
        rank_num = match.index[0] + 1 if not match.empty else "?"
        with st.expander(f"#{rank_num} {label}  —  Accuracy: {acc_str}"):
            st.markdown(f"**{label}**")
            st.write(reason)
            if not match.empty:
                r = match.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test Accuracy", f"{r['Test Accuracy']:.2f}%")
                c2.metric("Macro F1",      f"{r['Macro F1']:.2f}%")
                c3.metric("CV Mean",        f"{r['CV Mean']:.2f}%")
                c4.metric("Train Time",     f"{r['Train Time (s)']:.1f}s")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    apply_theme()

    data_path = "data/ConcatenatedDigitalAdData.xlsx"
    model_path = "notebook/model/adv_model.sav"

    try:
        df_data = load_data(data_path)
        model = load_model(model_path)
        job_types = df_data['JobType'].unique().tolist()
    except Exception as e:
        st.error(f"System Error: {e}")
        return

    with st.sidebar:
        st.title("🚀 AdClassifier Pro")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["Dashboard", "Classifier", "Batch Processor", "Model Comparison"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8;">
            <b style="color:#f8fafc;">v3.0</b> · Enterprise Edition<br>
            Accuracy: <b style="color:#22c55e;">97.3%</b><br>
            Training Samples: <b style="color:#f8fafc;">2,805</b><br>
            Categories: <b style="color:#f8fafc;">5</b>
        </div>
        """, unsafe_allow_html=True)

    if page == "Dashboard":
        dashboard_page(df_data)
    elif page == "Classifier":
        classifier_page(model, job_types)
    elif page == "Batch Processor":
        batch_page(model)
    elif page == "Model Comparison":
        comparison_page()


if __name__ == "__main__":
    main()
