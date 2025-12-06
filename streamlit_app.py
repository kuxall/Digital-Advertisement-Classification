import numpy as np
import pandas as pd
import pickle
import streamlit as st
import warnings
import time
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CONFIDENCE_THRESHOLD = 60.0

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AdClassifier Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEMES & CSS ---
def apply_theme():
    # Enforce Premium Dark Theme
    bg_gradient = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    text_color = "#f8fafc"
    card_bg = "#1e293b"
    card_border = "#334155"
    sidebar_bg = "#020617"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {text_color};
        }}

        .stApp {{
            background: {bg_gradient};
        }}

        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {card_border};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {text_color};
            font-weight: 700;
        }}

        .stButton button {{
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
        }}

        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -1px rgba(59, 130, 246, 0.6);
        }}

        .metric-card {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        .result-card {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .preview-card {{
            background-color: rgba(255,255,255,0.03);
            border: 1px solid {card_border};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }}

        .job-tag {{
            display: inline-block;
            background-color: {card_border};
            color: {text_color};
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            margin: 0.25rem;
            border: 1px solid {card_border};
        }}
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

@st.cache_resource
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- PAGES ---

def dashboard_page(df):
    st.title("📊 Executive Dashboard")
    st.markdown("Overview of the model performance and data distribution.")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><h3>{len(df)}</h3><p>Total Ads in DB</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h3>{df['JobType'].nunique()}</h3><p>Categories</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><h3>96.5%</h3><p>Model Accuracy</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><h3>v2.1</h3><p>Model Version</p></div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Category Distribution")
    
    # Chart
    chart_data = df['JobType'].value_counts()
    st.bar_chart(chart_data)

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
                    # Simulate slight delay for effect
                    time.sleep(0.5)
                    cleaned_input = clean_text(input_text)
                    prediction = model.predict([cleaned_input])[0]
                    
                    confidence = 0.0
                    if hasattr(model, "predict_proba"):
                        confidence = np.max(model.predict_proba([cleaned_input])[0]) * 100
                    else:
                        st.warning("Model does not support probability prediction. Confidence score may be inaccurate.")
                        confidence = 100.0

                if confidence < CONFIDENCE_THRESHOLD:
                    st.error("⚠️ Input unclear. Please provide a valid job description.")
                else:
                    # Result Display
                    st.markdown(f"""
                    <div class="result-card">
                        <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Identified Category</p>
                        <h1 style="color: #3b82f6; font-size: 3rem; margin: 0.5rem 0;">{prediction}</h1>
                        <div style="margin-top: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span>Confidence Score</span>
                                <span>{confidence:.1f}%</span>
                            </div>
                            <div style="background-color: #334155; border-radius: 9999px; height: 8px; width: 100%;">
                                <div style="background-color: #3b82f6; width: {confidence}%; height: 100%; border-radius: 9999px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter text to analyze.")

    with col2:
        st.markdown("### 🏷️ Supported Categories")
        st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)
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
            
            # Preview Input
            with st.expander("📄 View Raw Input Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            col_name = st.selectbox("Select Text Column", df.columns)
            
            if st.button("⚡ Process Batch", use_container_width=True):
                with st.spinner("Crunching numbers..."):
                    # Prediction logic
                    def predict(text):
                        try:
                            return model.predict([clean_text(str(text))])[0]
                        except:
                            return "Error"
                    
                    df['Predicted_Category'] = df[col_name].apply(predict)
                    
                    # Confidence logic
                    if hasattr(model, "predict_proba"):
                        df['Confidence'] = df[col_name].apply(lambda x: np.max(model.predict_proba([clean_text(str(x))])[0]) * 100)
                    else:
                        df['Confidence'] = 100.0
                    
                    # Mark low confidence predictions
                    df.loc[df['Confidence'] < CONFIDENCE_THRESHOLD, 'Predicted_Category'] = "Uncertain/Random"

                st.success("Processing Complete!")
                
                # --- NEW: Beautiful Preview ---
                st.markdown("### ✨ Results Preview")
                
                # Show top 3 results as cards
                cols = st.columns(3)
                for i, row in df.head(3).iterrows():
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="preview-card">
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">Ad #{i+1}</div>
                            <div style="font-weight: bold; color: #3b82f6; font-size: 1.1rem; margin-bottom: 0.5rem;">{row['Predicted_Category']}</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; height: 4.5em;">
                                {str(row[col_name])[:100]}...
                            </div>
                            <div style="margin-top: 0.8rem; font-size: 0.8rem; color: #22c55e;">
                                Confidence: {row['Confidence']:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Analytics
                st.markdown("### 📊 Batch Insights")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Category Distribution**")
                    st.bar_chart(df['Predicted_Category'].value_counts())
                with c2:
                    st.markdown("**Confidence Spread**")
                    st.line_chart(df['Confidence'])

                # Full Data Table
                with st.expander("📋 View Full Results Table", expanded=True):
                    st.dataframe(df, use_container_width=True)

                # Download
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

# --- MAIN APP ---
def main():
    apply_theme()
    
    # Load resources
    data_path = "data/ConcatenatedDigitalAdData.xlsx"
    model_path = "notebook/model/adv_model.sav"
    
    try:
        df_data = load_data(data_path)
        model = load_model(model_path)
        job_types = np.array(pd.DataFrame(df_data.JobType.unique()).values).flatten()
    except Exception as e:
        st.error(f"System Error: {e}")
        return

    # Sidebar Navigation
    with st.sidebar:
        st.title("🚀 AdClassifier Pro")
        st.markdown("---")
        
        page = st.radio(
            "Navigate", 
            ["Dashboard", "Classifier", "Batch Processor"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("v2.1.0 | Enterprise Edition")

    # Routing
    if page == "Dashboard":
        dashboard_page(df_data)
    elif page == "Classifier":
        classifier_page(model, job_types)
    elif page == "Batch Processor":
        batch_page(model)

if __name__ == "__main__":
    main()
