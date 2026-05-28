"""Professional Streamlit UI for sentiment classification"""

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
    .negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint (change to your deployed API)
API_URL = "http://localhost:8000"

# Sidebar
with st.sidebar:
    st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg")
    st.title("🎬 IMDB Sentiment Classifier")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app uses **BERT** (Bidirectional Encoder Representations from Transformers) "
        "fine-tuned on the IMDB dataset to classify movie reviews as positive or negative."
    )
    st.markdown("### How to use")
    st.markdown("1. Enter a movie review in the text box")
    st.markdown("2. Click 'Analyze Sentiment'")
    st.markdown("3. View results with confidence score")
    
    st.markdown("---")
    st.markdown("### Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV with 'review' column", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            if st.button("Process Batch"):
                with st.spinner("Processing..."):
                    # Send batch to API
                    response = requests.post(f"{API_URL}/predict-batch", json={"texts": df['review'].tolist()})
                    if response.status_code == 200:
                        results = response.json()
                        df['sentiment'] = [p['label'] for p in results['predictions']]
                        df['confidence'] = [p['confidence'] for p in results['predictions']]
                        st.dataframe(df)
                        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
                    else:
                        st.error("API Error")
        else:
            st.error("CSV must contain 'review' column")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter your movie review")
    review_text = st.text_area(
        "Movie Review",
        height=200,
        placeholder="Example: This movie was absolutely fantastic! The acting was brilliant and the story kept me engaged throughout..."
    )
    
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.markdown("### 📊 Quick Stats")
    st.metric("Model", "BERT-base-uncased")
    st.metric("Accuracy", "92.5%")
    st.metric("F1 Score", "0.92")

# Analysis
if analyze_button and review_text:
    with st.spinner("Analyzing sentiment..."):
        start_time = time.time()
        
        # Call API
        response = requests.post(f"{API_URL}/predict", json={"text": review_text})
        
        if response.status_code == 200:
            result = response.json()
            latency = (time.time() - start_time) * 1000
            
            # Display result
            col_result, col_gauge = st.columns(2)
            
            with col_result:
                if result['label'] == 'positive':
                    st.markdown(f"""
                    <div class="positive">
                        <h2>😊 POSITIVE</h2>
                        <p>Confidence: {result['confidence']:.2%}</p>
                        <p>Latency: {latency:.0f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="negative">
                        <h2>😞 NEGATIVE</h2>
                        <p>Confidence: {result['confidence']:.2%}</p>
                        <p>Latency: {latency:.0f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_gauge:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['confidence'] * 100,
                    title={'text': "Confidence Score (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f8d7da"},
                            {'range': [50, 100], 'color': "#d4edda"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['confidence'] * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability distribution
            st.markdown("### 📈 Probability Distribution")
            if 'probabilities' in result:
                probs = result['probabilities']
                fig = go.Figure(data=[
                    go.Bar(name='Negative', x=['Negative'], y=[probs['negative'] * 100], marker_color='#f8d7da'),
                    go.Bar(name='Positive', x=['Positive'], y=[probs['positive'] * 100], marker_color='#d4edda')
                ])
                fig.update_layout(yaxis_title="Probability (%)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Examples
            with st.expander("💡 Try these examples"):
                examples = [
                    "An absolute masterpiece! I was moved to tears.",
                    "Terrible acting and boring plot. Waste of time.",
                    "It was okay, some good moments but nothing special.",
                    "Best movie I've seen all year! Must watch!"
                ]
                for ex in examples:
                    if st.button(ex, key=ex[:20]):
                        st.session_state.review_text = ex
                        st.rerun()
                        
        else:
            st.error(f"API Error: {response.status_code}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using BERT, FastAPI, and Streamlit")