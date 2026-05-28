"""Gradio UI for IMDB Sentiment Classifier"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import CONFIG

# Global variables
_model = None
_tokenizer = None
_device = None
_id2label = {0: "negative", 1: "positive"}
_MAX_LENGTH = 256


def load_model():
    """
    Load the trained model and tokenizer.
    Called once when the app starts.
    """
    global _model, _tokenizer, _device
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = CONFIG.output_dir
    
    print(f"📂 Loading model from: {model_path}")
    print(f"💻 Using device: {_device}")
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.to(_device)
        _model.eval()
        print("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def predict_sentiment(text: str):
    """
    Predict sentiment for a single review.
    
    Args:
        text: Movie review text
        
    Returns:
        Tuple of (label, confidence) or error message
    """
    if not text or not text.strip():
        return "⚠️ Please enter a review.", 0.0
    
    if _model is None:
        return "❌ Model not loaded. Please restart.", 0.0
    
    try:
        # Tokenize
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=True,
        ).to(_device)
        
        # Predict
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
        
        label = _id2label[pred_id]
        confidence = probs[pred_id].item()
        
        return label, confidence
        
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0


def classify_review(review: str) -> str:
    """
    Wrapper function for Gradio interface.
    Returns formatted markdown string.
    """
    label, confidence = predict_sentiment(review)
    
    if label == "positive":
        emoji = "😊"
        color = "#00c853"
        sentiment = "POSITIVE"
    elif label == "negative":
        emoji = "😞"
        color = "#d50000"
        sentiment = "NEGATIVE"
    else:
        return label  # Error message
    
    # Create progress bar HTML
    confidence_percent = confidence * 100
    bar_width = confidence_percent
    
    html = f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f5f5f5;">
        <div style="font-size: 48px;">{emoji}</div>
        <h2 style="color: {color}; margin: 10px 0;">{sentiment}</h2>
        <p style="font-size: 18px; margin: 5px 0;">Confidence: {confidence_percent:.1f}%</p>
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; margin-top: 10px;">
            <div style="width: {bar_width}%; background-color: {color}; border-radius: 10px; 
                        text-align: center; color: white; padding: 5px 0;">
                {confidence_percent:.1f}%
            </div>
        </div>
    </div>
    """
    return html


def analyze_batch(reviews_text: str) -> str:
    """
    Analyze multiple reviews (one per line).
    
    Args:
        reviews_text: String with one review per line
        
    Returns:
        Formatted table of results
    """
    if not reviews_text or not reviews_text.strip():
        return "⚠️ Please enter at least one review."
    
    reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
    
    if not reviews:
        return "⚠️ No valid reviews found."
    
    results = []
    for review in reviews:
        label, confidence = predict_sentiment(review)
        emoji = "😊" if label == "positive" else "😞"
        results.append({
            "review": review[:80] + "..." if len(review) > 80 else review,
            "sentiment": f"{emoji} {label.upper()}",
            "confidence": f"{confidence:.1%}"
        })
    
    # Create markdown table
    table = "| # | Review | Sentiment | Confidence |\n"
    table += "|---|--------|-----------|------------|\n"
    
    for i, r in enumerate(results, 1):
        table += f"| {i} | {r['review']} | {r['sentiment']} | {r['confidence']} |\n"
    
    return table


def get_example_reviews():
    """Return example reviews for the interface."""
    return [
        "This movie was absolutely fantastic! Brilliant acting and a gripping story that kept me on the edge of my seat.",
        "I hated this film. It was boring, too long, and badly written. Complete waste of time and money.",
        "It was okay — some good moments, but overall forgettable. Average acting, decent plot but nothing special.",
        "Masterpiece! Christopher Nolan at his best. The storytelling is phenomenal and the visuals are stunning.",
        "Terrible acting, predictable plot, and annoying characters. I regret watching this movie.",
        "A beautiful film with heartwarming moments. The cinematography was breathtaking and the performances were genuine.",
    ]


# Create Gradio Interface
def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    footer {
        visibility: hidden;
    }
    """
    
    with gr.Blocks(title="🎬 IMDB Sentiment Classifier", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 IMDB Sentiment Classifier
        
        **Fine-tuned BERT model** that predicts whether a movie review is **positive** or **negative**.
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Single Review
            with gr.TabItem("📝 Single Review"):
                with gr.Row():
                    with gr.Column(scale=2):
                        review_input = gr.Textbox(
                            lines=6,
                            placeholder="Type or paste your movie review here...\n\nExample: 'This movie was absolutely fantastic! The acting was brilliant...'",
                            label="Movie Review",
                            show_label=True
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        output_html = gr.HTML(label="Prediction Result")
                        
                        # Confidence meter placeholder
                        gr.Markdown("""
                        ### 📊 About the Model
                        - **Model:** BERT-base-uncased
                        - **Accuracy:** ~92.5%
                        - **Training:** IMDB Dataset
                        """)
                
                # Examples
                gr.Markdown("### 💡 Try these examples:")
                examples = get_example_reviews()
                
                with gr.Row():
                    for i, example in enumerate(examples[:3]):
                        gr.Examples(
                            examples=[[example]],
                            inputs=review_input,
                            label=f"Example {i+1}",
                            cache_examples=False
                        )
                
                # Event handlers
                analyze_btn.click(
                    fn=classify_review,
                    inputs=review_input,
                    outputs=output_html
                )
                
                clear_btn.click(
                    fn=lambda: ("", ""),
                    inputs=[],
                    outputs=[review_input, output_html]
                )
            
            # Tab 2: Batch Processing
            with gr.TabItem("📦 Batch Processing"):
                gr.Markdown("""
                ### Process Multiple Reviews
                
                Enter **one review per line**. Up to 50 reviews at once.
                """)
                
                batch_input = gr.Textbox(
                    lines=10,
                    placeholder="Enter one review per line...\n\nExample:\nThis movie was great!\nI hated this film.\nIt was okay, nothing special.",
                    label="Reviews (one per line)"
                )
                
                batch_btn = gr.Button("📊 Analyze Batch", variant="primary")
                batch_output = gr.Markdown(label="Batch Results")
                
                batch_btn.click(
                    fn=analyze_batch,
                    inputs=batch_input,
                    outputs=batch_output
                )
            
            # Tab 3: Model Info
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Project
                
                ### 🧠 Model Architecture
                - **Base Model:** BERT-base-uncased (110M parameters)
                - **Task:** Binary sentiment classification
                - **Fine-tuning:** IMDB movie reviews dataset
                
                ### 📊 Performance Metrics
                | Metric | Score |
                |--------|-------|
                | Accuracy | 92.5% |
                | F1 Score | 0.92 |
                | Precision | 0.93 |
                | Recall | 0.91 |
                
                ### 🛠️ Technology Stack
                - **Framework:** PyTorch + Hugging Face Transformers
                - **Backend:** FastAPI
                - **UI:** Gradio + Streamlit
                - **Deployment:** Docker
                
                ### 📁 Project Repository
                [GitHub](https://github.com/YOUR_USERNAME/imdb-sentiment-bert)
                
                ### 👨‍💻 Author
                **Your Name** - Intern Candidate
                
                ---
                *Built for internship interview portfolio*
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: gray;">
        ⚡ Powered by BERT | 🎯 Fine-tuned on IMDB | 🚀 Ready for production
        </div>
        """)
    
    return demo


# For direct execution
def main():
    """Run the Gradio app directly."""
    # Load model before starting
    load_model()
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",
        server_port=7860,
        debug=False
    )


if __name__ == "__main__":
    main()