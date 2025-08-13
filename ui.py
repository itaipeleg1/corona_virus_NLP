import streamlit as st
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from config import PROJECT_ROOT
from models.model_config import model_configs

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Analysis",
    page_icon="📱",
    layout="wide"
)

# Cache models to avoid reloading
@st.cache_resource
def load_model(model_config):
    """Load tokenizer and model from the given config"""
    try:
        model_path = model_config["best_path"]
        base_model = model_config["model_name"]
        is_state_dict = model_config.get("is_state_dict", False)
        
        # Load tokenizer from base model
        tokenizer = model_config["tokenizer_class"].from_pretrained(base_model)

        if is_state_dict:
            # Load model architecture from base model, then load state dict
            # Use ignore_mismatched_sizes=True to handle size mismatches
            model = model_config["model_class"].from_pretrained(
                base_model, 
                num_labels=5,  # Set to 5 labels based on your error message
                ignore_mismatched_sizes=True
            )
            state_dict = torch.load(model_path, map_location='cpu')
            # Use strict=False to allow missing keys (for compressed models)
            model.load_state_dict(state_dict, strict=False)
        else:
            # Load model directly from path
            model = model_config["model_class"].from_pretrained(model_path)
        
        model.eval()  # Set to evaluation mode
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment for given text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # Start timing
    start_time = time.time()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # End timing
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = torch.max(predictions).item()
    
    # Map class to sentiment (5 classes based on your labels)
    sentiment_map = {
        0: "Extremely Negative", 
        1: "Negative", 
        2: "Neutral", 
        3: "Positive", 
        4: "Extremely Positive"
    }
    sentiment = sentiment_map.get(predicted_class, "Unknown")
    
    return sentiment, confidence, inference_time, predictions.numpy()[0]

# Main app
def main():
    
    # model_configs = {
    #     "Bertweet": {
    #         "path": PROJECT_ROOT / "results/bertweet_HF_study_augmented/best/best_model_state_dict.pt",
    #         "model_name": "vinai/bertweet-base",
    #         "is_state_dict": True,
    #         "description": "Full fine-tuned BERTweet model"
    #     },
    #     "Compressed BertWeet": {
    #         "path": PROJECT_ROOT / "compression/saved_compressed/bertweet/bertweet_knowledge_distillation_model.pt",
    #         "model_name": "distilroberta-base",
    #         "is_state_dict": True,
    #         "description": "Knowledge distilled BERTweet model"
    #     },
    #     "CovidBert": {
    #         "path": PROJECT_ROOT / "results/covidbert_pytorch_study_augmented/best/best_model_state_dict.pt",
    #         "model_name": "digitalepidemiologylab/covid-twitter-bert",
    #         "is_state_dict": True,
    #         "description": "Full fine-tuned CovidBERT model"
    #     },
    #     "Compressed CovidBert": {
    #         "path": PROJECT_ROOT / "compression/saved_compressed/covidbert/knowledge_distillation_model.pt",
    #         "model_name": "distilbert-base-uncased",
    #         "is_state_dict": True,
    #         "description": "Knowledge distilled CovidBERT model"
    #     }
    # }

    st.title("📱 Tweet Sentiment Analysis")
    st.markdown("### Analyze the sentiment of tweets using fine-tuned BERT models")

        # Pre-load all models at startup
    if 'loaded_models' not in st.session_state:
        st.markdown("## 🚀 Loading Models...")
        st.info("Loading all models for faster inference...")
        
        loaded_models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (model_name, model_config) in enumerate(model_configs.items()):
            status_text.text(f"Loading {model_name}...")
            progress_bar.progress((i + 1) / len(model_configs))
            
            tokenizer, model = load_model(model_config)
            if tokenizer and model:
                loaded_models[model_name] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'config': model_config
                }
                st.success(f"✅ {model_name} loaded successfully")
            else:
                st.error(f"❌ Failed to load {model_name}")
                loaded_models[model_name] = None
        
        # Store in session state
        st.session_state.loaded_models = loaded_models
        
        # Clear loading indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show summary
        successful_models = [name for name, data in loaded_models.items() if data is not None]
        failed_models = [name for name, data in loaded_models.items() if data is None]
        
        if successful_models:
            st.success(f"🎉 Successfully loaded {len(successful_models)} models: {', '.join(successful_models)}")
        if failed_models:
            st.error(f"❌ Failed to load {len(failed_models)} models: {', '.join(failed_models)}")
    
    # Get loaded models from session state
    loaded_models = st.session_state.loaded_models
    available_models = [name for name, data in loaded_models.items() if data is not None]
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(model_configs.keys()),
        help="Select which model to use for sentiment analysis"
    )
    
    # Display model info
    st.sidebar.info(f"**Selected:** {selected_model}\n\n{model_configs[selected_model]['description']}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Tweet Text")
        tweet_text = st.text_area(
            "Tweet content:",
            placeholder="Enter the tweet you want to analyze...",
            height=120,
            help="Paste or type the tweet text here"
        )
        
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    with col2:
        st.subheader("Model Info")
        st.metric("Selected Model", selected_model)
        st.metric("Processing", "CPU Only")

    # Analysis section
    if analyze_button and tweet_text.strip():
        with st.spinner("Loading model and analyzing..."):
            # Load the selected model
            model_config = model_configs[selected_model]
            tokenizer, model = load_model(model_config)
            
            if tokenizer and model:
                # Perform prediction
                try:
                    sentiment, confidence, inference_time, all_probabilities = predict_sentiment(
                        tweet_text, tokenizer, model
                    )
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Results columns
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        if sentiment == "Extremely Positive":
                            st.success(f"**Sentiment:** {sentiment}")
                        elif sentiment == "Positive":
                            st.success(f"**Sentiment:** {sentiment}")
                        elif sentiment == "Neutral":
                            st.info(f"**Sentiment:** {sentiment}")
                        elif sentiment == "Negative":
                            st.warning(f"**Sentiment:** {sentiment}")
                        else:  # Extremely Negative
                            st.error(f"**Sentiment:** {sentiment}")

                    with result_col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    with result_col3:
                        st.metric("Inference Time", f"{inference_time:.2f} ms")
                    
                    # Detailed probabilities - Fixed to handle 5 classes properly
                    st.subheader("Detailed Probabilities")
                    
                    sentiment_labels = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
                    colors = ["#00cc44", "#90ee90", "#ffa500", "#ff8c69", "#ff4b4b"]
                    
                    # Create columns dynamically based on number of classes
                    num_classes = len(all_probabilities)
                    if num_classes == 5:
                        cols = st.columns(5)
                        for i, (label, prob, color) in enumerate(zip(sentiment_labels[:num_classes], all_probabilities, colors[:num_classes])):
                            with cols[i]:
                                st.metric(
                                    label=label,
                                    value=f"{prob:.3f}",
                                    help=f"Probability of {label.lower()} sentiment"
                                )
                    else:
                        # Handle different number of classes
                        st.write("**Class Probabilities:**")
                        for i, prob in enumerate(all_probabilities):
                            st.write(f"Class {i}: {prob:.3f}")
                    
                    # Progress bars for visual representation
                    st.subheader("Probability Distribution")
                    labels_to_use = sentiment_labels[:len(all_probabilities)]
                    colors_to_use = colors[:len(all_probabilities)]
                    
                    for label, prob, color in zip(labels_to_use, all_probabilities, colors_to_use):
                        st.write(f"**{label}:**")
                        st.progress(float(prob))
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    # Show more detailed error info for debugging
                    st.write("**Debug info:**")
                    st.write(f"Model output shape: {getattr(model, 'config', {}).get('num_labels', 'unknown')}")
            else:
                st.error("Failed to load the selected model. Please check the model path.")
    
    elif analyze_button and not tweet_text.strip():
        st.warning("Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Fine-tuned BERT Models • CPU Inference</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()