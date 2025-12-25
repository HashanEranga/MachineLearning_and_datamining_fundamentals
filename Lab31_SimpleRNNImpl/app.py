import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-size: 2rem;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-size: 2rem;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load word index
@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reversed_word_index = {value: key for key, value in word_index.items()}
    return word_index, reversed_word_index

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper functions
def preprocess_text(text, word_index, max_len=500):
    """Preprocess text for prediction"""
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

def predict_sentiment(review, model, word_index):
    """Predict sentiment of a review"""
    preprocessed_input = preprocess_text(review, word_index)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return sentiment, prediction[0][0], confidence

def create_gauge_chart(score):
    """Create a gauge chart for sentiment score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Simple RNN Deep Learning Model</div>', unsafe_allow_html=True)

    # Load resources
    word_index, reversed_word_index = load_word_index()
    model = load_trained_model()

    if model is None:
        st.error("Failed to load model. Please ensure 'model.keras' exists in the current directory.")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Sentiment", "Model Info", "Batch Analysis", "Sample Reviews"])

    # Predict Sentiment Page
    if page == "Predict Sentiment":
        st.header("Analyze Your Movie Review")

        # Text input
        review_text = st.text_area(
            "Enter your movie review:",
            height=150,
            placeholder="Type your movie review here... e.g., 'This movie was absolutely fantastic! The acting was superb and the plot was engaging.'"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

        if predict_button and review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, score, confidence = predict_sentiment(review_text, model, word_index)

                # Display results
                st.markdown("---")
                st.subheader("Analysis Results")

                # Create two columns for results
                col1, col2 = st.columns(2)

                with col1:
                    if sentiment == "Positive":
                        st.markdown(f'<div class="sentiment-positive">Sentiment: {sentiment}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentiment-negative">Sentiment: {sentiment}</div>', unsafe_allow_html=True)

                    st.markdown(f'<div class="confidence-score">Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                    st.markdown(f'**Raw Score:** {score:.4f}')

                    # Interpretation
                    st.markdown("### Interpretation")
                    if score > 0.8:
                        st.success("Strongly Positive Review!")
                    elif score > 0.6:
                        st.success("Moderately Positive Review")
                    elif score > 0.4:
                        st.warning("Moderately Negative Review")
                    else:
                        st.error("Strongly Negative Review!")

                with col2:
                    # Gauge chart
                    fig = create_gauge_chart(score)
                    st.plotly_chart(fig, use_container_width=True)

        elif predict_button:
            st.warning("Please enter a review to analyze.")

    # Model Info Page
    elif page == "Model Info":
        st.header("Model Architecture & Information")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Summary")

            # Get model summary
            st.text("Model: Sequential")
            st.text("━" * 50)
            st.text("Layer (type)              Output Shape         Param #")
            st.text("━" * 50)
            st.text("embedding (Embedding)     (32, 500, 128)       1,280,000")
            st.text("simple_rnn (SimpleRNN)    (32, 128)            32,896")
            st.text("dense (Dense)             (32, 1)              129")
            st.text("━" * 50)
            st.text("Total params: 1,313,025 (5.01 MB)")
            st.text("Trainable params: 1,313,025 (5.01 MB)")

        with col2:
            st.subheader("Model Configuration")
            config_data = {
                "Parameter": [
                    "Vocabulary Size",
                    "Embedding Dimension",
                    "Max Sequence Length",
                    "RNN Units",
                    "Activation (RNN)",
                    "Output Activation",
                    "Loss Function",
                    "Optimizer",
                    "Dataset"
                ],
                "Value": [
                    "10,000 words",
                    "128",
                    "500 tokens",
                    "128",
                    "ReLU",
                    "Sigmoid",
                    "Binary Crossentropy",
                    "Adam",
                    "IMDB Reviews"
                ]
            }
            st.dataframe(pd.DataFrame(config_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("How It Works")

        st.markdown("""
        1. **Embedding Layer**: Converts word indices into dense vectors of fixed size (128 dimensions)
        2. **SimpleRNN Layer**: Processes the sequence of word embeddings to capture temporal dependencies
        3. **Dense Layer**: Output layer with sigmoid activation for binary classification (positive/negative)
        4. **Training**: Model was trained on 25,000 IMDB movie reviews with early stopping
        5. **Performance**: Achieved approximately 84% validation accuracy
        """)

    # Batch Analysis Page
    elif page == "Batch Analysis":
        st.header("Batch Review Analysis")
        st.write("Analyze multiple reviews at once")

        # Text area for multiple reviews
        batch_reviews = st.text_area(
            "Enter multiple reviews (one per line):",
            height=200,
            placeholder="Review 1: This movie was great!\nReview 2: Terrible film, waste of time.\nReview 3: Amazing cinematography and acting."
        )

        if st.button("Analyze All Reviews", type="primary"):
            if batch_reviews.strip():
                reviews = [r.strip() for r in batch_reviews.split('\n') if r.strip()]

                if reviews:
                    with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                        results = []

                        for idx, review in enumerate(reviews, 1):
                            sentiment, score, confidence = predict_sentiment(review, model, word_index)
                            results.append({
                                "Review #": idx,
                                "Review Text": review[:100] + "..." if len(review) > 100 else review,
                                "Sentiment": sentiment,
                                "Score": f"{score:.4f}",
                                "Confidence": f"{confidence*100:.2f}%"
                            })

                        # Display results
                        st.subheader("Analysis Results")
                        df = pd.DataFrame(results)

                        # Style the dataframe
                        def highlight_sentiment(row):
                            if row['Sentiment'] == 'Positive':
                                return ['background-color: #d4edda'] * len(row)
                            else:
                                return ['background-color: #f8d7da'] * len(row)

                        styled_df = df.style.apply(highlight_sentiment, axis=1)
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                        # Summary statistics
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Reviews", len(results))
                        with col2:
                            positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                            st.metric("Positive Reviews", positive_count)
                        with col3:
                            negative_count = len(results) - positive_count
                            st.metric("Negative Reviews", negative_count)
            else:
                st.warning("Please enter at least one review.")

    # Sample Reviews Page
    else:
        st.header("Sample Reviews for Testing")
        st.write("Try these sample reviews to see how the model performs")

        sample_reviews = {
            "Positive Reviews": [
                "This movie was absolutely fantastic! The acting was superb and the plot was engaging throughout.",
                "An incredible masterpiece that will be remembered for years. Brilliant performances from the entire cast.",
                "I loved every minute of this film. The cinematography was stunning and the soundtrack was perfect.",
                "Highly recommended! One of the best movies I have seen in years.",
                "Amazing storytelling with unexpected twists. The director did an outstanding job."
            ],
            "Negative Reviews": [
                "This was a complete waste of time. The plot was terrible and the acting was even worse.",
                "I fell asleep halfway through this boring disaster. Nothing happened for two hours.",
                "The worst film I have ever seen in my entire life. Avoid at all costs.",
                "Completely disappointed with this overhyped mess. The story made no sense.",
                "Terrible movie with poor acting and a ridiculous plot. I want my money back."
            ],
            "Mixed Reviews": [
                "The movie had some good moments but overall it was just okay.",
                "Great visuals but the story was lacking depth and character development.",
                "Interesting concept but poor execution. Could have been much better.",
                "The first half was entertaining but the second half was disappointing."
            ]
        }

        for category, reviews in sample_reviews.items():
            st.subheader(category)

            for idx, review in enumerate(reviews, 1):
                with st.expander(f"Sample {idx}: {review[:50]}..."):
                    st.write(f"**Full Review:** {review}")

                    if st.button(f"Analyze This Review", key=f"{category}_{idx}"):
                        sentiment, score, confidence = predict_sentiment(review, model, word_index)

                        col1, col2 = st.columns(2)
                        with col1:
                            if sentiment == "Positive":
                                st.success(f"**Sentiment:** {sentiment}")
                            else:
                                st.error(f"**Sentiment:** {sentiment}")
                            st.info(f"**Confidence:** {confidence*100:.2f}%")
                            st.info(f"**Raw Score:** {score:.4f}")

                        with col2:
                            fig = create_gauge_chart(score)
                            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit and TensorFlow | Simple RNN Model for Sentiment Analysis</p>
        <p>Model trained on IMDB Movie Reviews Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
