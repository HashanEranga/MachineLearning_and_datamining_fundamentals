import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    model = load_model('model.keras')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

def predict_the_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word based on input text"""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

def main():
    st.title("Next Word Prediction using LSTM")
    st.write("This app predicts the next word based on Shakespeare's Hamlet text")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
        max_sequence_len = 14  # As defined in the training notebook

        st.success("Model and tokenizer loaded successfully!")

        # Input section
        st.subheader("Enter your text")
        input_text = st.text_input(
            "Type a phrase from Hamlet:",
            value="Bar. Say, what is",
            help="Enter a phrase and the model will predict the next word"
        )

        # Prediction button
        if st.button("Predict Next Word"):
            if input_text:
                with st.spinner("Predicting..."):
                    predicted_word = predict_the_next_word(
                        model,
                        tokenizer,
                        input_text.lower(),
                        max_sequence_len
                    )

                if predicted_word:
                    st.success(f"**Input:** {input_text}")
                    st.success(f"**Predicted next word:** {predicted_word}")
                    st.info(f"**Complete phrase:** {input_text} {predicted_word}")
                else:
                    st.error("Could not predict the next word. Try a different phrase.")
            else:
                st.warning("Please enter some text first!")

        # Model information
        st.sidebar.header("Model Information")
        st.sidebar.write("**Architecture:** LSTM")
        st.sidebar.write("**Training Data:** Shakespeare's Hamlet")
        st.sidebar.write("**Max Sequence Length:** 14")
        st.sidebar.write("**Total Words:** 4818")

        st.sidebar.header("Model Layers")
        st.sidebar.write("1. Embedding Layer (100 dimensions)")
        st.sidebar.write("2. LSTM Layer (150 units)")
        st.sidebar.write("3. Dropout (0.2)")
        st.sidebar.write("4. LSTM Layer (100 units)")
        st.sidebar.write("5. Dense Layer (softmax)")

        # Example inputs
        st.sidebar.header("Example Inputs")
        examples = [
            "to be or not to",
            "what is your",
            "the king is",
            "good night sweet"
        ]
        for example in examples:
            st.sidebar.write(f"- {example}")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'model.keras' and 'tokenizer.pkl' are in the same directory as this app.")

if __name__ == "__main__":
    main()
