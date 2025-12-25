# IMDB Movie Review Sentiment Analysis

A Streamlit web application for analyzing movie review sentiment using a Simple RNN (Recurrent Neural Network) deep learning model.

## Project Overview

This application uses a trained SimpleRNN model to classify movie reviews as either positive or negative. The model was trained on the IMDB movie reviews dataset containing 25,000 reviews and achieved approximately 84% validation accuracy.

## Features

### 1. Predict Sentiment
- Enter your own movie review
- Get instant sentiment analysis (Positive/Negative)
- View confidence scores and visualization
- Interactive gauge chart showing sentiment score

### 2. Model Info
- View complete model architecture
- Model configuration details
- Understanding of how the RNN model works
- Training parameters and dataset information

### 3. Batch Analysis
- Analyze multiple reviews at once
- Enter reviews line by line
- Get comprehensive results table
- Summary statistics (total, positive, negative counts)

### 4. Sample Reviews
- Pre-loaded sample reviews for testing
- Categories: Positive, Negative, and Mixed reviews
- Quick analysis with one-click buttons

## Model Architecture

```
Layer (type)              Output Shape         Param #
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
embedding (Embedding)     (32, 500, 128)       1,280,000
simple_rnn (SimpleRNN)    (32, 128)            32,896
dense (Dense)             (32, 1)              129
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total params: 1,313,025 (5.01 MB)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit tensorflow numpy pandas plotly
```

### Step 2: Ensure Model File Exists

Make sure the `model.keras` file is present in the `Lab31_SimpleRNNImpl` directory.

## Running the Application

### Method 1: From the Lab31_SimpleRNNImpl directory

```bash
cd Lab31_SimpleRNNImpl
streamlit run app.py
```

### Method 2: From the project root

```bash
streamlit run Lab31_SimpleRNNImpl/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Analyzing a Single Review

1. Navigate to the "Predict Sentiment" page
2. Enter your movie review in the text area
3. Click "Analyze Sentiment"
4. View the results including:
   - Sentiment classification (Positive/Negative)
   - Confidence score
   - Raw prediction score
   - Visual gauge chart

### Batch Analysis

1. Go to "Batch Analysis" page
2. Enter multiple reviews, one per line
3. Click "Analyze All Reviews"
4. View results in a formatted table with sentiment highlighting
5. Check summary statistics

### Testing with Samples

1. Visit the "Sample Reviews" page
2. Browse through pre-loaded positive, negative, and mixed reviews
3. Click "Analyze This Review" on any sample
4. See instant results

## Technical Details

### Model Configuration
- **Vocabulary Size**: 10,000 words
- **Embedding Dimension**: 128
- **Max Sequence Length**: 500 tokens
- **RNN Units**: 128
- **Activation Function**: ReLU (RNN), Sigmoid (Output)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Dataset**: IMDB Movie Reviews (25,000 training samples)

### How It Works

1. **Text Preprocessing**: Input text is converted to lowercase, tokenized, and encoded using the IMDB word index
2. **Padding**: Sequences are padded to a fixed length of 500 tokens
3. **Embedding**: Words are converted to dense 128-dimensional vectors
4. **RNN Processing**: SimpleRNN layer processes the sequence to capture temporal dependencies
5. **Classification**: Dense layer with sigmoid activation outputs a probability score (0-1)
6. **Interpretation**: Score > 0.5 = Positive, Score ≤ 0.5 = Negative

## File Structure

```
Lab31_SimpleRNNImpl/
├── app.py                    # Main Streamlit application
├── model.keras               # Trained RNN model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── simpleRnn.ipynb          # Basic RNN tutorial
├── DLUsingRNN.ipynb         # Model training notebook
└── predictionRNN.ipynb      # Prediction examples
```

## Troubleshooting

### Model Not Found Error
- Ensure `model.keras` is in the same directory as `app.py`
- Check file permissions

### Import Errors
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using Python 3.8+

### Performance Issues
- The first prediction might be slow as TensorFlow initializes
- Subsequent predictions will be faster due to caching

## Future Enhancements

- Add support for LSTM and GRU models
- Include model training interface
- Add sentiment score distribution charts
- Export analysis results to CSV
- Support for custom model uploads
- Real-time sentiment analysis from text files

## Credits

- **Dataset**: IMDB Movie Reviews Dataset (via Keras)
- **Framework**: TensorFlow/Keras for deep learning
- **UI**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts

## License

This project is for educational purposes as part of the Machine Learning and Data Mining Fundamentals course.
