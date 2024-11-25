# Stock-Market Sentiment Analysis

## Overview
This project aims to analyze financial news articles and predict their sentiment using a pre-trained sentiment analysis model. The sentiment analysis focuses on financial news to help understand market trends and investor sentiment, which can be crucial for decision-making in stock trading and investments.

---

## Features
- **Sentiment Analysis:** Predicts whether financial news articles have a positive, neutral, or negative sentiment.
- **Model Integration:** Utilizes a fine-tuned version of `distilroberta` optimized for financial sentiment analysis.
- **Dataset Support:** Processes financial datasets for model training and evaluation.

---

## Model
We use the **DistilRoBERTa model** fine-tuned for financial sentiment analysis, published on Hugging Face by [mrm8488](https://huggingface.co/mrm8488).  
Model link: [DistilRoBERTa Fine-tuned Financial News Sentiment Analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)

### Why This Model?
This model is specifically fine-tuned to analyze financial texts, making it highly suitable for extracting sentiments from financial news articles, unlike generic sentiment analyzers.

---

## Datasets
The project uses financial news datasets made available on Hugging Face by [NickyNicky](https://huggingface.co/NickyNicky).  
Dataset link: [Financial News Dataset](https://huggingface.co/datasets/NickyNicky/finance-financialmodelingprep-stock-news-sentiments-rss-feed)

### Dataset Highlights:
- Comprises financial news articles tagged with sentiment labels.
- Suitable for both model training and evaluation.
