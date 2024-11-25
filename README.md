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

---

## Fine-tune the model
The model that can be fine-tuned can be seen here: [msr2903/mrm8488-distilroberta-fine-tuned-financial-sentiment](https://huggingface.co/msr2903/mrm8488-distilroberta-fine-tuned-financial-sentiment)

This model is a fine-tuned version of [mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) on the [NickyNicky/finance-financialmodelingprep-stock-news-sentiments-rss-feed](https://huggingface.co/datasets/NickyNicky/finance-financialmodelingprep-stock-news-sentiments-rss-feed) dataset. It achieves the following results on the evaluation set:

- Loss: 0.4090
- Accuracy: 0.9171

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- num_epochs: 5

### Training results

| Training Loss | Epoch | Validation Loss |
|:-------------:|:-----:|:---------------:|
| 0.318500      | 1.0   | 0.294045        |
| 0.281700      | 2.0   | 0.298364        |
| 0.250100	    | 3.0   | 0.302255        |
| 0.186400      | 4.0   | 0.380530        |
| 0.179100      | 5.0   | 0.409072        |
