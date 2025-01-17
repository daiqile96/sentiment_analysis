# Sentiment Analysis for Marketing via Bidirectional Transformer Model

This project focuses on sentiment analysis for Amazon product reviews, aiming to classify reviews into two categories: positive and negative sentiment. We leverage the BERT (Bidirectional Encoder Representations from Transformers) model, a state-of-the-art pre-trained transformer, to perform sentiment classification.

See full scripts and result of the analyses [here](https://nbviewer.org/github/daiqile96/sentiment_analysis/blob/main/amazon_sentiment.ipynb).

**Last updated:** Jan 3, 2025

## Data
The dataset contains Amazon product reviews labeled as either positive (1) or negative (0) sentiment. It was sourced from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). The original dataset is large, with the following label counts:

- **Original Train Dataset:** 1,800,000 positive reviews and 1,800,000 negative reviews (3.6 million total samples)
- **Original Test Dataset:** 200,000 positive reviews and 200,000 negative reviews (400,000 total samples)

- Example Reviews:

    - Positive: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^

    - Negative: Batteries died within a year ...: I bought this charger in Jul 2003 and it worked OK for a while. The design is nice and convenient. However, after about a year, the batteries would not hold a charge. Might as well just get alkaline disposables, or look elsewhere for a charger that comes with batteries that have better staying power.


Due to limited computational resources, we randomly sampled a subset of the data while preserving the original label distribution:
- **Sampled Train Dataset:** 1800 samples (balanced: 934 positive, 866 negative)
- **Sampled Test Dataset:** 200 samples (balanced: 101 positive, 99 negative)



## Method
The analysis consists of the following steps:
- **Text Data Processing:** The text data was preprocessed to prepare it for the BERT model. This involved steps such as tokenization and padding to convert the text into token IDs.
- **Exploratory Data Analysis (EDA):** The label distribution was checked to ensure balance, and a subset of the data was used to make the project computationally efficient.
- **Model Configuration:** A pre-trained BERT model was fine-tuned for sentiment analysis using the training data.
- **Evaluation:** The model was evaluated on the test data for performance metrics, including accuracy and loss.



## Results
- **Model Performance:** The fine-tuned BERT model achieved high accuracy on the test set. The following table summarizes the precision, recall, F1-score, and support for each class, as well as the overall accuracy:

| Metric          | Class 0 (Negative) | Class 1 (Positive) | Macro Avg | Weighted Avg | Overall Accuracy |
|------------------|--------------------|--------------------|-----------|--------------|------------------|
| **Precision**    | 0.83              | 0.81              | 0.82      | 0.82         | 0.82             |
| **Recall**       | 0.80              | 0.84              | 0.82      | 0.82         | -                |
| **F1-Score**     | 0.81              | 0.83              | 0.82      | 0.82         | -                |
| **Support**      | 99                | 101               | -         | -            | 200              |


## Future Direction
The project can be extended to:
- Analyze a larger dataset for improved generalization.
- Explore multi-class sentiment classification (e.g., adding neutral sentiment).
- Apply other transformer models for comparative analysis.
