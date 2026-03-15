# JSC270 Assignment 4 — Natural Language Processing

## Overview
This repository contains the code, report, and dataset for **Assignment 4 (Natural Language Processing)** in **JSC270: Data Science II (Winter 2025)**.

The project explores two main NLP tasks:

1. **Sentiment Analysis of COVID-19 Tweets**  
   Building machine learning models to classify tweets as **negative, neutral, or positive**.

2. **Product Category Classification**  
   Predicting the category of Amazon products using their textual descriptions.

The project demonstrates common NLP preprocessing techniques, text vectorization methods, and several machine learning models for text classification.

---

# Repository Structure


---

# Authors and Contributions

- **Yukun Wang**
  - Prepared presentation slides
  - Model training for Part II

- **Patuan Purba**
  - Model training for Part II
  - Implementation of Part I preprocessing and modeling

- **Jack Li**
  - Authored the written report
  - Implemented one model for Part II

---

# Part I — Sentiment Analysis of COVID-19 Tweets

## Dataset

The dataset contains approximately **45,000 tweets related to COVID-19**.

- **Training set:** 41,155 tweets  
- **Test set:** 3,798 tweets  

Each tweet is labeled with sentiment:

| Label | Sentiment |
|------|-----------|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The objective is to **predict the sentiment using only the tweet text**.

---

## Preprocessing Pipeline

The following preprocessing steps were applied:

1. **Tokenization**
   - Convert tweets into lists of tokens.

2. **URL Removal**
   - Remove tokens starting with `http`.

3. **Text Cleaning**
   - Remove punctuation and special characters
   - Convert all text to lowercase.

4. **Stemming**
   - Porter Stemmer used to normalize word forms.

5. **Stopword Removal**
   - The first 100 stopwords from the NLTK English stopword list were removed.

6. **Vectorization**
   - Count vectors using `CountVectorizer`
   - TF-IDF vectors using `TfidfVectorizer`.

---

## Naive Bayes Model

A **Multinomial Naive Bayes classifier** with **Laplace smoothing** was used.

### Results

| Model | Training Accuracy | Test Accuracy |
|------|------|------|
| Count Vector + NB | 0.7879 | 0.6743 |
| TF-IDF + NB | 0.7223 | 0.6222 |
| TF-IDF + Lemmatization | 0.7224 | 0.6214 |

The **count-vector model performed best** among the Naive Bayes implementations.

---

# Part II — Product Category Classification

## Research Question

**Can we automatically classify a product into its category using only its textual description?**

This problem is challenging because:

- Product descriptions may be ambiguous
- Products can belong to multiple categories
- Text may contain redundant or irrelevant information

---

## Dataset

We used the **Amazon Product Dataset (2020)** from Kaggle.

Dataset characteristics:

- **9,147 observations**
- **31 features**
- Text features combined into a single column called **Textual Information**

The following text fields were merged:

- Product title
- Product specification
- About product description

---

## Exploratory Data Analysis

Key findings:

- The dataset was highly imbalanced.
- **"Toys & Games" accounted for about 66% of observations.**
- To reduce bias, second-level categories were used for those entries.

EDA included:

- Category distribution plots
- Word frequency analysis
- Text preprocessing similar to Part I

---

# Machine Learning Models

Five models were evaluated:

| Model | Description |
|------|------|
| Naive Bayes | Baseline probabilistic classifier |
| Logistic Regression | Linear classifier using softmax |
| Linear SVM | Margin-based classifier |
| Random Forest | Ensemble of decision trees |
| LSTM | Deep learning model for sequential text |

All models were trained using a **60 / 20 / 20 split** for:

- Training
- Validation
- Testing

---

# Model Performance

| Model | Training Accuracy | Validation Accuracy |
|------|------|------|
| Naive Bayes | 0.789 | 0.686 |
| Logistic Regression | 0.831 | 0.721 |
| Linear SVM | 0.984 | 0.746 |
| Random Forest | 0.980 | 0.758 |
| LSTM | 0.914 | 0.841 |

### Key observations

- **SVM and Random Forest showed signs of overfitting.**
- **LSTM performed best**, achieving approximately **84% validation accuracy**.
- Sequential models capture contextual information better than bag-of-words approaches.

---

# Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Scikit-learn
- NLTK
- TensorFlow / Keras
- Matplotlib / Seaborn

---
