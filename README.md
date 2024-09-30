# SMS Spam Classifier

## Overview
This project implements an SMS Spam Classifier using machine learning algorithms. The classifier is designed to predict whether a given SMS message is spam or ham (not spam). It provides an end-to-end pipeline including data preprocessing, model training, and performance evaluation. SMS Spam Classification is a common use case of natural language processing (NLP) and machine learning, where we aim to detect whether a message is a spam message or a legitimate one. This notebook walks through the steps of building such a classifier from data preparation to model evaluation.

## Features:
Preprocesses the text data (SMS messages) using techniques like tokenization and TF-IDF transformation.
Builds several machine learning models including Logistic Regression, Naive Bayes, and Support Vector Machines to classify SMS messages.
Evaluates models using accuracy, precision, recall, and F1-score.
Provides visual insights into model performance using confusion matrices and other plots.

## Technologies Used:
Python: Main programming language
scikit-learn: For machine learning algorithms and evaluation
Pandas: For data manipulation and analysis
Numpy: For numerical operations
Matplotlib & Seaborn: For data visualization
NLTK: For text preprocessing and tokenization
Dataset
The dataset used for this classifier contains labeled SMS messages. Each message is marked as either "spam" or "ham." The dataset can be downloaded from popular machine learning repositories such as Kaggle.

## Dataset summary:
Total Samples: 5,572 SMS messages
Classes: Spam, Ham

## Usage
Open the SMS Spam Classifier.ipynb notebook.
Follow the steps to preprocess the data, train the classifier models, and evaluate performance.

### To execute all code cells in the notebook:

Use any Jupyter-compatible environment such as Jupyter Notebook, JupyterLab, or Google Colab.

## Model Evaluation
Several machine learning models have been evaluated, and results have been compared based on:

Accuracy
Precision
Recall
F1 Score

You can visualize the confusion matrix and other evaluation metrics for a detailed performance comparison among models.
