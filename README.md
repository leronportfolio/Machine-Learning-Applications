# Machine-Learning-Applications

***NOTE**** Notebook 9 is consist of creating my own notebook based on the dataset provided and apply machine learning techniques. *****

# Sentiment Analysis Machine Learning Workflow

This README file provides an overview of a sentiment analysis machine learning workflow implemented in a Jupyter Notebook. The goal of this workflow is to classify movie reviews as either positive or negative based on their sentiment. The workflow includes data preprocessing, feature extraction, model training, hyperparameter experimentation, evaluation, and prediction on new samples of text.

## Introduction

In this notebook, we will demonstrate a sentiment analysis machine learning workflow using a linear classifier from the scikit-learn library. The dataset used for this task is the Sentiment Polarity Data Set v2.0 from Movie Review Data by Pang, Lee, and Vaithyanathan. The dataset contains a complete training set and a test set, which will be utilized in this notebook.

## Data Preprocessing

Before diving into the sentiment analysis ML workflow, the data is explored and preprocessed to gain insights and prepare it for further analysis. This step includes printing column names, data overview, analyzing the distribution of sentiment labels, checking for missing values, and necessary data cleaning or preprocessing.

The data preprocessing steps performed in this notebook include:

- Inspecting the data by printing column names and data overview.
- Analyzing the distribution of sentiment labels to check for class imbalance.
- Checking for missing values in the dataset.
- Applying data cleaning and preprocessing functions to remove noise or irrelevant information from the text.

## Feature Extraction

To train a machine learning model on text data, the textual content needs to be converted into numerical features that the model can understand. In this step, relevant features are extracted from the preprocessed movie reviews.

In this notebook, a bag-of-words approach is used to represent the movie reviews as numerical feature vectors. The TF-IDF algorithm is employed to convert the text data into TF-IDF feature vectors. The TfidfVectorizer from the scikit-learn library is used for this purpose. The training and test data are transformed into TF-IDF feature vectors.

## Model Training

Once the text data has been transformed into numerical features, a linear classifier model is trained using the scikit-learn library. The training data is split into training and validation sets. The model is trained on the training set and evaluated on the validation set to assess its performance.

In this notebook, a logistic regression model is utilized as the linear classifier. The training data is split into training and validation sets using the train_test_split function from scikit-learn. The logistic regression model is then trained on the training set and used to make predictions on the validation set. The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Hyperparameter Experimentation

To optimize the performance of the model, hyperparameter experimentation is conducted to find the best combination of hyperparameters that yield the highest performance. In this notebook, grid search is used as the hyperparameter search technique.

A parameter grid is defined, including hyperparameters such as C, penalty, and solver for the logistic regression model. The TfidfVectorizer is used to transform the text data into TF-IDF vectors. The logistic regression model is created, and GridSearchCV from scikit-learn is employed to perform grid search with 5-fold cross-validation. The best hyperparameters and the corresponding score are printed.

## Evaluation

After determining the optimal hyperparameters, the final model is trained using the entire training dataset. The model's performance is evaluated on the test dataset, including metrics such as accuracy, precision, recall, and F1-score.

In this notebook, the logistic regression model is trained on the entire training dataset. Predictions are made on the test dataset, and a classification report is generated to evaluate the model's performance.

## Prediction

Using the trained model, predictions are made on new samples of text to showcase the model's
