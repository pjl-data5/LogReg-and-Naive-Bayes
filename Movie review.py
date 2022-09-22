# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:39:32 2022

@author: paull
"""

# =============================================================================
#  Movie review sentiment classification
#  Logistic regression and Naive Bayes
# =============================================================================
# =============================================================================
# IMDB dataset having 50K movie reviews for
#  natural language processing or Text 
#  analytics. This is a dataset for binary 
#  sentiment classification. Set
#  of 25,000 highly polar movie reviews
#  for training and 25,000 for testing.
#  Predict the number of positive and
#  negative reviews using logistic regression
# =============================================================================
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

review = pd.read_csv("C:/Users/paull/Downloads/Prac_10_NaiveBayes/IMDB.csv")
#review = pd.read_csv('IMDB.csv')
print(review.head())


# Label Encoder
# Encoding the labels to binary values 0 or 1
le = preprocessing.LabelEncoder()

review.sentiment = le.fit_transform(review.sentiment)
# Drop the sentiment column
review.columns=['review','target']

arr = list(review.review)
# Only use the first 2000 rows
X_arr = arr[:2000]
print(review.head())

# =============================================================================
# Use Term Frequency Inverse Document Frequency
# (TFIDF) to get the numerical values for the text
# 
# =============================================================================
tfidf=TfidfVectorizer(max_df=0.5, min_df=2,
                      ngram_range=(1,2), 
                      stop_words='english',
                      token_pattern=r'\b[^\d\W]+\b')

vectors=tfidf.fit_transform(X_arr)
feature_names=tfidf.get_feature_names()
bag_of_words = vectors.todense()
vectorized_text = pd.DataFrame(bag_of_words,columns=feature_names)
#print(feature_names)
#print(bag_of_words)

# Split 1000 reviews training and 1000 reviews testing
X=vectorized_text            #vector
y=review.target[0:2000]      #target
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=42)


# =============================================================================
# Logistic regression
# See relevant repo
# =============================================================================
logreg = LogisticRegression(solver='lbfgs')
logistic_model = logreg.fit(X_train,Y_train)
acc = logistic_model.score(X_train,Y_train)
ratio_class1 = Y_train.mean()
print(acc)
# Positive
print(ratio_class1)

predicted = logistic_model.predict(X_test)
probs = logistic_model.predict_proba(X_test)
acc_score = logistic_model.score(X_test,Y_test)
prob = probs[:,1]
auc_score = metrics.roc_auc_score(Y_test, prob)
print(probs)
print(acc_score)
print(auc_score)
print(metrics.classification_report(Y_test, predicted))

# =============================================================================
# If the model performed random,
# the percentage of positive reviews would be 50%, 
# but since the model is not at random, the model 
# is more accurate
# =============================================================================


# =============================================================================
# Visualisation
# =============================================================================

# =============================================================================
# ROC curve
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='green', label='AUC = 0.92')
    plt.plot([0, 1], [0, 1], color='black', linestyle='-')
    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
fpr, tpr, thresholds = metrics.roc_curve(Y_test, prob)
plot_roc_curve(fpr, tpr)


# =============================================================================
# Naive Bayes Classifier
# =============================================================================
# =============================================================================
# Create a Countvectorizer vectorization (from X_arr)
# epeat the classification 
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

c = CountVectorizer()
counts = c.fit_transform(X_arr)

transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)

X=counts
y=review.target[0:2000]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X,y,test_size=0.5,random_state=42)

model = MultinomialNB().fit(X_train1,Y_train1)

pred = model.predict(X_test1)
acc1 = model.score(X_test1,Y_test1)

probs1 = model.predict_proba(X_test1)
prob1 = probs1[:,1]

print(acc1)
#print(predicted)
#print(acc_score)
#print(auc_score)
print(print(metrics.classification_report(Y_test1, pred)))

def plot_roc_curve(fpr1, tpr1):
    plt.plot(fpr1, tpr1, color='red', label='AUC = 0.89')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.show()
    
fpr1, tpr1, thresholds = metrics.roc_curve(Y_test1, prob1)
plot_roc_curve(fpr1, tpr1)


# =============================================================================
# Interpret
# =============================================================================
# =============================================================================
# Logistic Regression Model should be used for sentiment
# classification since it yielde higher 
# precision and recall scores and 
# a better auc for the ROC curve
# =============================================================================
