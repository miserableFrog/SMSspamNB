# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:46:07 2017

@author: olejk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import danych
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#przygotowanie tekstu - pozbycie się znakow 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []


#petla dla kazdej recenzji pozbywa sie znakow specjalnych, nieuzytecznych slow (stopwords)
#oraz przeprowadza konwersje slow na podstawowe formy (Stemming)
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] )
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Tworzenie Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1560)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
                
#rozdzielenie zbioru na dane treningowe i testowe
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

#zastosowanie Naive Bayes do danych treningowych
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predykcja zestawu testowego
y_pred = classifier.predict(X_test)

#Macierz 
from sklearn.metrics import confusion_matrix    
cm = confusion_matrix(y_test, y_pred)            


