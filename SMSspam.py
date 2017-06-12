# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Import i przygotowanie danych
sms = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
sms = sms[['message', 'label']]
all_sms = []

#petla z kazdego smsa pozbywa sie znakow specjalnych (np. : ; , . # & ^), slow o malym znaczeniu (stop words)
#oraz pozbywa sie koncowek fleksyjnych ze slow i pozostawia temat wyrazu (stemming, algorytm Portera)
for i in range(0, 5572):
    message = re.sub('[^a-zA-Z]', ' ', sms['message'][i] )
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    all_sms.append(message)
    
#stworzenie Bag of Words (vectorisation)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)
X = cv.fit_transform(all_sms).toarray()
y = sms.iloc[:, 1]
                 
#rozdzielenie zbioru na dane treningowe i testowe
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#zastosowanie klasyfikatora Naive Bayes do danych treningowych
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#zastosowanie klasyfikatora Naive Bayes do danych testowych
y_pred = classifier.predict(X_test)

#Macierz pomylek
from sklearn.metrics import confusion_matrix    
cm = confusion_matrix(y_test, y_pred)            

