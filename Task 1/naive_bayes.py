import re
import pickle
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


#Load data
category = {'tech': 5, 'business': 1, 'sport': 4, 'entertainment':2 , 'politics': 3}
invert_cat = {5: 'tech', 1: 'business', 4: 'sport', 1: 'entertainment', 3: 'politics'}
def load_data(path_name = "bbc-text.csv"):
    BBC_data = pd.read_csv(path_name)
    X_data = list(BBC_data['text'].values)
    y_data = list(BBC_data['category'].values)
    for i in range(len(y_data)):
        y_data[i] = category[y_data[i]]
    BBC_data_Hy = pd.read_csv("data_crawled.csv")
    X_data_Hy = list(BBC_data_Hy['content'].values)
    X_data.extend(X_data_Hy)
    y_data_Hy = list(BBC_data_Hy['category'].values)
    y_data.extend(y_data_Hy)
    return X_data, y_data

# X_data, y_data = load_data()
# print(X_data[:2])

#preprocessing:
#remove stop words, numbers, punctuation, whitespace, lemmatization
#lowcase

def Punctuation(string): 
  
    # punctuation marks 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string.lower(): 
        if x in punctuations:
            string = string.replace(x, "") 

    # Print string without punctuation 
    return string

def preprocess(X_data):
    dic = set()
    sentences = []
    stops = set(stopwords.words("english"))
    i = 0
    for sentence in X_data:
        i += 1

        #Chi giu cac ki tu alphabet
        # sentence = sentence.lower()
        # letter_only = re.sub("[^a-zA-Z]", " ", sentence)
        letter_only = sentence.split()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        new_sentence = ""
        #Loai bo stop words va bien the cua tu
        for word in letter_only:
            if word not in stops:
                word = Punctuation(word)
                # word = stemmer.stem(word)
                # word = lemmatizer.lemmatize(word)
                new_sentence = new_sentence + word + " "
        sentences.append(new_sentence)
    return sentences

def predict_new_data(model, x, vectorlize):
    x = preprocess(x)
    x = vectorlize.transform(x)
    return invert_cat[model.predict(x)[0]]

if __name__ == "__main__":
    X_data, y_data = load_data()
    X_data = preprocess(X_data)

    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(X_data)
    X_data_binary = count_vect.transform(X_data)

    tfidf_vect = TfidfVectorizer(analyzer = "word", ngram_range=(1,2))
    tfidf_vect.fit(X_data)
    X_data_tfidf = tfidf_vect.transform(X_data)

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_data_tfidf, y_data, test_size = 0.3, random_state= 22, shuffle=True)
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_data_binary, y_data, test_size = 0.3, random_state= 22, shuffle=True)

    clf_tfidf = MultinomialNB(alpha = 0.08)
    clf_binary = MultinomialNB(alpha = 0.08)
    clf_tfidf.fit(X_train_tfidf, y_train_tfidf)
    clf_binary.fit(X_train_binary, y_train_binary)
    predict_vals_tfidf = clf_tfidf.predict(X_test_tfidf)
    predict_vals_binary = clf_binary.predict(X_test_binary)
    print('===================================')
    print("results validation for tfidf vectorlize")
    print(classification_report(y_test_tfidf, predict_vals_tfidf))
    print('===================================')

    print('===================================')
    print("results validation for binary vectorlize")
    print(classification_report(y_test_binary, predict_vals_binary))
    print('===================================')


