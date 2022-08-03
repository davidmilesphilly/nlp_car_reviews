import itertools
from typing import List, Any

import numpy as np
import csv
import random
import pandas as pd
import numpy as np
import string
import re
class Sentiment:
    Negative = "Negative"
    # Neutral = "Neutral"
    Positive = "Positive"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score < 3:
            return Sentiment.Negative
        # elif self.score == 3:
        #     return Sentiment.Neutral
        else:  # rating of 4 or 5
            return Sentiment.Positive

class Rev_Container:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.Negative, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.Positive, self.reviews))
        # neutral =  list(filter(lambda x: x.sentiment == Sentiment.Neutral, self.reviews))
        pos_shrunk = positive[:len(negative)]
        self.reviews = negative + pos_shrunk
        random.shuffle(self.reviews)

reviews = []
with open(r"C:\Users\Dave\Desktop\atom_projects\Scraped_Car_Review_dodge.csv", encoding="utf8") as f:
    review = csv.DictReader(f, delimiter=",")
    for line in review:
        # if len(line['Review']) > 2: # removes entries with no written review
        if line["Rating"]  != None:
            reviews.append(Review(line["Review"], float(line["Rating"])))

###### ML ########
from sklearn.model_selection import train_test_split, GridSearchCV
training, test= train_test_split(reviews, test_size=.33, random_state=42)

train_container = Rev_Container(training)
test_container = Rev_Container(test)
train_container.evenly_distribute()
test_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

####### Remove punctuation #########
new_train_x = []
for contents in train_x:
    words = ' '
    for letters in contents:
        if letters not in string.punctuation:
            words += letters
    new_train_x.append(words)
# n
#### remove digits #####
# print(new_train_x)

def digit_remover(big_list):
    no_digit = []
    lets =[]
    for words in big_list:
        for i in words:
            if not i.isdigit():
                lets.append(i)
                letus = "".join(lets)
        no_digit.append(letus)
        lets = []

    return no_digit

x = digit_remover(new_train_x)

######### Stem sentences/lemma sentences ########


import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
porter=PorterStemmer()

def stemSentence(file):
    stemmed_words =[]
    for lines in file:
            stemmedwords = porter.stem(lines)
            stemmed_words.append(stemmedwords)
    return (stemmed_words)

new_train_x =stemSentence(x)
# print(x)
# print(len(x))

from nltk.tokenize import sent_tokenize, word_tokenize


# ######## stop word attempts########
stop_words = stopwords.words("english")
def stops_out(page):
    new_train_x2 = []
    for words in page:
        word = word_tokenize(words)
        if word not in stop_words:
            new_train_x2.append(words)
    return new_train_x2

training_x = stops_out(new_train_x)


###### Bag of words ######
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
vec = TfidfVectorizer(use_idf=True, max_df=.7)
# vec = CountVectorizer(ngram_range=(1,3))
train_x_vectors = vec.fit_transform(training_x)
test_x_vectors = vec.transform(test_x)
#
# # ##### Linear svm #########
from sklearn import svm
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

# #### Linear Svm #######
svm_score = clf_svm.score(test_x_vectors, test_y)
print("The Svm Score is", svm_score)
#
# # ###### f1 scores #####
from sklearn.metrics import f1_score

svm_f1_score =f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels= [Sentiment.Positive, Sentiment.Negative])
print("The F1 scores for svm is", svm_f1_score)

