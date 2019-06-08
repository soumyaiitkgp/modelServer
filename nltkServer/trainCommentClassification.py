import nltk
import random
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

##############Function to find feature dictionary of each document####################
def findFeatures(document,features):
    featureDict = {}
    for f in features:
        featureDict[f] = 0

    for w in document:
        if w in features:
            featureDict[w] += 1

    return featureDict
######################################################################################

#Adding punctuations to stop words
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)
for w in punctuations:
    stop_words.add(w)

#training data
given = [("plant closed",'plant'),("lunch time",'lunch'),("shift change",'shift'),
("dinner time",'dinner'),("plant close",'plant'),("tea time",'tea'),("dinner time",'dinner'),
("lunch time changed",'lunch'),("plant closed",'plant'),("tea time",'tea'),
("lunch time",'lunch'),("tea time",'tea'),("We faced some issue in tea break.",'tea'),
("plant closed",'plant'),("DISCUSSION ABOUT PARTS PROBLEM AMOUNG TEAM",'meeting'),
("MASKING PROBLEM",'machine'),("PLAN NOT UPDATE FOR 28.05.2019",'others'),
("plant closed",'plant'),("dinner time",'dinner'),("Machine stopped",'machine'),
("plant closed",'plant'),("plant closed",'plant'),("lunch time",'lunch'),
("dinner time",'dinner'),("tea time",'tea'),("plant closed",'plant'),("lunch time",'lunch'),
("plant closed",'plant'),("tea time",'tea'),("Machine not working",'machine'),
("dinner time",'dinner'),("plant closed",'plant'),("dinner time",'dinner'),
("PLAN NOT UPDATE FOR 28.05.2019",'others'),("manpower shifted to another line.",'others'),
("tool problem",'machine'),("plant closed",'plant'),("dinner time",'dinner'),
("plant closed",'plant'),("tea time",'tea'),("plant closed",'plant'),("lunch time",'lunch'),
("tea time",'tea'),("dinner time",'dinner'),("plant closed",'plant'),("lunch time",'lunch'),
("dinner time",'dinner'),("plant closed",'plant'),("lunch time",'lunch'),
("dinner time",'dinner'),("plant closed",'plant'),("shift change",'shift'),
("PLAN NOT UPDATE FOR 28.05.2019",'others')]

train_text = []
vocabulary = []

#tokenizing, removing stop words and lemmatizing to get the training data the way we want
for word,category in given:
    tokenized = word_tokenize(word)
    lemmatized = []
    for w in tokenized:
        if w.lower() not in stop_words:
            root = wnl.lemmatize(w)
            lemmatized.append(root.lower())
            vocabulary.append(root.lower())

    train_text.append((lemmatized,category))
#################################################################

#Shuffling data for good model
random.shuffle(train_text)

#Finding features which are top frequency words
vocab = FreqDist(vocabulary)
vocab = vocab.most_common(8)
features = []
for w,i in vocab:
    features.append(w)

save_feature = open("features.pickle","wb")
pickle.dump(features,save_feature)
save_feature.close()

print("\nFeatures used:\n",features)

#Making training data as required in the nltk NaiveBayesClassifier i.e list of tuples of
#(dictionary of feature count of document i,category of document i)
new_train_text = []
for word,category in train_text:
    featureDict = findFeatures(word,features)
    new_train_text.append((featureDict,category))

#Training data with NB classifier
classifier = nltk.NaiveBayesClassifier.train(new_train_text)
classifier.show_most_informative_features()

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()
