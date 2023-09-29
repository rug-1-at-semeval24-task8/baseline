from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import json
import pandas
import sklearn

labels_a = {
    0: "human",
    1: "machine"
}

labels_b = {
    0: "human",
    1: "ChatGPT",
    2: "Cohere",
    3: "Davinci",
    4: "Bloomz",
    5: "Dolly"
} 

train_data_A = pandas.read_json("data/SubtaskA/subtaskA_train_monolingual.jsonl", lines=True)
dev_data_A = pandas.read_json("data/SubtaskA/subtaskA_dev_monolingual.jsonl", lines=True)

X_train = train_data_A["text"]
y_train = train_data_A["label"]
X_dev = dev_data_A["text"]
y_dev = dev_data_A["label"]

clf = make_pipeline(TfidfVectorizer(ngram_range=(1,3), min_df=0.02, max_df=0.5), RandomForestClassifier())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_dev)

print(classification_report(y_dev, y_pred))
print(confusion_matrix(y_dev, y_pred))