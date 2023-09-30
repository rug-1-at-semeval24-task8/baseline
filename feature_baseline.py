from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import pandas
import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="Which task to evaluate (a_mono, a_multi, b, c)")
    parser.add_argument("-m", "--model", type=str, help="Which model to use (NB, RF, SVC, DT, KNN)")
    parser.add_argument("-a", "--all", action="store_true", help="Run all models on all tasks")
    parser.add_argument("-f", "--fraction", type=float, default=0.05, help="Fraction of the data to use (default 0.05)")
    return parser.parse_args()

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

def run_pipeline(task, model, fraction):
    print(f"--- Now running task {task} with model {model} on {int(fraction*100)}% of the data---")
    if task == "a_mono":
        train_data = pandas.read_json("data/SubtaskA/subtaskA_train_monolingual.jsonl", lines=True)
        dev_data = pandas.read_json("data/SubtaskA/subtaskA_dev_monolingual.jsonl", lines=True)
    elif task == "a_multi":
        train_data = pandas.read_json("data/SubtaskA/subtaskA_train_multilingual.jsonl", lines=True)
        dev_data = pandas.read_json("data/SubtaskA/subtaskA_dev_multilingual.jsonl", lines=True)
    elif task == "b":
        train_data = pandas.read_json("data/SubtaskB/subtaskB_train.jsonl", lines=True)
        dev_data = pandas.read_json("data/SubtaskB/subtaskB_dev.jsonl", lines=True)
    elif task == "c":
        train_data = pandas.read_json("data/SubtaskC/subtaskC_train.jsonl", lines=True)
        dev_data = pandas.read_json("data/SubtaskC/subtaskC_dev.jsonl", lines=True)

    train_data = train_data.sample(frac=fraction)
    dev_data = dev_data.sample(frac=fraction)

    X_train = train_data["text"]
    y_train = train_data["label"]
    X_dev = dev_data["text"]
    y_dev = dev_data["label"]        

    if model == "NB":
        clf = MultinomialNB()
    elif model == "RF":
        clf = RandomForestClassifier()
    elif model == "SVM":
        clf = SVC(kernel="sigmoid")
    elif model == "DT":
        clf = DecisionTreeClassifier()
    elif model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=3)

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=0.02, max_df=0.8), clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_dev)

    print(classification_report(y_dev, y_pred))
    print(confusion_matrix(y_dev, y_pred))

def main():
    args = create_arg_parser()
    if args.all:
        for task in ["a_mono", "a_multi", "b"]:
            for model in ["NB", "RF", "SVM", "DT", "KNN"]:
                run_pipeline(task, model, 1.0)
    else:
        run_pipeline(args.task, args.model, args.fraction)
    

if __name__ == "__main__":
     main()