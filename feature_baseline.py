from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVR, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
import pandas
import argparse
from statistics import mean

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="Which task to evaluate (a_mono, a_multi, b, c)")
    parser.add_argument("-m", "--model", type=str, help="Which model to use")
    parser.add_argument("-f", "--fraction", type=float, default=1.0, help="Fraction of the data to use (default 1.0 = 100%)")
    parser.add_argument("-cm", "--c_method", type=str, default="split", help="Method to use for subtask C (regression, split)")
    parser.add_argument("-sm", "--split_method", type=str, help="Which split method to use (first, max, binary)")
    return parser.parse_args()

# labels_a = {
#     0: "human",
#     1: "machine"
# }

# labels_b = {
#     0: "human",
#     1: "ChatGPT",
#     2: "Cohere",
#     3: "Davinci",
#     4: "Bloomz",
#     5: "Dolly"
# }

tasks = {
    "a_mono": {
        "train_data": "data/SubtaskA/subtaskA_train_monolingual.jsonl",
        "dev_data": "data/SubtaskA/subtaskA_dev_monolingual.jsonl"
    },
    "a_multi": {
        "train_data": "data/SubtaskA/subtaskA_train_mutlilingual.jsonl",
        "dev_data": "data/SubtaskA/subtaskA_dev_multilingual.jsonl"
    },
    "b": {
        "train_data": "data/SubtaskB/subtaskB_train.jsonl",
        "dev_data": "data/SubtaskB/subtaskB_dev.jsonl"
    },
    "c": {
        "train_data": "data/SubtaskC/subtaskC_train.jsonl",
        "dev_data": "data/SubtaskC/subtaskC_dev.jsonl"
    }
}

models = {
    # Classification models
    "NB": MultinomialNB(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(kernel="sigmoid"),
    "DT": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    # Regression models
    "SGD": SGDRegressor(max_iter=10000, verbose=1),
    "LR": LinearRegression(),
    "LSVR": LinearSVR(),
    "SVR": SVR(kernel="sigmoid")
}

# find splitting point using binary search
def binary_split(array):
    start = 0
    end = len(array) - 1
    mid = int((start + end) / 2)
    while (start < mid):
        # first half must be closest to human (0)
        score_first = 1 - mean(array[start:mid])
        # second half must be closest to machine (1)
        score_second = mean(array[mid:end])
        if score_first < score_second:
            # first half is further away from optimal,
            # so it might still contain mixed text
            end = mid - 1
            mid = int((start + end) / 2)
        else:
            # second half is further away from optimal,
            # so it might still contain mixed text
            start = mid
            mid = int((start + end) / 2)
    return mid

# find most optimal splitting point among all possible points
def max_split(tags):
    # score each split by sum of closeness of each part to its optimal value
    # (0 for first part, 1 for second part)
    scores = [1 - mean(tags[:i]) + mean(tags[i:]) for i in range(1, len(tags))]
    split_index = scores.index(max(scores))
    return split_index

# use split method to predict boundary of a single text
def find_split(model, data, method):
    tokens = data.split(" ")
    labels = model.predict(tokens).tolist()
    if method == "first":
        split_index = labels.index(1)
    elif method == "max":
        split_index = max_split(labels)
    elif method == "binary":
        split_index = binary_split(labels)
    return split_index

# process data text-by-text and predict boundaries
def split_predict(model, data, method):
    y = []
    for sample in data:
        y.append(find_split(model, sample, method))
    return y

def run_pipeline(task, model, fraction):
    print(f"--- Now running task {task} using a {model} and {int(fraction*100)}% of the data ---")

    train_data = pandas.read_json(tasks[task]["train_data"], lines=True)
    dev_data = pandas.read_json(tasks[task]["dev_data"], lines=True)

    train_data = train_data.sample(frac=fraction)
    dev_data = dev_data.sample(frac=fraction)

    X_train = train_data["text"]
    y_train = train_data["label"]
    X_dev = dev_data["text"]
    y_dev = dev_data["label"]        
    
    clf = models[model]

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=0.02, max_df=0.8), clf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_dev)
    
    if task == "c":
        print(mean_absolute_error(y_dev, y_pred))
    else:
        print(classification_report(y_dev, y_pred))
        print(confusion_matrix(y_dev, y_pred))

def run_pipeline_split(model, fraction, split_method):
    print(f"--- Now running task c using a {model} + {split_method}-split model and {int(fraction*100)}% of the data ---")
    
    train_data = pandas.read_json(tasks["a_mono"]["train_data"], lines=True)
    dev_data = pandas.read_json(tasks["c"]["dev_data"], lines=True)

    train_data = train_data.sample(frac=fraction)
    dev_data = dev_data.sample(frac=fraction)

    X_train = train_data["text"]
    y_train = train_data["label"]
    X_dev = dev_data["text"]
    y_dev = dev_data["label"]        
    
    clf = models[model]

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=0.02, max_df=0.8), clf)
    pipe.fit(X_train, y_train)

    y_pred = split_predict(pipe, X_dev, split_method)
    print(mean_absolute_error(y_dev, y_pred))

def main():
    args = create_arg_parser()
    if args.task == "c" and args.c_method == "split":
        run_pipeline_split(args.model, args.fraction, args.split_method)
    else:
        run_pipeline(args.task, args.model, args.fraction)

if __name__ == "__main__":
     main()