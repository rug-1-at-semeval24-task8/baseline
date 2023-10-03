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
    parser.add_argument("-m", "--model", type=str, help="Which model to use (NB, RF, SVC, DT, KNN)")
    parser.add_argument("-a", "--all", action="store_true", help="Run all models on all tasks")
    parser.add_argument("-f", "--fraction", type=float, default=0.05, help="Fraction of the data to use (default 0.05)")
    parser.add_argument("-sm", "--split_method", type=str, help="Which split method to use (first, max_split, binary)")
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
        "train_data": "data/SubtaskA/subtaskA_train_monolingual.jsonl",
        "dev_data": "data/SubtaskC/subtaskC_dev.jsonl"
    }
}

models = {
    "NB": MultinomialNB(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(kernel="sigmoid"),
    "DT": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

regression_models = {
    "SGD": SGDRegressor(max_iter=10000, verbose=1),
    "LR": LinearRegression(),
    "LSVR": LinearSVR(),
    "SVR": SVR(kernel="sigmoid")
}

def binary_split(array):
    start = 0
    end = len(array) - 1
    mid = int((start + end) / 2)
    while (start < mid):
        before = mean(array[start:mid])
        after = mean(array[mid:end])
        bias_before = abs(before - 0)
        bias_after = abs(1 - after)
        if bias_before > bias_after:
            end = mid - 1
            mid = int((start + end) / 2)
        else:
            start = mid
            mid = int((start + end) / 2)
    return mid

# find most optimal splitting point
def max_split(tags):
    scores = [1 - mean(tags[:i]) + mean(tags[i:]) for i in range(1, len(tags))]
    split_index = scores.index(max(scores))
    return split_index

def find_split(model, data, method):
    tokens = data.split(" ")
    labels = model.predict(tokens).tolist()
    # find split index
    if method == "first":
        split_index = labels.index(1)
    elif method == "max":
        split_index = max_split(labels)
    elif method == "binary":
        split_index = binary_split(labels)
    return split_index

def task_c_predict(model, data, method):
    y = []
    for sample in data:
        y.append(find_split(model, sample, method))
    return y

def run_pipeline(task, model, fraction, split_method):
    print(f"--- Now running task {task} with model {model} on {int(fraction*100)}% of the data {f'using {split_method}-split as the split method' if task == 'c' else ''}---")

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
    
    if task == "c":
        y_pred = task_c_predict(pipe, X_dev, split_method)
        print(mean_absolute_error(y_dev, y_pred))
    else:
        y_pred = pipe.predict(X_dev)
        print(classification_report(y_dev, y_pred))
        print(confusion_matrix(y_dev, y_pred))

def main():
    args = create_arg_parser()
    if args.all:
        for task in tasks.keys():
            for model in models.keys():
                run_pipeline(task, model, 1.0)
    else:
        run_pipeline(args.task, args.model, args.fraction, args.split_method)
    

if __name__ == "__main__":
     main()