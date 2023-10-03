# Task Baseline System

Note that the data itself is not provided here. You may find it at [mbzuai0nlp/SemEval2024-task8](https://github.com/mbzuai-nlp/SemEval2024-task8).

## Feature-based Baseline

### Setup

The code requires the <b>Scikit-learn</b> package. To install it run:
```
pip install scikit-learn
```

### Usage

```
python feature_baseline.py [options]
```

<b>Note:</b> the code assumes that the data is structured like the official data-set, and is located in `./data/`:
```
- project
  |- feature_baseline.py
  |- data
     |- SubtaskA...
     |- SubtaskB...
     |- SubtaskC...
```

#### Arguments

| Argument | Description |
| --- | --- |
| `-t`<br/>`--task` | Specify which task to run <br/> Values: `a_mono, a_multi, b, c` |
| `-m`<br/>`--model` | Specify which model to use <br/> Values: `NB, DT, RF, KNN, SVM, SGD`* <br/> * SGD is only used for task C in  regression mode |
| `-cm`<br/>`--c_method` | Specify which method to use for task C <br/> Values: `regression, split` <br/> <b>`regression`</b> method uses a regression model specified by `-m` trained on subtask C data. <b>`split`</b> method uses a classifier model specified by `-m` trained on subtask A (monolingual) data and employing a split-method specified by `-sm`.|
| `-sm`<br/>`--split_method` | Specify which split-method to use for task C in split mode <br/> Values: `first, max, binary` <br/> <b>`first`</b> picks the index of the first token which is classified as machine-generated. <b>`max`</b> scores all possible splits and selects the best one. <b>`binary`</b> uses binary search to find the point of change. |
| `-f`<br/>`--fraction` | Specify what fraction of the data to use <br/> Default is 1.0 (=100%)



#### Examples

Subtask A monolingual using a Decision Tree classifier:
```
python feature_baseline.py -t a_mono -m DT
```
Subtask A multilingual using a Random Forest classifier:
```
python feature_baseline.py -t a_multi -m RF
```

Subtask B using a Random Forest classifier:
```
python feature_baseline.py -t b -m RF
```

Subtask C using a SGD regression model:
```
python feature_baseline.py -t c -cm regression -m SGD
```
Subtask C using a Decision Tree model + binary split:
```
python feature_baseline.py -t c -cm split -m DT -sm binary
```
