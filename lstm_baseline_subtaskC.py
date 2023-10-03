import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np

torch.manual_seed(1)
device = torch.device("cuda")


def evaluate_position_difference(actual_position, predicted_position):
    """
    Compute the absolute difference between the actual and predicted start positions.

    Args:
        - actual_position (int): Actual start position of machine-generated text.
        - predicted_position (int): Predicted start position of machine-generated text.

    Returns:
        - int: Absolute difference between the start positions.
    """
    return abs(actual_position - predicted_position)


def get_start_position(sequence, mapping=None, token_level=False):
    """
    Get the start position from a sequence of labels or predictions.

    Args:
        - sequence (np.array): A sequence of labels or predictions.
        - mapping (np.array): Mapping from index to word for the sequence.
        - token_level (bool): If True, return positional indices; else, return word mappings.

    Returns:
        - int or str: Start position in the sequence.
    """
    # Locate the position of label '1'
    if mapping is not None:
        mask = mapping != -100

    index = np.where(sequence == 1)[0]
    value = index[0] if index.size else (len(sequence) - 1)

    if not token_level:
        value = mapping[value]

    return value


def evaluate_machine_start_position(
    labels, predictions, idx2word=None, token_level=False
):
    """
    Evaluate the starting position of machine-generated text in both predicted and actual sequences.

    Args:
        - labels (np.array): Actual labels.
        - predictions (np.array): Predicted labels.
        - idx2word (np.array): Mapping from index to word for each sequence in the batch.
        - token_level (bool): Flag to determine if evaluation is at token level. If True, return positional indices; else, return word mappings.

    Returns:
        - float: Mean absolute difference between the start positions in predictions and actual labels.
    """
    predicted_positions = predictions.argmax(axis=-1)
    actual_starts = []
    predicted_starts = []

    if not token_level and idx2word is None:
        raise ValueError(
            "idx2word must be provided if evaluation is at word level (token_level=False)"
        )

    for idx in range(labels.shape[0]):
        # Remove padding
        mask = labels[idx] != -100
        predict, label, mapping = (
            predicted_positions[idx][mask],
            labels[idx][mask],
            idx2word[idx][mask] if not token_level else None,
        )

        # If token_level is True, just use the index; otherwise, map to word
        predicted_value = get_start_position(predict, mapping, token_level)
        actual_value = get_start_position(label, mapping, token_level)

        predicted_starts.append(predicted_value)
        actual_starts.append(actual_value)

    position_differences = [
        evaluate_position_difference(actual, predict)
        for actual, predict in zip(actual_starts, predicted_starts)
    ]
    mean_position_difference = np.mean(position_differences)

    return mean_position_difference


def compute_metrics(p):
    pred, labels = p
    mean_absolute_diff = evaluate_machine_start_position(labels, pred, token_level=True)

    return {
        "mean_absolute_diff": mean_absolute_diff,
    }


def convert_to_tags(text, label):
    words = text.lower().split(" ")
    if label == -1:
        labels = ["H" for _ in words]
    else:
        labels = ["H" for _ in words[:label]]
        labels += ["M" for _ in words[label:]]
    return words, labels


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)


def get_starting_point(sequence):
    index = np.where(sequence == 1)[0]
    value = index[0] if index.size else (len(sequence) - 1)
    return value


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


if __name__ == "__main__":
    train_file = "data/SubtaskC/subtaskC_train.jsonl"
    dev_file = "data/SubtaskC/subtaskC_train.jsonl"

    with open(train_file, "r") as f:
        train_data_raw = [json.loads(line) for line in f]
    with open(dev_file, "r") as f:
        dev_data_raw = [json.loads(line) for line in f]

    training_data = []

    for d in train_data_raw:
        text = d["text"]
        label = d["label"]
        training_data.append(convert_to_tags(text, label))
    dev_data = []

    for d in dev_data_raw:
        text = d["text"]
        label = d["label"]
        dev_data.append(convert_to_tags(text, label))

    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {"H": 0, "M": 1}

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)).to(
        device
    )
    loss_function = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores_before = model(inputs)

    for epoch in range(20):
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    vals = []
    with torch.no_grad():
        for dt in dev_data:
            inputs = prepare_sequence(dt[0], word_to_ix)
            tag_scores = model(inputs)
            gold = np.array([0 if i == "H" else 1 for i in dt[1]])
            scores = tag_scores.cpu().detach()
            predicted_position = get_starting_point(np.argmax(scores, axis=1))
            actual_position = get_starting_point(gold)
            vals.append(abs(actual_position - predicted_position))

    print(sum(vals) / len(vals))
