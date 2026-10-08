import argparse
import csv
from pathlib import Path

import numpy as np


DATA_DIR = Path("Training Data")
WEIGHTS_PATH = Path("Wights.npz")

CLASS_FILES = [
    "hand_close.txt",
    "hand_open.txt",
    "index_finger.txt",
    "palm_down.txt",
    "palm_up.txt",
    "peace.txt",
    "pinch.txt",
    "thumb_down.txt",
    "thumb_up.txt",
    "touch.txt",
]


def relu(values):
    return np.maximum(values, 0)


def relu_back(values):
    return values > 0


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_class_file(path, label):
    rows = []
    with path.open("r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                values = [float(value) for value in row[1:]]
            except ValueError as error:
                raise ValueError(f"Bad numeric value in {path}:{line_number}") from error
            if len(values) != 39:
                raise ValueError(f"Expected 39 features in {path}:{line_number}, got {len(values)}")
            rows.append(values)

    if not rows:
        raise ValueError(f"{path} has no samples")
    return np.array(rows, dtype=np.float64), np.full(len(rows), label, dtype=np.int64)


def load_dataset(data_dir):
    features = []
    labels = []
    class_names = []

    for label, filename in enumerate(CLASS_FILES):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing training data file: {path}")
        class_features, class_labels = load_class_file(path, label)
        features.append(class_features)
        labels.append(class_labels)
        class_names.append(path.stem)

    return np.vstack(features), np.concatenate(labels), np.array(class_names)


def train_validation_split(features, labels, validation_ratio, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * validation_ratio)))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )


def init_weights(seed):
    rng = np.random.default_rng(seed)
    return {
        "WI": rng.normal(0, np.sqrt(2 / 39), (39, 128)),
        "WH": rng.normal(0, np.sqrt(2 / 128), (128, 128)),
        "WH2": rng.normal(0, np.sqrt(2 / 128), (128, 64)),
        "WH3": rng.normal(0, np.sqrt(2 / 64), (64, 32)),
        "WO": rng.normal(0, np.sqrt(2 / 32), (32, 10)),
        "BI": np.zeros(128),
        "BH": np.zeros(128),
        "BH2": np.zeros(64),
        "BH3": np.zeros(32),
        "BO": np.zeros(10),
    }


def forward(features, weights):
    hidden_1_raw = features @ weights["WI"] + weights["BI"]
    hidden_1 = relu(hidden_1_raw)
    hidden_2_raw = hidden_1 @ weights["WH"] + weights["BH"]
    hidden_2 = relu(hidden_2_raw)
    hidden_3_raw = hidden_2 @ weights["WH2"] + weights["BH2"]
    hidden_3 = relu(hidden_3_raw)
    hidden_4_raw = hidden_3 @ weights["WH3"] + weights["BH3"]
    hidden_4 = relu(hidden_4_raw)
    logits = hidden_4 @ weights["WO"] + weights["BO"]
    probabilities = softmax(logits)
    cache = (
        features,
        hidden_1_raw,
        hidden_1,
        hidden_2_raw,
        hidden_2,
        hidden_3_raw,
        hidden_3,
        hidden_4_raw,
        hidden_4,
    )
    return probabilities, cache


def backward(probabilities, labels, cache, weights):
    batch_size = labels.size
    features, hidden_1_raw, hidden_1, hidden_2_raw, hidden_2, hidden_3_raw, hidden_3, hidden_4_raw, hidden_4 = cache
    grad_logits = probabilities.copy()
    grad_logits[np.arange(batch_size), labels] -= 1
    grad_logits /= batch_size

    grads = {}
    grads["WO"] = hidden_4.T @ grad_logits
    grads["BO"] = np.sum(grad_logits, axis=0)

    grad_hidden_4 = grad_logits @ weights["WO"].T * relu_back(hidden_4_raw)
    grads["WH3"] = hidden_3.T @ grad_hidden_4
    grads["BH3"] = np.sum(grad_hidden_4, axis=0)

    grad_hidden_3 = grad_hidden_4 @ weights["WH3"].T * relu_back(hidden_3_raw)
    grads["WH2"] = hidden_2.T @ grad_hidden_3
    grads["BH2"] = np.sum(grad_hidden_3, axis=0)

    grad_hidden_2 = grad_hidden_3 @ weights["WH2"].T * relu_back(hidden_2_raw)
    grads["WH"] = hidden_1.T @ grad_hidden_2
    grads["BH"] = np.sum(grad_hidden_2, axis=0)

    grad_hidden_1 = grad_hidden_2 @ weights["WH"].T * relu_back(hidden_1_raw)
    grads["WI"] = features.T @ grad_hidden_1
    grads["BI"] = np.sum(grad_hidden_1, axis=0)
    return grads


def loss_and_accuracy(features, labels, weights):
    probabilities, _ = forward(features, weights)
    loss = -np.mean(np.log(probabilities[np.arange(labels.size), labels] + 1e-9))
    accuracy = np.mean(np.argmax(probabilities, axis=1) == labels)
    return loss, accuracy


def train(args):
    features, labels, class_names = load_dataset(args.data_dir)
    x_train, y_train, x_val, y_val = train_validation_split(features, labels, args.validation_ratio, args.seed)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    weights = init_weights(args.seed)
    rng = np.random.default_rng(args.seed)
    best_weights = {name: value.copy() for name, value in weights.items()}
    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(y_train.size)
        for start in range(0, y_train.size, args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            probabilities, cache = forward(x_train[batch_indices], weights)
            grads = backward(probabilities, y_train[batch_indices], cache, weights)
            for name, grad in grads.items():
                weights[name] -= args.learning_rate * grad

        train_loss, train_accuracy = loss_and_accuracy(x_train, y_train, weights)
        val_loss, val_accuracy = loss_and_accuracy(x_val, y_val, weights)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = {name: value.copy() for name, value in weights.items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
            )

    np.savez(
        args.output,
        **best_weights,
        mean=mean,
        std=std,
        class_names=class_names,
    )
    print(f"saved {args.output} with best val_acc={best_val_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hand gesture classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
