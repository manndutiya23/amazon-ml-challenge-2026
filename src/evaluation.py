import numpy as np


def f05(precision, recall):
    beta2 = 0.5 ** 2

    denominator = beta2 * precision + recall

    if denominator == 0:
        return 0.0

    return (
        (1 + beta2)
        * precision
        * recall
        / denominator
    )


def entity_f05(predicted, actual):
    """
    Calculate F0.5 for one Source 1 entity.
    """

    predicted = set(predicted)
    actual = set(actual)

    if not predicted and not actual:
        return 1.0

    if not predicted or not actual:
        return 0.0

    true_positive = len(predicted & actual)

    precision = true_positive / len(predicted)
    recall = true_positive / len(actual)

    return f05(precision, recall)


def macro_f05(predictions, ground_truth):
    """
    Macro-average F0.5 across Source 1 entities.

    predictions:
        dict[S1_ID] -> list of predicted S2/S3 IDs

    ground_truth:
        dict[S1_ID] -> list of true S2/S3 IDs
    """

    scores = []

    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, [])

        scores.append(
            entity_f05(predicted, actual)
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


if __name__ == "__main__":

    gt = {
        "S1-A": ["S2-1", "S3-1"],
        "S1-B": [],
        "S1-C": ["S2-2"],
    }

    pred = {
        "S1-A": ["S2-1", "S3-1"],
        "S1-B": [],
        "S1-C": ["S2-WRONG"],
    }

    print("Macro F0.5:", macro_f05(pred, gt))