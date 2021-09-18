import sklearn


def levenshtein_distance(first, second):
    distance = [[0 for _ in range(len(second) + 1)] for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (first[i - 1] != second[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)
    return distance[len(first)][len(second)]


def cer(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein_distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars


def wer(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_words = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        gt_words, pred_words = gt_text.split(), pred_text.split()
        lev_distances += levenshtein_distance(pred_words, gt_words)
        num_gt_words += len(gt_words)
    return lev_distances / num_gt_words


def string_accuracy(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    correct = 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        correct += int(pred_text == gt_text)
    return correct / len(gt_texts)


def acc(targets, outputs):
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy().argmax(axis=1)
    return sklearn.metrics.accuracy_score(targets, outputs)
