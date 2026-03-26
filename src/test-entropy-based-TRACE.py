import argparse
from pprint import pformat
import pandas as pd
import logging
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datasets import tqdm
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)


def get_wandb_run_name(args):
    return f"v2-our-entropy-grid-{args.grid}-metric-{args.metric}-model-{args.model}-context-{args.context_size}-{args.config}-train-topic-ood-llm-dataset-threshold-{args.threshold}"


def get_results(y_pred_probs, y_true, threshold, author_id_map, id_author_map, author_type_map):
    results = []
    assert len(y_pred_probs) == len(y_true)
    top1_probs = np.max(y_pred_probs, axis=1)
    top1_classes = np.argmax(y_pred_probs, axis=1)
    accepted_mask = top1_probs >= threshold
    class_f1s = []
    class_accuracies = []
    class_precisions = []
    class_recalls = []
    n_classes = len(author_id_map)
    for class_id in range(n_classes):
        class_mask = (y_true == class_id)

        if np.sum(class_mask) == 0:
            continue  # Skip classes with no samples

        class_predicted_correct = (top1_classes == class_id) & class_mask & accepted_mask

        class_recall = np.sum(class_predicted_correct) / np.sum(class_mask)

        predicted_as_class = (top1_classes == class_id) & accepted_mask
        if np.sum(predicted_as_class) > 0:
            class_precision = np.sum(class_predicted_correct) / np.sum(predicted_as_class)
        else:
            class_precision = 0

        class_accuracy = np.sum(class_predicted_correct) / np.sum(class_mask)

        if class_precision + class_recall > 0:
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        else:
            class_f1 = 0

        class_accuracies.append(class_accuracy)
        class_precisions.append(class_precision)
        class_recalls.append(class_recall)
        class_f1s.append(class_f1)
        author = id_author_map[class_id]
        results.append({
            'author': author,
            'label_id': class_id,
            'f1': class_f1,
            'acc': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
            'author_type': author_type_map[author],
        })
    return pd.DataFrame(results)


def get_unseen_results(y_pred_probs, threshold, unseen_author, author_type_map):
    # unseen author case - model says I don't know
    top1_probs = np.max(y_pred_probs, axis=1)
    print(top1_probs)
    accepted_mask = top1_probs < threshold
    y_pred = [int(curr) for curr in accepted_mask]
    y_true = [1] * len(top1_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'unseen_author': unseen_author,
        'f1': f1,
        'acc': acc,
        'precision': prec,
        'recall': rec,
        'author_type': author_type_map[unseen_author],
        'samples': len(top1_probs),
    }


# basically other closest model in the same family
FAMILY_MAP = {
    'gpt-4.1': 'gpt-5.1',
    'gpt-5.1': 'gpt-4.1',
    'gemini-2.5-pro': 'gemini-2.5-flash',
    'gemini-2.5-flash': 'gemini-2.5-pro',
    'qwen3-235b-a22b-2507': 'qwen3-max',
    'qwen3-max': 'qwen3-235b-a22b-2507',
}
assert len(FAMILY_MAP) == 6


def get_unseen_family_results(y_pred_probs, y_pred_labels, threshold, unseen_author, author_type_map, author_id_map):
    # unseen author case - model says I know and matches to other model from the family
    top1_probs = np.max(y_pred_probs, axis=1)
    accepted_mask = (top1_probs >= threshold) & (y_pred_labels == author_id_map[FAMILY_MAP[unseen_author]])
    y_pred = [int(curr) for curr in accepted_mask]
    y_true = [1] * len(top1_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'unseen_author': unseen_author,
        'f1': f1,
        'acc': acc,
        'precision': prec,
        'recall': rec,
        'author_type': author_type_map[unseen_author],
        'samples': len(top1_probs),
    }


def create_signature(data, grid_size, max_value):
    kde = gaussian_kde(data)

    x_grid = np.linspace(0, max_value, grid_size)
    y_grid = np.linspace(0, max_value, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(positions).reshape(X.shape)
    return X, Y, Z


def get_fingerprint(values, grid_size, max_value):
    pairs = [(values[i - 1], values[i]) for i in range(1, len(values))]
    pair_x = [pair[0] for pair in pairs]
    pair_y = [pair[1] for pair in pairs]
    data = np.vstack([pair_x, pair_y])
    X, Y, Z = create_signature(data, grid_size, max_value)
    return Z


def get_ref_fingerprint(args, method, author, data, max_value):
    author_data = data[data['author'] == author]
    print("Author: {}; {}".format(author, len(author_data)))
    fingerprints = []

    for _, row in author_data.iterrows():
        curr_novel = row['novel_name']
        entropies = np.load(
            f"./data/{method}/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-{method}.npz")[
            method]
        pairs = [(entropies[i - 1], entropies[i]) for i in range(1, len(entropies))]
        pair_x = [pair[0] for pair in pairs]
        pair_y = [pair[1] for pair in pairs]
        data = np.vstack([pair_x, pair_y])
        X, Y, Z = create_signature(data, args.grid, max_value)
        fingerprints.append(Z)
    return fingerprints


def compute_normals(Z, dx, dy):
    # Compute gradients using central differences
    Zx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * dx)
    Zy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * dy)
    # Normal vector: (-Zx, -Zy, 1)
    normals = np.stack([-Zx, -Zy, np.ones_like(Z)], axis=-1)
    # Normalize
    norm = np.sqrt(normals[..., 0] ** 2 + normals[..., 1] ** 2 + normals[..., 2] ** 2)
    normals = normals / norm[..., np.newaxis]
    return normals


def get_mean_normal_angle(Z1, Z2, grid_size, max_value):
    x = np.linspace(0, max_value, grid_size)
    y = np.linspace(0, max_value, grid_size)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    normals1 = compute_normals(Z1, dx, dy)
    normals2 = compute_normals(Z2, dx, dy)

    dot_product = np.sum(normals1 * normals2, axis=-1)
    dot_product = np.clip(dot_product, -1, 1)  # Avoid numerical errors
    angle_diff = np.arccos(dot_product) * 180 / np.pi

    mean_angle = np.mean(angle_diff)
    return mean_angle * 100


def get_score(metric, fingerprint1, fingerprint2, grid, max_value):
    if metric == 'cos_sim':
        return cosine_similarity([fingerprint1.flatten()], [fingerprint2.flatten()])[0, 0]
    elif metric == 'frob_norm':
        return -1 * np.linalg.norm(fingerprint1 - fingerprint2, ord=2)
    elif metric == 'wass_dist':
        raise NotImplementedError
    elif metric == 'js_dist':
        return 1 - jensenshannon(fingerprint1.flatten(), fingerprint2.flatten())
    elif metric == "ssim":
        raise NotImplementedError
    elif metric == 'norm_mean':
        Z1 = fingerprint1
        Z2 = fingerprint2
        return -1 * get_mean_normal_angle(Z1, Z2, grid, max_value)


def get_predicted_author(args, entropies, author_ref_fingerprints_map, metric, max_value):
    curr_fingerprint = get_fingerprint(entropies, args.grid, max_value)
    scores = []
    for author, author_fingerprints in author_ref_fingerprints_map.items():
        scores.append(np.max([get_score(metric, author_fingerprint, curr_fingerprint, args.grid, max_value) for author_fingerprint in author_fingerprints]))
    return np.argmax(scores), np.array(scores)


def get_entropy_predictions(args, test_data, author_ref_fingerprints_map, max_value):
    y_pred = []
    y_scores = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        curr_novel = row['novel_name']
        entropy = np.load(
            f"./data/entropy/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-entropy.npz")[
            'entropy']
        pred_author, score = get_predicted_author(args, entropy, author_ref_fingerprints_map, args.metric, max_value)
        y_pred.append(pred_author)
        y_scores.append(score)
    return np.array(y_pred), np.array(y_scores)


def get_unseen_entropy_predictions(args, test_data, unseen_author, max_value):
    author_ref_fingerprints_map = {}
    for author in authors:
        if author == unseen_author:
            continue
        author_ref_fingerprints_map[author] = get_ref_fingerprint(args, "entropy", author, train_data, max_value)
    y_pred = []
    y_scores = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        curr_novel = row['novel_name']
        entropy = np.load(
            f"./data/entropy/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-entropy.npz")[
            'entropy']
        pred_author, score = get_predicted_author(args, entropy, author_ref_fingerprints_map, args.metric, max_value)
        y_pred.append(pred_author)
        y_scores.append(score)
    return np.array(y_pred), np.array(y_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="dataset split")
    parser.add_argument('--threshold', type=str, help="threshold used for prediction")
    parser.add_argument('--grid', type=str, help="grid size used for final fingerprint density map")
    parser.add_argument('--metric', type=str, help="similarity metric used")
    parser.add_argument('--model', type=str, help="evaluator language model used")
    parser.add_argument('--hf_model', type=str, help="HF name for evaluator language model")
    parser.add_argument('--context_size', type=str, help="Context used in the evaluator language model")
    args = parser.parse_args()

    args.threshold = float(args.threshold)
    args.grid = int(args.grid)
    args.context_size = int(args.context_size)
    logging.info("Args:\n%s", pformat(args))


    THRESHOLD = args.threshold
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    vocab_size = tokenizer.vocab_size
    max_value = np.log2(vocab_size)
    print("vocab size: ", vocab_size, "max_value: ", max_value)
    train_data = pd.read_csv(
        f"./ghostwritebench/{args.config}-train-dataset.csv")

    NO_THRESHOLD = 0
    if args.metric == "norm_mean":
        NO_THRESHOLD = -1 * 100 * 180

    authors = sorted(list(set(train_data['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}
    author_type_map = {}
    for author in authors:
        author_type = list(set(train_data[train_data['author'] == author]['author_type']))
        assert len(author_type) == 1
        author_type = author_type[0]
        author_type_map[author] = author_type

    test_data = pd.read_csv(f"./ghostwritebench/{args.config}-test-dataset.csv")
    ID_test_data = test_data[test_data['type'] == 'ID']
    ID_test_texts = list(ID_test_data['text'])
    ID_test_labels = np.array([author_id_map[author] for author in list(ID_test_data['author'])])
    assert len(ID_test_texts) == len(ID_test_labels)

    OOD_test_data = test_data[test_data['type'] == 'OOD']
    OOD_test_texts = list(OOD_test_data['text'])
    OOD_test_labels = np.array([author_id_map[author] for author in list(OOD_test_data['author'])])
    assert len(OOD_test_texts) == len(OOD_test_labels)

    author_ref_fingerprints_map = {}
    for author in authors:
        author_ref_fingerprints_map[author] = get_ref_fingerprint(args, "entropy", author, train_data,
                                                                max_value)

    ID_test_predictions, ID_test_probs = get_entropy_predictions(args, ID_test_data, author_ref_fingerprints_map, max_value)
    OOD_test_predictions, OOD_test_probs = get_entropy_predictions(args, OOD_test_data, author_ref_fingerprints_map, max_value)

    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, NO_THRESHOLD, author_id_map, id_author_map, author_type_map)
    print("\nNO THRESHOLD:")
    print("ID:")
    print("less_prolific: ", np.mean(no_threshold_ID_results[no_threshold_ID_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ", np.mean(no_threshold_ID_results[no_threshold_ID_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(no_threshold_ID_results['f1']))

    no_threshold_OOD_results = get_results(OOD_test_probs, OOD_test_labels, NO_THRESHOLD, author_id_map, id_author_map, author_type_map)
    print("OOD:")
    print("less_prolific: ", np.mean(no_threshold_OOD_results[no_threshold_OOD_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ", np.mean(no_threshold_OOD_results[no_threshold_OOD_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(no_threshold_OOD_results['f1']))

    no_threshold_result = {
            "no_threshold/ID/less_prolific": np.mean(no_threshold_ID_results[no_threshold_ID_results['author_type'] == 'less_prolific']['f1']),
            "no_threshold/ID/more_prolific": np.mean(no_threshold_ID_results[no_threshold_ID_results['author_type'] == 'more_prolific']['f1']),
            "no_threshold/ID/all": np.mean(no_threshold_ID_results['f1']),
            "no_threshold/OOD/less_prolific": np.mean(no_threshold_OOD_results[no_threshold_OOD_results['author_type'] == 'less_prolific']['f1']),
            "no_threshold/OOD/more_prolific": np.mean(no_threshold_OOD_results[no_threshold_OOD_results['author_type'] == 'more_prolific']['f1']),
            "no_threshold/OOD/all": np.mean(no_threshold_OOD_results['f1'])
    }
    no_threshold_ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-no_threshold_ID_results.csv", index=False)
    no_threshold_OOD_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-no_threshold_OOD_results.csv",
                                   index=False)

    print("\nTHRESHOLD:", THRESHOLD)
    ID_results = get_results(ID_test_probs, ID_test_labels, THRESHOLD, author_id_map, id_author_map, author_type_map)
    print("ID:")
    print("less_prolific: ", np.mean(ID_results[ID_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ", np.mean(ID_results[ID_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(ID_results['f1']))

    OOD_results = get_results(OOD_test_probs, OOD_test_labels, THRESHOLD, author_id_map, id_author_map, author_type_map)
    print("OOD:")
    print("less_prolific: ", np.mean(OOD_results[OOD_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ", np.mean(OOD_results[OOD_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(OOD_results['f1']))


    ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_ID_results.csv",
                                   index=False)
    OOD_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_OOD_results.csv",
                                    index=False)
    unseen_results = []
    unseen_family_results = []
    no_threshold_unseen_family_results = []
    for unseen_author in authors:
        print("Unseen author: ", unseen_author)
        curr_authors = authors.copy()
        curr_authors.remove(unseen_author)
        assert len(curr_authors) == 9
        curr_author_id_map = {author: idx for idx, author in enumerate(curr_authors)}
        unseen_author_data = test_data[test_data['author'] == unseen_author]
        unseen_author_labels, unseen_author_probs = get_unseen_entropy_predictions(args, unseen_author_data, unseen_author, max_value)
        unseen_results.append(get_unseen_results(unseen_author_probs, THRESHOLD, unseen_author, author_type_map))
        if unseen_author in FAMILY_MAP:
            unseen_family_results.append(
                get_unseen_family_results(unseen_author_probs, unseen_author_labels, THRESHOLD, unseen_author,
                                          author_type_map, author_id_map))
            no_threshold_unseen_family_results.append(
                get_unseen_family_results(unseen_author_probs, unseen_author_labels, NO_THRESHOLD, unseen_author,
                                          author_type_map, curr_author_id_map))
    unseen_results = pd.DataFrame(unseen_results)
    unseen_family_results = pd.DataFrame(unseen_family_results)
    no_threshold_unseen_family_results = pd.DataFrame(no_threshold_unseen_family_results)
    print("Unseen:")
    print("less_prolific: ", np.mean(unseen_results[unseen_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ", np.mean(unseen_results[unseen_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(unseen_results['f1']))
    print("Unseen Family:")
    print("less_prolific: ",
          np.mean(unseen_family_results[unseen_family_results['author_type'] == 'less_prolific']['f1']))
    print("more_prolific: ",
          np.mean(unseen_family_results[unseen_family_results['author_type'] == 'more_prolific']['f1']))
    print("all: ", np.mean(unseen_family_results['f1']))

    threshold_result = {
            "threshold/ID/less_prolific": np.mean(ID_results[ID_results['author_type'] == 'less_prolific']['f1']),
            "threshold/ID/more_prolific": np.mean(ID_results[ID_results['author_type'] == 'more_prolific']['f1']),
            "threshold/ID/all": np.mean(ID_results['f1']),
            "threshold/OOD/less_prolific": np.mean(OOD_results[OOD_results['author_type'] == 'less_prolific']['f1']),
            "threshold/OOD/more_prolific": np.mean(OOD_results[OOD_results['author_type'] == 'more_prolific']['f1']),
            "threshold/OOD/all": np.mean(OOD_results['f1']),
            "threshold/Unseen/less_prolific": np.mean(unseen_results[unseen_results['author_type'] == 'less_prolific']['f1']),
            "threshold/Unseen/more_prolific": np.mean(unseen_results[unseen_results['author_type'] == 'more_prolific']['f1']),
            "threshold/Unseen/all": np.mean(unseen_results['f1']),
            "threshold/UnseenFamily/less_prolific": np.mean(
                unseen_family_results[unseen_family_results['author_type'] == 'less_prolific']['f1']),
            "threshold/UnseenFamily/more_prolific": np.mean(
                unseen_family_results[unseen_family_results['author_type'] == 'more_prolific']['f1']),
            "threshold/UnseenFamily/all": np.mean(unseen_family_results['f1']),
            "no_threshold/UnseenFamily/less_prolific": np.mean(
                no_threshold_unseen_family_results[no_threshold_unseen_family_results['author_type'] == 'less_prolific']['f1']),
            "no_threshold/UnseenFamily/more_prolific": np.mean(
                no_threshold_unseen_family_results[no_threshold_unseen_family_results['author_type'] == 'more_prolific']['f1']),
            "no_threshold/UnseenFamily/all": np.mean(no_threshold_unseen_family_results['f1'])
    }
    unseen_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_unseen_results.csv",
                       index=False)
    unseen_family_results.to_csv(
        f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_unseen_family_results.csv",
        index=False)
    no_threshold_unseen_family_results.to_csv(
        f"./data/results/{get_wandb_run_name(args)}-threshold-{NO_THRESHOLD}_unseen_family_results.csv",
        index=False)
