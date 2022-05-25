import os
import torch
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from datasets.datasets_test import get_normal_dataset_test, get_test_loader_one_vs_all, get_test_loader_one_vs_one


def save_results(results, results_path):
    if isinstance(results, dict):
        results_df = pd.DataFrame.from_dict(results, orient='index')
    else:
        results_df = pd.DataFrame(results)
    results_df.to_csv(results_path)


def knn_score(train_feature_space, test_feature_space, n_neighbours=2):
    index = faiss.IndexFlatL2(train_feature_space.shape[1])
    index.add(train_feature_space)
    distances, _ = index.search(test_feature_space, n_neighbours)
    return np.sum(distances, axis=1)


def extract_feature_space(model, device, data_loader):
    feature_space = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in tqdm(data_loader, desc='Feature Extraction'):
            x = x.to(device)
            _, features = model(x)
            feature_space.append(features.cpu())
            all_labels.append(y)
        feature_space = torch.cat(feature_space, dim=0).contiguous().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    return feature_space, all_labels


def eval_one_vs_one(args, model, device, train_feature_space):
    auc_results = []
    for ano_label in range(args.nnd_class_len):
        test_loader = get_test_loader_one_vs_one(args.label, ano_label, args.normal_data_path, args.download_dataset,
                                                 args.eval_batch_size)
        test_feature_space, test_labels = extract_feature_space(model, device, test_loader)
        distances = knn_score(train_feature_space, test_feature_space)
        auc = roc_auc_score(test_labels, distances)
        auc_results.append(auc)
        print(f'AUROC on the One-vs-One setting for the anomaly class {ano_label} is: {auc}')
    return auc_results


def eval_one_vs_all(args, model, device, train_feature_space):
    test_loader_one_vs_all = get_test_loader_one_vs_all(args.dataset, args.label, args.normal_data_path,
                                                        args.download_dataset, args.eval_batch_size)
    test_feature_space, test_labels = extract_feature_space(model, device, test_loader_one_vs_all)
    distances = knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)
    print(f'AUROC on the One-vs-All setting is: {auc}')
    return auc


def evaluate_model(args, model, device):
    final_results = {}
    model.eval()
    normal_train_loader = get_normal_dataset_test(args.dataset, args.label, args.normal_data_path,
                                                  args.download_dataset, args.eval_batch_size)
    print('Extract training feature space')
    train_feature_space, _ = extract_feature_space(model, device, normal_train_loader)
    print('Evaluate on the One-vs-All setting:')
    auc_one_vs_all = eval_one_vs_all(args, model, device, train_feature_space)
    final_results['one_vs_all'] = auc_one_vs_all
    if args.nnd:
        print('Evaluate on the One-vs-One setting:')
        auc_one_vs_one = eval_one_vs_one(args, model, device, train_feature_space)
        save_results(auc_one_vs_one, os.path.join(args.output_dir, f'results_all_one_vs_one_{args.label}.csv'))
        final_results['one_vs_one'] = min(auc_one_vs_one)
    save_results(final_results, os.path.join(args.output_dir, f'results_{args.dataset}_{args.label}.csv'))
