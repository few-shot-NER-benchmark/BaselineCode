import json
import os
import sys
import numpy as np
from eval_util import batch_span_eval


def evaluate(submit_dir, truth_dir, output_dir):
    datasets = ['CoNLL', 'WikiGold', 'WNUT17', 'MIT_Movie',
                'MIT_Restaurant', 'SNIPS', 'ATIS', 'Multiwoz']
    optional_datasets = ['Onto', 'I2B2']
    dataset_f1 = []
    for dataset in (datasets + optional_datasets):
        pred_file_path = os.path.join(submit_dir, dataset, 'out.json')
        if not os.path.isfile(pred_file_path):
            microf1 = -0.9999
        else:
            with open(os.path.join(submit_dir, dataset, 'out.json')) as f_pred:
                pred = json.load(f_pred)
            with open(os.path.join(truth_dir, dataset, 'out.json')) as f_truth:
                truth = json.load(f_truth)
            predict, gold = [], []
            for k in truth:
                assert k in pred, \
                    f'Example not included in submission.\nWords: "{k}"\nDataset: "{dataset}"'
                example_truth = truth[k].split(' ')
                example_pred = pred[k].split(' ')
                gold.append(example_truth)
                predict.append(example_pred + ['O'] * max(0, len(example_truth) - len(example_pred)))
            npred, ngold, ncorrect = batch_span_eval(predict, gold)
            microp = ncorrect / npred if npred > 0 else 0
            micror = ncorrect / ngold if ngold > 0 else 0
            microf1 = 2 * microp * micror / (microp + micror) if (microp + micror) > 0 else 0
        print(f'{dataset}:{microf1}')
        dataset_f1.append(microf1)
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        for dataset, f1 in zip(datasets + optional_datasets, dataset_f1):
            f.write(f'{dataset}: {round(f1, 4)}\n')
        avg_public = np.mean(dataset_f1[:len(datasets)]) if all(x >= 0 for x in dataset_f1[:len(datasets)]) else -0.9999
        print(f'Avg_Public: {round(avg_public, 4)}')
        f.write(f'Avg_Public: {round(avg_public, 4)}\n')
        avg = np.mean(dataset_f1) if all(x >= 0 for x in dataset_f1) else -0.9999
        print(f'Avg: {round(avg, 4)}')
        f.write(f'Avg: {round(avg, 4)}\n')


if __name__ == '__main__':
    '''
    Usage: python evaluate.py <input_dir> <output_dir>

    input_dir: folder that contains two subfolders with data, 'ref' for reference data and 'res' for predictions.
    output_dir: folder where the output file 'scores.txt' is going to be stored
    '''
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        evaluate(submit_dir, truth_dir, output_dir)
