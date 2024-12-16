import os
import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from libs.dataloader import MultiModalDataLoader
from libs.model import ECG_Inception, PPG_Inception, EDA_LSTM
from libs.helper import calculate_eer_threshold

logfile = None
def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    if logfile is not None:
        with open(logfile, "a") as f:
            print(*args, **kwargs, file=f)

def load_eer_thresholds(subject, out_dir):
    ecg_results_path = os.path.join(out_dir, 'ecg', subject, 'results.json')
    ppg_results_path = os.path.join(out_dir, 'ppg', subject, 'results.json')
    eda_results_path = os.path.join(out_dir, 'eda', subject, 'results.json')

    with open(ecg_results_path, 'r') as f:
        ecg_results = json.load(f)
    with open(ppg_results_path, 'r') as f:
        ppg_results = json.load(f)
    with open(eda_results_path, 'r') as f:
        eda_results = json.load(f)
    eer_thresholds = {
        "ecg": ecg_results['eer_thresholds'],
        "ppg": ppg_results['eer_thresholds'],
        "eda": eda_results['eer_thresholds']
    }
    return eer_thresholds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, help="The subject id") # ex: c1s01
    parser.add_argument('-v', '--version', type=int, help="Version of the data") # ex: 1
    parser.add_argument('--mode', default='avgp', help="Mode of the model: avgp or mv")
    parser.add_argument('--out_dir', default="./results")
    args = parser.parse_args()

    metadata_path = os.path.join('./data', args.subject, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if args.version is None:
        versions = list(metadata.keys())
    else:
        versions = [f"v{args.version}"]

    types = ['ECG+PPG+EDA+TEMP', 'ECG+PPG+EDA', 'ECG+PPG', 'PPG+EDA', 'EDA+TEMP']
    out_dir = os.path.join(args.out_dir, 'MoE_{}'.format(args.mode), args.subject)
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, "LR.log")
    with open(logfile, 'w') as f:
        pass

    
    # input_data = ['ecg', 'ppg', 'eda']
    # if args.mode == 0:
    #     input_data = ['ecg', 'ppg', 'eda']
    # elif args.mode == 1:
    #     input_data = ['ecg', 'ppg']
    # elif args.mode == 2:
    #     input_data = ['ppg', 'eda']
    
    for i, version in enumerate(versions):
        print_and_log(f"Subject: {args.subject}, Version: {version}")
        
        MoE_dir = os.path.join(args.out_dir, 'MoE_{}'.format(args.mode), args.subject, version)
        train_df = pd.read_csv(os.path.join(MoE_dir, "MoE_train.csv"))
        val_df = pd.read_csv(os.path.join(MoE_dir, "MoE_val.csv"))

        for type in types:
            _type = type.split('+')
            input_data = [t.lower() for t in _type]
            
            # Initialize logistic regression model
            MoE_model = LogisticRegression(class_weight='balanced')
            # Train the model
            MoE_model.fit(train_df[input_data], train_df['label'])

            # Predict on the validation set
            val_pred = MoE_model.predict(val_df[input_data])
            val_df['pred'] = val_pred

            # Calculate the accuracy
            acc = (val_df['pred'] == val_df['label']).mean()
            print_and_log(f"{type} Acc: {acc:.2f}")
        print_and_log("=========================================")

        # # save the results
        # os.makedirs(os.path.join(out_dir, version), exist_ok=True)
        # train_df.to_csv(os.path.join(out_dir, version, "MoE_train.csv"), index=False)
        # val_df.to_csv(os.path.join(out_dir, version, "MoE_val.csv"), index=False)
        # val_df.to_csv(os.path.join(out_dir, version, "majority_vote_val.csv"), index=False)