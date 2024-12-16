import os
import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from libs.helper import load_eer_thresholds, calculate_eer_threshold
from libs.dataloader import MultiModalDataLoader
from libs.model import ECG_Inception, PPG_Inception, EDA_LSTM


logfile = None
def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    if logfile is not None:
        with open(logfile, "a") as f:
            print(*args, **kwargs, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, help="The subject id") # ex: c1s01
    parser.add_argument('-v', '--version', type=int, help="Version of the data") # ex: 1
    parser.add_argument('--mode', default='mv', help="Mode of the model: avgp or mv")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--out_dir', default="./results")
    args = parser.parse_args()

    ecg_data_path = os.path.join(args.data_dir, args.subject, 'ecg.pkl')
    ppg_data_path = os.path.join(args.data_dir, args.subject, 'ppg.pkl')
    eda_data_path = os.path.join(args.data_dir, args.subject, 'eda.pkl')
    temp_data_path = os.path.join(args.data_dir, args.subject, 'temp.pkl')
    metadata_path = os.path.join(args.data_dir, args.subject, 'metadata.json')

    ecg_df = pd.read_pickle(ecg_data_path)
    ppg_df = pd.read_pickle(ppg_data_path)
    eda_df = pd.read_pickle(eda_data_path)
    temp_df = pd.read_pickle(temp_data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if args.version is None:
        versions = list(metadata.keys())
    else:
        versions = [f"v{args.version}"]

    out_dir = os.path.join(args.out_dir, f'MoE_{args.mode}', args.subject)
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, "main.log")
    with open(logfile, 'w') as f:
        pass

    eer_thresholds = load_eer_thresholds(args.subject, args.out_dir)
    print_and_log(f"EER Thresholds: {eer_thresholds}")
    for i, version in enumerate(versions):
        ecg_model = ECG_Inception()
        ppg_model = PPG_Inception()
        eda_model = EDA_LSTM()
        print_and_log(f"Subject: {args.subject}, Version: {version}")
        ecg_eer_threshold = eer_thresholds['ecg'][i]
        ppg_eer_threshold = eer_thresholds['ppg'][i]
        eda_eer_threshold = eer_thresholds['eda'][i]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for data_type in ['ecg', 'ppg', 'eda']:
            out_ver_dir = os.path.join(args.out_dir, data_type, args.subject, version)
            ckpt_path = os.path.join(out_ver_dir, "best.pth")
            assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

            if data_type == 'ecg':
                ecg_model.load_state_dict(torch.load(ckpt_path))
                ecg_model.eval()
                ecg_model.to(device)
            elif data_type == 'ppg':
                ppg_model.load_state_dict(torch.load(ckpt_path))
                ppg_model.eval()
                ppg_model.to(device)
            elif data_type == 'eda':
                eda_model.load_state_dict(torch.load(ckpt_path))
                eda_model.eval()
                eda_model.to(device)

        loader = MultiModalDataLoader(ecg_df, ppg_df, eda_df, temp_df, metadata[version], verbose=False)
        train_loader = loader.get_loader("train", batch_size=1, shuffle=False)
        val_loader = loader.get_loader("val", batch_size=1, shuffle=False)

        train_cgm_df = defaultdict(list)
        with torch.no_grad():
            for ecg_data, ppg_data, eda_data, temp_data, hypo_label, (glucose, CGM_idx) in tqdm.tqdm(train_loader):
                ecg_data = ecg_data.float().to(device).squeeze(0)
                ppg_data = ppg_data.float().to(device).squeeze(0)
                eda_data = eda_data.float().to(device).squeeze(0)
                temp_data = temp_data.float().to(device).squeeze(0)
                hypo_label = hypo_label.to(device)

                ecg_output = ecg_model(ecg_data)
                ppg_output = ppg_model(ppg_data)
                eda_output = eda_model(eda_data[0].unsqueeze(0), eda_data[1].unsqueeze(0))

                if ecg_data.shape[0] == 1:
                    ecg_output = ecg_output.unsqueeze(0)
                if ppg_data.shape[0] == 1:
                    ppg_output = ppg_output.unsqueeze(0)
                eda_output = eda_output.unsqueeze(0)

                ecg_output = ecg_output.cpu().numpy()
                ppg_output = ppg_output.cpu().numpy()
                eda_output = eda_output.cpu().numpy()
                temp_output = temp_data.cpu().numpy().mean()
                hypo_label = hypo_label.item()

                if args.mode == 'avgp':
                    ecg_output = ecg_output.mean()
                    ppg_output = ppg_output.mean()
                    eda_output = eda_output.mean()

                elif args.mode == 'mv':
                    binarized_ecg_output = (ecg_output > ecg_eer_threshold)
                    ecg_output = (binarized_ecg_output.sum(axis=0) > (binarized_ecg_output.shape[0]//2)).astype(int)
                    binarized_ppg_output = (ppg_output > ppg_eer_threshold)
                    ppg_output = (binarized_ppg_output.sum(axis=0) > (binarized_ppg_output.shape[0]//2)).astype(int)
                    binarized_eda_output = (eda_output > eda_eer_threshold)
                    eda_output = (binarized_eda_output.sum(axis=0) > (binarized_eda_output.shape[0]//2)).astype(int)
                else:
                    raise ValueError(f"Invalid mode: {args.mode}")

                train_cgm_df['ecg'].append(ecg_output)
                train_cgm_df['ppg'].append(ppg_output)
                train_cgm_df['eda'].append(eda_output)
                train_cgm_df['temp'].append(temp_output)
                train_cgm_df['label'].append(hypo_label)
        train_cgm_df = pd.DataFrame(train_cgm_df)


        val_df = defaultdict(list)
        val_cgm_df = defaultdict(list)
        with torch.no_grad():
            for ecg_data, ppg_data, eda_data, temp_data, hypo_label, (glucose, CGM_idx) in tqdm.tqdm(val_loader):
                ecg_data = ecg_data.float().to(device).squeeze(0)
                ppg_data = ppg_data.float().to(device).squeeze(0)
                eda_data = eda_data.float().to(device).squeeze(0)
                temp_data = temp_data.float().to(device).squeeze(0)
                hypo_label = hypo_label.to(device)

                ecg_output = ecg_model(ecg_data)
                ppg_output = ppg_model(ppg_data)
                eda_output = eda_model(eda_data[0].unsqueeze(0), eda_data[1].unsqueeze(0))

                if ecg_data.shape[0] == 1:
                    ecg_output = ecg_output.unsqueeze(0)
                if ppg_data.shape[0] == 1:
                    ppg_output = ppg_output.unsqueeze(0)
                eda_output = eda_output.unsqueeze(0)

                ecg_output = ecg_output.cpu().numpy()
                ppg_output = ppg_output.cpu().numpy()
                eda_output = eda_output.cpu().numpy()
                temp_output = temp_data.cpu().numpy()
                temp_output = temp_output.mean()
                hypo_label = hypo_label.item()

                binarized_ecg_output = (ecg_output > ecg_eer_threshold)
                binarized_ecg_output = (binarized_ecg_output.sum(axis=0) > (binarized_ecg_output.shape[0]//2)).astype(int)
                binarized_ppg_output = (ppg_output > ppg_eer_threshold)
                binarized_ppg_output = (binarized_ppg_output.sum(axis=0) > (binarized_ppg_output.shape[0]//2)).astype(int)
                binarized_eda_output = (eda_output > eda_eer_threshold)
                binarized_eda_output = (binarized_eda_output.sum(axis=0) > (binarized_eda_output.shape[0]//2)).astype(int)
                
                if args.mode == 'avgp':
                    ecg_output = ecg_output.mean()
                    ppg_output = ppg_output.mean()
                    eda_output = eda_output.mean()
                elif args.mode == 'mv':
                    ecg_output = binarized_ecg_output
                    ppg_output = binarized_ppg_output
                    eda_output = binarized_eda_output
                else:
                    raise ValueError(f"Invalid mode: {args.mode}")
                
                val_df['ecg'].append(binarized_ecg_output)
                val_df['ppg'].append(binarized_ppg_output)
                val_df['eda'].append(binarized_eda_output)
                val_df['label'].append(hypo_label)


                val_cgm_df['ecg'].append(ecg_output)
                val_cgm_df['ppg'].append(ppg_output)
                val_cgm_df['eda'].append(eda_output)
                val_cgm_df['temp'].append(temp_output)
                val_cgm_df['label'].append(hypo_label)

        # plot the accuracy for ensemble
        val_df = pd.DataFrame(val_df)
        val_cgm_df = pd.DataFrame(val_cgm_df)

        # individual model accuracy
        ecg_acc = (val_df['ecg'] == val_df['label']).mean()
        ppg_acc = (val_df['ppg'] == val_df['label']).mean()
        eda_acc = (val_df['eda'] == val_df['label']).mean()


        # Initialize logistic regression model
        MoE_model = LogisticRegression(class_weight='balanced')
        # Train the model
        MoE_model.fit(train_cgm_df[['ecg', 'ppg', 'eda', 'temp']], train_cgm_df['label'])

        # Predict on the validation set
        val_pred = MoE_model.predict_proba(val_cgm_df[['ecg', 'ppg', 'eda', 'temp']])[:, 1]
        eer_threshold = calculate_eer_threshold(val_pred, val_cgm_df['label'])
        val_cgm_df['pred'] = (val_pred > eer_threshold).astype(int)

        # Calculate the accuracy
        acc = (val_cgm_df['pred'] == val_cgm_df['label']).mean()
        print_and_log(f"Majority Vote Acc - ECG: {ecg_acc:.2f}| PPG: {ppg_acc:.2f}| EDA: {eda_acc:.2f}")
        print_and_log(f"MoE Acc: {acc:.2f}")
        print_and_log("=========================================")

        # save the results
        os.makedirs(os.path.join(out_dir, version), exist_ok=True)
        train_cgm_df.to_csv(os.path.join(out_dir, version, "MoE_train.csv"), index=False)
        val_cgm_df.to_csv(os.path.join(out_dir, version, "MoE_val.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, version, "majority_vote_val.csv"), index=False)