import os
import json
import time
import tqdm
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from libs.model import ECG_Inception, PPG_Inception, EDA_LSTM
from torch.utils.data import DataLoader
from libs.dataloader import SeNSEDataset, BalancedBatchSampler
## multi-gpu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

# torchrun command sets the following environment variables: RANK, LOCAL_RANK, WORLD_SIZE
# torchrun --standalone --nproc_per_node=4 train.py
def init_ddp():
    # RANK is the global rank of the current process across all nodes
    # LOCAL_RANK is the rank of the current process on the current node
    # WORLD_SIZE is the number of processes participating in the run (usually the number of GPUs)
    ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available(), "DistributedDataLoader requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device

logfile = None
def print_and_log(*args, **kwargs):
    if int(os.environ.get("RANK", -1)) == 0 or int(os.environ.get("RANK", -1)) == -1:
        print(*args, **kwargs)
        if logfile is not None:
            with open(logfile, "a") as f:
                print(*args, **kwargs, file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, help="The subject id") # ex: c1s01
    parser.add_argument('-v', '--version', type=int, help="Version of the data") # ex: 1
    parser.add_argument('--data_type', help="Type of data: ecg or ppg or eda")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--config_dir', default="./configs")
    parser.add_argument('--out_dir', default="./results")
    args = parser.parse_args()

    with open(f"{args.config_dir}/{args.data_type}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print_and_log(f"Config: {config}")


    out_dir = os.path.join(args.out_dir, args.data_type, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    data_path = f'{args.data_dir}/{args.subject}/{args.data_type}.pkl'
    metadata_path = f'{args.data_dir}/{args.subject}/metadata.json'
    data = pd.read_pickle(data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if args.version is None:
        versions = list(metadata.keys())
    else:
        versions = [f"v{args.version}"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = init_ddp()
    print(f"Running with DDP: {ddp}, device: {device}, world size: {ddp_world_size}", flush=True)

    for version in versions:
        print_and_log(f"Training version {version}")
        out_ver_dir = os.path.join(out_dir, f"{version}")
        os.makedirs(out_ver_dir, exist_ok=True)

        if master_process:
            ckpt_path = os.path.join(out_ver_dir, "best.pth")
            logfile = os.path.join(out_ver_dir, "train.log")
            with open(logfile, 'w') as f:
                pass

        train_data = SeNSEDataset(data, metadata[version]['train'], data_type=args.data_type, verbose=False)
        val_data = SeNSEDataset(data, metadata[version]['val'], data_type=args.data_type, verbose=False)

        print_and_log("Training on device: {}".format(device))
        print_and_log("Normal/Hypo ratio: {}".format(train_data.normal_hypo_ratio))
        if ddp:
            train_sampler = BalancedBatchSampler(data_source=train_data, batch_size=config['batch_size'])
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], sampler=train_sampler)
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], sampler=DistributedSampler(val_data))
        else:
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)

        if args.data_type == 'ecg':
            model = ECG_Inception(normal_hypo_ratio=train_data.normal_hypo_ratio)
        elif args.data_type == 'ppg':
            model = PPG_Inception(normal_hypo_ratio=train_data.normal_hypo_ratio)
        elif args.data_type == 'eda':
            model = EDA_LSTM(normal_hypo_ratio=train_data.normal_hypo_ratio)
        else:
            raise ValueError(f"Unknown data type: {args.data_type}")
        model.to(device)

        print_and_log("Model size: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(config['patience']*0.7), verbose=True)

        # training
        best_loss = 1e9
        training_losses = []
        validating_losses = []
        saved_epoch = 0
        early_stopping = 0
        for epoch in range(config['epochs']):
            model.train()
            training_loss = 0
            if ddp:
                train_sampler.set_epoch(epoch)
            else:
                train_data.stratified_sampling(batch_size=config['batch_size'])

            start_t = time.time()
            
            num_batches = len(train_loader)
            for i, (signal_data, hypo_label, glucose, cgm_idx) in enumerate(train_loader):
                optimizer.zero_grad()
                # Forward pass
                if args.data_type == 'eda':
                    phasic, tonic = signal_data
                    phasic, tonic, hypo_label, glucose = phasic.float().to(device), tonic.float().to(device), hypo_label.float().to(device), glucose.float().to(device)
                    pred_label = model(phasic, tonic)
                    num_data = phasic.shape[0]
                    if phasic.shape[0] == 1:
                        pred_label = pred_label.unsqueeze(0)
                else:
                    signal_data, hypo_label, glucose = signal_data.float().to(device), hypo_label.float().to(device), glucose.float().to(device)
                    pred_label = model(signal_data)
                    num_data = signal_data.shape[0]
                    if signal_data.shape[0] == 1:
                        pred_label = pred_label.unsqueeze(0)
                loss = model.loss(pred_label, hypo_label, weighted=True)
                loss.backward()
                optimizer.step()
                training_loss += (loss.item() * num_data)

            
            if ddp:
                local_train_loss = torch.tensor(training_loss, device=device)
                dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM) # Sum the loss across all GPUs
                training_loss = local_train_loss.item() / (len(train_loader.dataset))
            else:
                training_loss = training_loss / len(train_loader.dataset)

            model.eval()
            validating_loss = 0
            with torch.no_grad():
                for signal_data, hypo_label, glucose, cgm_idx in val_loader:
                    if args.data_type == 'eda':
                        phasic, tonic = signal_data
                        phasic, tonic, hypo_label, glucose = phasic.float().to(device), tonic.float().to(device), hypo_label.float().to(device), glucose.float().to(device)
                        pred_label = model(phasic, tonic)
                        num_data = phasic.shape[0]
                        if phasic.shape[0] == 1:
                            pred_label = pred_label.unsqueeze(0)
                    else:
                        signal_data, hypo_label, glucose = signal_data.float().to(device), hypo_label.float().to(device), glucose.float().to(device)
                        pred_label = model(signal_data)
                        num_data = signal_data.shape[0]
                        if signal_data.shape[0] == 1:
                            pred_label = pred_label.unsqueeze(0)
                    loss = model.loss(pred_label, hypo_label, weighted=True)
                    validating_loss += (loss.item() * num_data)

            if ddp:
                local_val_loss = torch.tensor(validating_loss, device=device)
                dist.all_reduce(local_val_loss, op=dist.ReduceOp.SUM)
                validating_loss = local_val_loss.item() / (len(val_loader.dataset))
            else:
                validating_loss = validating_loss / len(val_loader.dataset)

            if master_process:
                print_and_log(f"epoch: {epoch}| training_loss: {training_loss:.4f}| validating_loss: {validating_loss:.4f}| time: {time.time()-start_t:.2f}s")

                # Track the best model
                if validating_loss < best_loss:
                    best_loss = validating_loss
                    torch.save(model.state_dict(), ckpt_path)
                    saved_epoch = epoch
                    early_stopping = 0
                else:
                    early_stopping += 1

                scheduler.step(validating_loss)

                # Save the loss plot
                training_losses.append(training_loss)
                validating_losses.append(validating_loss)
                plt.figure(figsize=(10, 5))
                plt.plot(training_losses, label='train', color='blue')
                plt.plot(validating_losses, label='val', color='red')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Subject {} - Version {}'.format(args.subject, version))
                plt.tight_layout()
                plt.savefig(f"{out_ver_dir}/loss.png")
                plt.close()

            # Broadcast early stopping signal
            if ddp:
                early_stop_signal = torch.tensor(1 if early_stopping >= config['patience'] else 0, device=device)
                dist.broadcast(early_stop_signal, src=0)
                require_early_stop = early_stop_signal.item() == 1
                if require_early_stop:
                    print_and_log(f"Early stopping at epoch {epoch}")
                    break
            else:
                require_early_stop = early_stopping >= config['patience']
                if require_early_stop:
                    print_and_log(f"Early stopping at epoch {epoch}")
                    break

        if master_process:
            print_and_log(f"Saving model to {ckpt_path} at epoch {saved_epoch}")

    if ddp:
        destroy_process_group()