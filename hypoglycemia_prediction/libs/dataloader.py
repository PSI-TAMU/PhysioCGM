import torch
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

class SeNSEDataset(Dataset):
    def __init__(self, df, metadata, data_type, verbose=True):
        self.data_type = data_type
        self.data = df[df['CGM_idx'].isin(metadata)].reset_index(drop=True)
        self.df_glucose = self.data.drop_duplicates(subset='CGM_idx', keep='first')

        self.hypo_threshold = 70
        self.num_hypo = len(self.data[self.data['glucose'] < self.hypo_threshold])
        self.num_normal = len(self.data[self.data['glucose'] >= self.hypo_threshold])
        self.normal_hypo_ratio = self.num_normal/self.num_hypo

        self.num_cgm_hypo = len(self.df_glucose[self.df_glucose['glucose'] < self.hypo_threshold])
        self.num_cgm_normal = len(self.df_glucose[self.df_glucose['glucose'] >= self.hypo_threshold])
        self.cgm_normal_hypo_ratio = self.num_cgm_normal/self.num_cgm_hypo
        if verbose:
            print("Dataset info:")
            print(" - {} data: {} normal, {} hypo, ratio {:.2f}".format(data_type, self.num_normal, self.num_hypo, self.normal_hypo_ratio))
            print(" - CGM data: {} normal, {} hypo, ratio {:.2f}".format(self.num_cgm_normal, self.num_cgm_hypo, self.cgm_normal_hypo_ratio))

    def stratified_sampling(self, batch_size):
        df_hypo = self.data[self.data['glucose'] < self.hypo_threshold]
        df_normal = self.data[self.data['glucose'] >= self.hypo_threshold]

        # shuffle the whole dataset
        df_hypo = df_hypo.sample(frac=1).reset_index(drop=True)
        df_normal = df_normal.sample(frac=1).reset_index(drop=True)

        stratified_data = []
        num_hypo_in_batch = int(batch_size // (1 + self.normal_hypo_ratio))
        num_normal_in_batch = batch_size - num_hypo_in_batch

        hypo_idx, normal_idx = 0, 0

        while hypo_idx < len(df_hypo) and normal_idx < len(df_normal):
            batch_hypo = df_hypo.iloc[hypo_idx:hypo_idx + num_hypo_in_batch]
            batch_normal = df_normal.iloc[normal_idx:normal_idx + num_normal_in_batch]

            # Ensure the batch is not empty
            if batch_hypo.empty or batch_normal.empty:
                break
            
            hypo_idx += len(batch_hypo)
            normal_idx += len(batch_normal)

            batch = pd.concat([batch_hypo, batch_normal]).reset_index(drop=True)
            batch = batch.sample(frac=1).reset_index(drop=True)  # Shuffle the batch

            stratified_data.append(batch)
        
        # Concatenate all the batches
        self.data = pd.concat(stratified_data).reset_index(drop=True)
    
    def __getitem__(self, idx):
        row_data = self.data.iloc[idx]
        if self.data_type == "ecg":
            data = row_data["ecg"]
        elif self.data_type == "ppg":
            data = row_data["ppg"]
        elif self.data_type == "eda":
            phasic = row_data["phasic"]
            tonic = row_data["tonic"]
            data = (phasic, tonic)
        glucose = row_data["glucose"]
        cgm_idx = row_data["CGM_idx"]
        hypo_label = 1 if glucose < self.hypo_threshold else 0
        return data, hypo_label, glucose, cgm_idx

    def __len__(self):
        return len(self.data)

class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, num_replicas=None, rank=None):
        self.data_source = data_source  # Store the dataset as an instance attribute
        self.batch_size = batch_size
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()

        # Create pre-organized balanced batches
        self._create_balanced_batches()

    def _create_balanced_batches(self):
        # Separate data into 'hypo' and 'normal' groups based on some threshold
        df_hypo = self.data_source.data[self.data_source.data['glucose'] < self.data_source.hypo_threshold].index.tolist()
        df_normal = self.data_source.data[self.data_source.data['glucose'] >= self.data_source.hypo_threshold].index.tolist()

        # Shuffle within each group
        np.random.shuffle(df_hypo)
        np.random.shuffle(df_normal)

        num_hypo_in_batch = int(self.batch_size // (1 + self.data_source.normal_hypo_ratio))
        num_normal_in_batch = self.batch_size - num_hypo_in_batch

        self.batches = []
        hypo_idx, normal_idx = 0, 0

        # Create balanced batches with a specific ratio
        while hypo_idx < len(df_hypo) and normal_idx < len(df_normal):
            batch_hypo = df_hypo[hypo_idx:hypo_idx + num_hypo_in_batch]
            batch_normal = df_normal[normal_idx:normal_idx + num_normal_in_batch]

            # Ensure the batch is not empty
            if not batch_hypo or not batch_normal:
                break

            hypo_idx += len(batch_hypo)
            normal_idx += len(batch_normal)

            # Combine and shuffle within the batch
            batch = batch_hypo + batch_normal
            np.random.shuffle(batch)
            self.batches.append(batch)

        # Partition the batches among replicas for DDP
        self.partitioned_batches = [self.batches[i::self.num_replicas] for i in range(self.num_replicas)]

    def __iter__(self):
        # Return an iterator for the batches assigned to the current process (rank)
        return iter([idx for batch in self.partitioned_batches[self.rank] for idx in batch])

    def set_epoch(self, epoch):
        # Reshuffle and repartition the data for a new epoch
        self._create_balanced_batches()

    def __len__(self):
        # Return the number of samples for this replica
        return len(self.partitioned_batches[self.rank]) * self.batch_size

class MultiModalDataset(Dataset):
    def __init__(self, ecg_df, ppg_df, eda_df, temp_df, CGM_indices, temp_norm=None):
        self.ecg_df = ecg_df
        self.ppg_df = ppg_df
        self.eda_df = eda_df
        self.temp_df = temp_df
        self.temp_norm = temp_norm
        self.CGM_indices = CGM_indices

    def __len__(self):
        return len(self.CGM_indices)
    
    def __getitem__(self, idx):
        CGM_idx = self.CGM_indices[idx]
        ecg_df = self.ecg_df[self.ecg_df['CGM_idx'] == CGM_idx].reset_index(drop=True)
        ppg_df = self.ppg_df[self.ppg_df['CGM_idx'] == CGM_idx].reset_index(drop=True)
        eda_df = self.eda_df[self.eda_df['CGM_idx'] == CGM_idx].reset_index(drop=True)
        temp_df = self.temp_df[self.temp_df['CGM_idx'] == CGM_idx].reset_index(drop=True)

        ecg_df = ecg_df.sort_values(by='start_t').reset_index(drop=True)
        ppg_df = ppg_df.sort_values(by='start_t').reset_index(drop=True)
        eda_df = eda_df.sort_values(by='start_t').reset_index(drop=True)

        ecg_data = np.stack(ecg_df['ecg'].values, axis=0)
        ppg_data = np.stack(ppg_df['ppg'].values, axis=0)

        phasic = np.stack(eda_df["phasic"].values, axis=0)[0]
        tonic = np.stack(eda_df["tonic"].values, axis=0)[0]
        eda_data = np.stack([phasic, tonic], axis=0)

        temp_data = np.stack(temp_df['temp'].values, axis=0)
        if self.temp_norm is not None:
            temp_data = (temp_data - self.temp_norm[0]) / self.temp_norm[1] # Normalize temperature
        
        timestamp = ecg_df['Timestamp'].values[0]
        glucose = ecg_df['glucose'].values[0]
        hypo_label = 1 if glucose < 70 else 0

        return ecg_data, ppg_data, eda_data, temp_data, hypo_label, (glucose, CGM_idx)

class MultiModalDataLoader:
    def __init__(self, ecg_data, ppg_data, eda_data, temp_data, metadata, verbose=True):
        super().__init__()
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        self.eda_data = eda_data
        self.temp_data = temp_data
        self.temp_mean = self.temp_data[self.temp_data['CGM_idx'].isin(metadata['train'])]['temp'].mean()
        self.temp_std = self.temp_data[self.temp_data['CGM_idx'].isin(metadata['train'])]['temp'].std()
        self.metadata = metadata

        if verbose:
            print(f"Temperature mean: {self.temp_mean:.2f}, std: {self.temp_std:.2f}")

    def get_loader(self, type, batch_size=32, shuffle=False):
        CGM_indices = self.metadata[type]
        ecg_data = self.ecg_data[self.ecg_data['CGM_idx'].isin(CGM_indices)].reset_index(drop=True)
        ppg_data = self.ppg_data[self.ppg_data['CGM_idx'].isin(CGM_indices)].reset_index(drop=True)
        eda_data = self.eda_data[self.eda_data['CGM_idx'].isin(CGM_indices)].reset_index(drop=True)
        temp_data = self.temp_data[self.temp_data['CGM_idx'].isin(CGM_indices)].reset_index(drop=True)
        dataset = MultiModalDataset(ecg_data, ppg_data, eda_data, temp_data, CGM_indices, temp_norm=(self.temp_mean, self.temp_std))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
