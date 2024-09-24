import os
import time
import glob
import tqdm
import pandas as pd
from datetime import datetime, timedelta

# Attempt to convert it to a datetime object
def is_valid_timestamp(timestamp_string):
    try:
        # Define the format based on the given string
        format_string = "%Y_%m_%d-%H_%M_%S"
        converted_time = datetime.strptime(timestamp_string, format_string)
        return True, converted_time
    except ValueError:
        return False, None

def verify_folder(folder_path):
    # Check if folder exists
    if not os.path.isdir(folder_path):
        return False

    folder_name = os.path.basename(folder_path)
    # Check if folder name is a valid timestamp
    if not is_valid_timestamp(folder_name)[0]:
        return False
    
    # Check if folder contains ECG and summary files
    file_path = glob.glob(os.path.join(folder_path, '*_ECG.csv'))
    if len(file_path) != 1:
        print("Skipping folder: {} (Doesn't have _ECG.csv)".format(folder_name))
        return False
    
    file_path = glob.glob(os.path.join(folder_path, '*_SummaryEnhanced.csv'))
    if len(file_path) != 1:
        print("Skipping folder: {} (Doesn't have _SummaryEnhanced.csv)".format(folder_name))
        return False

    return True

def Accel_read_and_combine(valid_folders):
    accel_combined_df = pd.DataFrame()
    for folder_path in tqdm.tqdm(valid_folders):
        if len(glob.glob(os.path.join(folder_path, '*_Accel.csv'))) == 0:
            print("Skipping folder: {} (Doesn't have _Accel.csv)".format(folder_path))
            continue
        file_path = glob.glob(os.path.join(folder_path, '*_Accel.csv'))[0]
        df = pd.read_csv(file_path)
        accel_combined_df = pd.concat([accel_combined_df, df])
    return accel_combined_df

def Breathing_read_and_combine(valid_folders):
    breathing_combined_df = pd.DataFrame()
    for folder_path in tqdm.tqdm(valid_folders):
        if len(glob.glob(os.path.join(folder_path, '*_Breathing.csv'))) == 0:
            print("Skipping folder: {} (Doesn't have _Breathing.csv)".format(folder_path))
            continue
        file_path = glob.glob(os.path.join(folder_path, '*_Breathing.csv'))[0]
        df = pd.read_csv(file_path)
        breathing_combined_df = pd.concat([breathing_combined_df, df])
    return breathing_combined_df


# combine ECG files
def ECG_read_and_combine(valid_folders):
    ecg_combined_df = pd.DataFrame()
    for folder_path in tqdm.tqdm(valid_folders):
        file_path = glob.glob(os.path.join(folder_path, '*_ECG.csv'))[0]
        df = pd.read_csv(file_path)        
        ecg_combined_df = pd.concat([ecg_combined_df, df])
    return ecg_combined_df

# combine summary files
def summary_read_and_combine(valid_folders):
    summary_combined_df = pd.DataFrame()

    for folder_path in tqdm.tqdm(valid_folders):
        file_path = glob.glob(os.path.join(folder_path, '*_SummaryEnhanced.csv'))[0]
        df = pd.read_csv(file_path)
        summary_combined_df = pd.concat([summary_combined_df, df])
    
    return summary_combined_df

def process_glucose(glucose_path):
    glucose_df = pd.read_csv(glucose_path,delimiter=',')
    glucose_df = glucose_df[glucose_df['Event Type'] == 'EGV'].reset_index(drop = True) 

    if glucose_df['Glucose Value (mg/dL)'].dtype == 'O':
        glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('Low','40')
    if glucose_df['Glucose Value (mg/dL)'].dtype == 'O':
        glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('High','400')

    glucose_df = glucose_df[['Timestamp (YYYY-MM-DDThh:mm:ss)','Glucose Value (mg/dL)', 'Index']]
    glucose_df.columns = ['Timestamp','glucose','Index']
    try:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%dT%H:%M:%S')
    except:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%d %H:%M:%S')
    glucose_df['glucose'] = glucose_df['glucose'].astype(float)
    glucose_df = glucose_df.sort_values('Timestamp').reset_index(drop = True)

    return glucose_df


### For empatica data
def combine_e4(data_root, out_dir):
    def process_1(data_root, feat):
        data_path = os.path.join(data_root, f'{feat}.csv')
        df = pd.read_csv(data_path,header = None)
        c = df.loc[0][0]
        start_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c))
        sampling_rate = df.loc[1][0]
        data = df.iloc[2:].to_numpy()

        new_data = {
            'Time': [],
            feat: []
        }
        for i in range(len(data)):
            new_data['Time'].append(datetime.strptime(start_t, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=i/sampling_rate))
            new_data[feat].append(data[i][0])

        new_df = pd.DataFrame(new_data)
        return new_df

    def process_hr(data_root):
        data_path = os.path.join(data_root, 'HR.csv')
        df = pd.read_csv(data_path,header = None)
        c = df.loc[0][0]
        start_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c))
        sampling_rate = df.loc[1][0]
        data = df.iloc[2:].to_numpy()

        new_data = {
            'Time': [],
            'HR': []
        }
        for i in range(len(data)):
            new_data['Time'].append(datetime.strptime(start_t, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=i/sampling_rate))
            new_data['HR'].append(data[i][0])

        new_df = pd.DataFrame(new_data)
        return new_df

    def process_acc(data_root):
        data_path = os.path.join(data_root, 'ACC.csv')
        df = pd.read_csv(data_path,header = None)

        c = df.loc[0][0]
        start_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c))
        sampling_rate = df.loc[1][0]
        data = df.iloc[2:].to_numpy()

        new_data = {
            'Time': [],
            'x': [],
            'y': [],
            'z': []
        }
        for i in range(len(data)):
            new_data['Time'].append(datetime.strptime(start_t, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=i/sampling_rate))
            new_data['x'].append(data[i][0])
            new_data['y'].append(data[i][1])
            new_data['z'].append(data[i][2])

        new_df = pd.DataFrame(new_data)
        return new_df

    valid_folders = [ d for d in sorted(glob.glob(os.path.join(data_root, '*'))) if os.path.isdir(d)]
    for feat in ['EDA', 'TEMP', 'BVP']:
        new_df = []
        for d in tqdm.tqdm(valid_folders):
            _new_df = process_1(d, feat)
            new_df.append(_new_df)
        new_df = pd.concat(new_df)
        new_df = new_df.sort_values(by='Time')
        new_df = new_df.reset_index(drop=True)
        out_path = os.path.join(out_dir, f'e4_{feat}.pkl')
        new_df.to_pickle(out_path)
    
    # process HR
    new_df = []
    for d in tqdm.tqdm(valid_folders):
        _new_df = process_hr(d)
        new_df.append(_new_df)
    new_df = pd.concat(new_df)
    new_df = new_df.sort_values(by='Time')
    new_df = new_df.reset_index(drop=True)
    out_path = os.path.join(out_dir, 'e4_HR.pkl')
    new_df.to_pickle(out_path)

    # process ACC
    new_df = []
    for d in tqdm.tqdm(valid_folders):
        _new_df = process_acc(d)
        new_df.append(_new_df)

    new_df = pd.concat(new_df)
    new_df = new_df.sort_values(by='Time')
    new_df = new_df.reset_index(drop=True)
    out_path = os.path.join(out_dir, 'e4_ACC.pkl')
    new_df.to_pickle(out_path)