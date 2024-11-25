import os
import tqdm
import glob
import pickle
import datetime
import argparse
import pandas as pd
from libs.combine_data import verify_folder, Accel_read_and_combine, Breathing_read_and_combine, ECG_read_and_combine, summary_read_and_combine, process_glucose, combine_e4

def populate_sensor_data(sensor_data, selected_df, columns):
    for col in columns:
        sensor_data[col] = selected_df[col].values if col in selected_df else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine ECG and summary files')
    parser.add_argument('--subject_id', type=str, help='subject id') # example: c1s01
    parser.add_argument('--data_path', default='./dataset/raw', type=str, help='path to folder containing raw data')
    parser.add_argument('--out_folder', default='./dataset/processed', type=str, help='path to folder to save processed data')
    args = parser.parse_args()

    subject_id = args.subject_id
    tmp_dir = os.path.join('./dataset/raw_combined', subject_id)
    raw_dir = os.path.join(args.data_path, subject_id)
    out_dir = os.path.join(args.out_folder, subject_id)

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)


    zephyr_dir = os.path.join(raw_dir, 'zephyr')

    valid_folders = []
    # Iterate over only directories in the folder
    folders = glob.glob(os.path.join(zephyr_dir, '*'))
    for folder in folders:
        if verify_folder(folder):
            valid_folders.append(folder)
    valid_folders.sort()
    print("Found {} valid folders".format(len(valid_folders)))

    # print("===========================================")
    # print("Combine Accel files")
    # accel_combined_df = Accel_read_and_combine(valid_folders)
    # accel_combined_df["Time"] = pd.to_datetime(accel_combined_df["Time"], format='%d/%m/%Y %H:%M:%S.%f')
    # print(accel_combined_df.head())
    # print(accel_combined_df.shape)
    # accel_combined_df.to_pickle(os.path.join(tmp_dir, '{}_Accel.pkl'.format(subject_id)))

    # print("===========================================")
    # print("Combine Breathing files")
    # breathing_combined_df = Breathing_read_and_combine(valid_folders)
    # breathing_combined_df["Time"] = pd.to_datetime(breathing_combined_df["Time"], format='%d/%m/%Y %H:%M:%S.%f')
    # print(breathing_combined_df.head())
    # print(breathing_combined_df.shape)
    # breathing_combined_df.to_pickle(os.path.join(tmp_dir, '{}_Breathing.pkl'.format(subject_id)))

    # print("===========================================")
    # print("Combine Summary files")
    # summary_combined_df = summary_read_and_combine(valid_folders)
    # summary_combined_df["Time"] = pd.to_datetime(summary_combined_df["Time"], format='%d/%m/%Y %H:%M:%S.%f')
    # print(summary_combined_df.head())
    # print(summary_combined_df.shape)
    # summary_combined_df.to_pickle(os.path.join(tmp_dir, '{}_Summary.pkl'.format(subject_id)))

    # print("===========================================")
    # print("Combine ECG files")
    # ecg_combined_df = ECG_read_and_combine(valid_folders)
    # ecg_combined_df["Time"] = pd.to_datetime(ecg_combined_df["Time"], format='%d/%m/%Y %H:%M:%S.%f')
    # print(ecg_combined_df.head())
    # print(ecg_combined_df.shape)
    # print("OK")

    # print("===========================================")
    # print("Combine summary files")
    # summary_combined_df = summary_read_and_combine(valid_folders)
    # summary_combined_df["Time"] = pd.to_datetime(summary_combined_df["Time"], format='%d/%m/%Y %H:%M:%S.%f')
    # print(summary_combined_df.head())
    # print(summary_combined_df.shape)
    # print("OK")

    # print("===========================================")
    # print("Processing glucose file")
    # glucose_path = os.path.join(raw_dir, 'cgm.csv')
    # glucose_df = process_glucose(glucose_path)
    # glucose_df["Timestamp"] = pd.to_datetime(glucose_df["Timestamp"], format='%d/%m/%Y %H:%M:%S.%f')
    # glucose_df['Time'] = glucose_df['Timestamp'].copy()
    # print(glucose_df.head())
    # print(glucose_df.shape)
    # print("OK")

    # print("===========================================")
    # print("Merging ECG, summary, and glucose files")
    # hr_df = summary_combined_df[["Time", "HR", "HRConfidence", "ECGNoise"]]
    # combined_df = pd.merge_asof(ecg_combined_df.sort_values('Time'), hr_df.sort_values('Time'), on='Time', direction='nearest', tolerance=pd.Timedelta('1s'))
    # final_df = pd.merge_asof(combined_df.sort_values('Time'), glucose_df.sort_values('Time'), on='Time', tolerance=pd.Timedelta('330s'), direction='forward', allow_exact_matches=True)
    # print(final_df.head())
    # print(final_df.shape)
    
    # print("===========================================")
    # print("Rows with missing values:")
    # print(final_df[final_df.isnull().any(axis=1)])
    # print("Dropping rows with missing values ...")
    # raw_final_df = final_df.dropna()

    # print("===========================================")
    # print("Saving the combined file")
    # raw_final_df.to_pickle(os.path.join(tmp_dir, '{}.pkl'.format(subject_id)))
    # print("Done")

    # print("===========================================")
    # print("Processing E4 data")
    # e4_dir = os.path.join(raw_dir, 'e4')
    # combine_e4(e4_dir, tmp_dir)
    # print("Done")

    print("===========================================")
    print("Start processing the combined file")
    df = pd.read_pickle(os.path.join(tmp_dir, '{}.pkl'.format(subject_id)))
    accel_df = pd.read_pickle(os.path.join(tmp_dir, '{}_Accel.pkl'.format(subject_id)))
    breathing_df = pd.read_pickle(os.path.join(tmp_dir, '{}_Breathing.pkl'.format(subject_id)))
    summary_df = pd.read_pickle(os.path.join(tmp_dir, '{}_Summary.pkl'.format(subject_id)))

    e4_acc_df = pd.read_pickle(os.path.join(tmp_dir, 'e4_ACC.pkl'))
    e4_hr_df = pd.read_pickle(os.path.join(tmp_dir, 'e4_HR.pkl'))
    e4_bvp_df = pd.read_pickle(os.path.join(tmp_dir, 'e4_BVP.pkl'))
    e4_eda_df = pd.read_pickle(os.path.join(tmp_dir, 'e4_EDA.pkl'))
    e4_temp_df = pd.read_pickle(os.path.join(tmp_dir, 'e4_TEMP.pkl'))

    cgm_df = df.drop_duplicates(subset='Index', keep='first')

    for i in tqdm.tqdm(range(cgm_df.shape[0])):
        # if os.path.exists(f'{out_dir}/{i}.pkl'):
        #     continue

        cgm_row = cgm_df.iloc[i]
        _data = {
            'Index': cgm_row['Index'],
            'Timestamp': cgm_row['Timestamp'],
            'glucose': cgm_row['glucose'],
            'zephyr': {
                'Accel': {'Time': None, "Vertical": None, "Lateral": None, "Sagittal": None},
                'Breathing': {'Time': None, 'BreathingWaveform': None},
                'ECG': {'Time': None, 'EcgWaveform': None},
                'Summary': {'Time': None, 'HR': None, 'BR': None, 'Posture': None, 'Activity': None, 'HRConfidence': None, 'ECGNoise': None},
            },
            'e4': {
                'ACC': {'Time': None, 'x': None, 'y': None, 'z': None},
                'HR': {'Time': None, 'HR': None},
                'BVP': {'Time': None, 'BVP': None},
                'EDA': {'Time': None, 'EDA': None},
                'TEMP': {'Time': None, 'TEMP': None},
            },
        }
        
        
        selected_df = df[df['Index'] == cgm_row['Index']]
        start_t, end_t = selected_df['Time'].values[0], selected_df['Time'].values[-1]

        # Zephyr sensor data
        accel_selected_df = accel_df[(accel_df['Time'] >= start_t) & (accel_df['Time'] <= end_t)]
        breathing_selected_df = breathing_df[(breathing_df['Time'] >= start_t) & (breathing_df['Time'] <= end_t)]
        summary_selected_df = summary_df[(summary_df['Time'] >= start_t) & (summary_df['Time'] <= end_t)]
        
        populate_sensor_data(_data['zephyr']['ECG'], selected_df, ['Time', 'EcgWaveform'])
        populate_sensor_data(_data['zephyr']['Accel'], accel_selected_df, ['Time', 'Vertical', 'Lateral', 'Sagittal'])
        populate_sensor_data(_data['zephyr']['Breathing'], breathing_selected_df, ['Time', 'BreathingWaveform'])
        populate_sensor_data(_data['zephyr']['Summary'], summary_selected_df, ['Time', 'HR', 'BR', 'Posture', 'Activity', 'HRConfidence', 'ECGNoise'])
        # Empatica E4 sensor data
        e4_acc_selected_df = e4_acc_df[(e4_acc_df['Time'] >= start_t) & (e4_acc_df['Time'] <= end_t)]
        e4_bvp_selected_df = e4_bvp_df[(e4_bvp_df['Time'] >= start_t) & (e4_bvp_df['Time'] <= end_t)]
        e4_eda_selected_df = e4_eda_df[(e4_eda_df['Time'] >= start_t) & (e4_eda_df['Time'] <= end_t)]
        e4_hr_selected_df = e4_hr_df[(e4_hr_df['Time'] >= start_t) & (e4_hr_df['Time'] <= end_t)]
        e4_temp_selected_df = e4_temp_df[(e4_temp_df['Time'] >= start_t) & (e4_temp_df['Time'] <= end_t)]

        populate_sensor_data(_data['e4']['ACC'], e4_acc_selected_df, ['Time', 'x', 'y', 'z'])
        populate_sensor_data(_data['e4']['HR'], e4_hr_selected_df, ['Time', 'HR'])
        populate_sensor_data(_data['e4']['BVP'], e4_bvp_selected_df, ['Time', 'BVP'])
        populate_sensor_data(_data['e4']['EDA'], e4_eda_selected_df, ['Time', 'EDA'])
        populate_sensor_data(_data['e4']['TEMP'], e4_temp_selected_df, ['Time', 'TEMP'])

        with open(f'{out_dir}/{i}.pkl', 'wb') as f:
            pickle.dump(_data, f)
        
    print("Done")
        