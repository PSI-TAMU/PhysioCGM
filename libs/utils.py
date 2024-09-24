import numpy as np
import pandas as pd

def filter_glucose_by_range(df, lower_bound=40, upper_bound=180, verbose=True):
    # Select only those within the proper range of glucose
    filtered_df = df[(df['glucose'] >= lower_bound) & (df['glucose'] <= upper_bound)]

    if verbose:
        print("Glucose Level Filtering (remove beats with glucose value outside 40-180 mg/dl ):")
        print(" - Original count: ", df.shape[0])
        print(" - Filtered count: ", filtered_df.shape[0])
        print(" - Availability ratio: {:.3f}".format(filtered_df.shape[0] / df.shape[0]))

    return filtered_df

def filter_noisy_data(df, HRConfidence_threshold=100, Noise_threshold=0.001, verbose=True):
    # Select only those within the proper range of HRConfidence and ECGNoise
    filtered_df = df[(df['HRConfidence'] >= HRConfidence_threshold) & (df['ECGNoise'] < Noise_threshold)]
    filtered_df = filtered_df.copy()
    filtered_df.loc[:, 'avg_HR'] = filtered_df['HR'].apply(lambda x: np.mean(x))
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])

    if verbose:
        print("Filter out noisy data (HRConfidence < 100 or ECGNoise >= 0.001):")
        print(" - Original: ", df.shape[0])
        print(" - Filtered: ", filtered_df.shape[0])
        print(" - Availability Ratio: {:.3f}".format(filtered_df.shape[0] / df.shape[0]))
    
    return filtered_df