import pandas as pd
import os
import numpy as np

# Directory paths
brainwave_dir = r"brainwave_data_directory"
heart_rate_dir = r"heart_rate_data_directory"
mfccs_dir = r"mfccs_data_directory"
output_dir = r"output_directory"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def standardize_timestamps(df, column_name='TimeStamp'):
    # Convert to datetime, assuming the format is correct and in UTC
    df[column_name] = pd.to_datetime(df[column_name], utc=True)
    return df

def interpolate_heart_rate(heart_rate_df):
    # Assuming the 'value' column holds the heart rate data
    heart_rate_df['value'] = heart_rate_df['value'].interpolate(method='linear')
    return heart_rate_df

def aggregate_mfccs(mfccs_df):
    # Resample and aggregate MFCC data by the mean over one-second intervals
    mfccs_aggregated = mfccs_df.resample('1S').mean()
    return mfccs_aggregated

def process_files(brainwave_file, heart_rate_file, mfccs_file, output_file):
    try:
        # Process brainwave data
        brainwave_df = pd.read_csv(brainwave_file)
        brainwave_df.columns.values[0] = 'TimeStamp'
        brainwave_df['TimeStamp'] = pd.to_datetime(brainwave_df['TimeStamp'], utc=True)
        brainwave_df.set_index('TimeStamp', inplace=True)

        # Process heart rate data
        heart_rate_df = pd.read_csv(heart_rate_file)
        heart_rate_df.columns.values[5] = 'startDate'
        heart_rate_df['startDate'] = pd.to_datetime(heart_rate_df['startDate'], utc=True)
        heart_rate_df.set_index('startDate', inplace=True)
        heart_rate_df = heart_rate_df[~heart_rate_df.index.duplicated(keep='first')]  # Handling duplicates
        heart_rate_df['value'] = heart_rate_df['value'].astype(float).interpolate(method='linear')

        # Process MFCC data
        mfccs_df = pd.read_csv(mfccs_file)
        mfccs_df.columns.values[0] = 'TimeStamp'
        mfccs_df['TimeStamp'] = pd.to_datetime(mfccs_df['TimeStamp'], utc=True)
        mfccs_df.set_index('TimeStamp', inplace=True)
        mfccs_df = mfccs_df[~mfccs_df.index.duplicated(keep='first')]  # Handling duplicates

        # Align and combine dataframes
        aligned_heart_rate = heart_rate_df.reindex(brainwave_df.index, method='nearest')
        aligned_mfccs = mfccs_df.reindex(brainwave_df.index, method='nearest')
        combined_df = pd.concat([brainwave_df, aligned_heart_rate, aligned_mfccs], axis=1)

        # Interpolate only numeric columns
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_cols] = combined_df[numeric_cols].interpolate(method='linear')

        combined_df.to_csv(output_file)
    except Exception as e:
        print(f"Error in processing files for session {output_file}: {e}")


# Iterate over the sessions and process
for session_id in range(1, 13):  # Replace with the actual number of sessions you have
    brainwave_file = os.path.join(brainwave_dir, f'brainwave_session{session_id}.csv')
    heart_rate_file = os.path.join(heart_rate_dir, f'heart_rate_session{session_id}.csv')
    mfccs_file = os.path.join(mfccs_dir, f'sound_session{session_id}_mfccs.csv')
    output_file = os.path.join(output_dir, f'combined_session{session_id}.csv')

    # Ensure all files exist before processing
    if os.path.exists(brainwave_file) and os.path.exists(heart_rate_file) and os.path.exists(mfccs_file):
        print(f"Processing session {session_id}")
        process_files(brainwave_file, heart_rate_file, mfccs_file, output_file)
        print(f"Processed session {session_id}")
    else:
        print(f"Data for session {session_id} is incomplete. Skipping.")
