import pandas as pd
import os
import librosa
from pydub import AudioSegment

def process_audio_file(audio_file):
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_df = pd.DataFrame(mfccs).T
    return mfccs_df, len(audio_data)/sample_rate  # Return MFCC DataFrame and audio duration

def assign_timestamps_to_mfccs(mfccs_df, audio_duration):
    interval = audio_duration / len(mfccs_df)  # Interval in seconds
    timestamps = [pd.Timestamp(0)]  # Starting with timestamp 0

    # Calculate subsequent timestamps
    for _ in range(1, len(mfccs_df)):
        timestamps.append(timestamps[-1] + pd.Timedelta(seconds=interval))

    mfccs_df['TimeStamp'] = timestamps
    mfccs_df = mfccs_df.set_index('TimeStamp')
    return mfccs_df

# Process each audio file, calculate MFCCs, and assign timestamps
audio_dir = r"path_to_audio_directory"
for file in os.listdir(audio_dir):
    if file.endswith('.m4a'):
        audio = AudioSegment.from_file(os.path.join(audio_dir, file), format='m4a')
        audio.export(os.path.join(audio_dir, file), format='wav')
    if file.endswith('.wav'):
        mfccs_df, duration = process_audio_file(os.path.join(audio_dir, file))
        mfccs_with_timestamps = assign_timestamps_to_mfccs(mfccs_df, duration)
        # Save the MFCCs with timestamps
        mfccs_with_timestamps.to_csv(os.path.join(audio_dir, f'{os.path.splitext(file)[0]}_mfccs.csv'))
