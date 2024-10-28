import numpy as np
import pandas as pd

def clean_string(entry):
    cleaned_entry = str(entry).replace('\n', ' ').replace('[', '').replace(']', '').strip()
    
    return [float(x) for x in cleaned_entry.split()]

def matrix_fourier_adjust(df_fft, buffer_size=256):
    X_acc_x = np.array([clean_string(entry) for entry in df_fft['fft_acc_x']]).reshape(-1, buffer_size, 1)
    X_acc_y = np.array([clean_string(entry) for entry in df_fft['fft_acc_y']]).reshape(-1, buffer_size, 1)
    X_acc_z = np.array([clean_string(entry) for entry in df_fft['fft_acc_z']]).reshape(-1, buffer_size, 1)
    X_gyro_x = np.array([clean_string(entry) for entry in df_fft['fft_gyro_x']]).reshape(-1, buffer_size, 1)
    X_gyro_y = np.array([clean_string(entry) for entry in df_fft['fft_gyro_y']]).reshape(-1, buffer_size, 1)
    X_gyro_z = np.array([clean_string(entry) for entry in df_fft['fft_gyro_z']]).reshape(-1, buffer_size, 1)
    class_labels = np.array([clean_string(entry) for entry in df_fft['class']])

    return X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels
