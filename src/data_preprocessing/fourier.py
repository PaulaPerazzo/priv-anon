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

    # class_labels = np.array([int(x[0]) for x in df_fft["class"]])

    return X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels

def fourier_transform(window_data, buffer_size=256, frequency_hz=50):
    fft_gx = np.fft.fft(window_data.iloc[:, 0])  # gyro x
    fft_gy = np.fft.fft(window_data.iloc[:, 1])  # gyro y
    fft_gz = np.fft.fft(window_data.iloc[:, 2])  # gyro z
    fft_ax = np.fft.fft(window_data.iloc[:, 3])  # acc x
    fft_ay = np.fft.fft(window_data.iloc[:, 4])  # acc y
    fft_az = np.fft.fft(window_data.iloc[:, 5])  # acc z
    freq = np.fft.fftfreq(buffer_size, d=1/frequency_hz)
    class_label = window_data.iloc[:, 6].mode()
    
    return freq, np.abs(fft_ax), np.abs(fft_ay), np.abs(fft_az), np.abs(fft_gx), np.abs(fft_gy), np.abs(fft_gz), class_label
