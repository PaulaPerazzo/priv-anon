import pandas as pd
import sys
import os
import numpy as np

from sklearn.metrics import confusion_matrix
from keras.models import load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.fourier import matrix_fourier_adjust, fourier_transform

def main():
    sensor_data = pd.read_csv("./data/motion_sense_test.csv", usecols=["rotationRate.x", "rotationRate.y", "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "act"])

    buffer_size = 256
    step_size = 128
    fft_data = []

    model_path = "./src/models/HAR.h5"
    model = load_model(model_path)

    for start in range(0, len(sensor_data) - buffer_size + 1, step_size):
        buffer_df = sensor_data.iloc[start:start + buffer_size]

        freq, fft_ax, fft_ay, fft_az, fft_gx, fft_gy, fft_gz, class_labels = fourier_transform(buffer_df)

        fft_features = {
            'freq': freq,
            'fft_acc_x': fft_ax,
            'fft_acc_y': fft_ay,
            'fft_acc_z': fft_az,
            'fft_gyro_x': fft_gx,
            'fft_gyro_y': fft_gy,
            'fft_gyro_z': fft_gz,
            'class': class_labels
        }

        fft_data.append(fft_features)
        df_fft = pd.DataFrame(fft_data)

        X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, _ = matrix_fourier_adjust(df_fft, testing=True)

        predictions = model.predict([X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z])
        
        predicted_class = predictions.argmax(axis=1)
        print("Activity predicted for window:", predicted_class)
        # print real class
        print("Actual class for window:", class_labels)

        fft_data.clear()

if __name__ == '__main__':
    main()
