import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.fourier import matrix_fourier_adjust, fourier_transform
from keras.models import load_model

def main():
    sensor_data = pd.read_csv("./data/motion_sense_test.csv", usecols=["rotationRate.x", "rotationRate.y", "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "act"])

    buffer_size = 256
    num_windows = (len(sensor_data) - buffer_size) + 1
    fft_data = []

    while True:
        buffer = []

        for line in sensor_data.values:
            buffer.append(line.tolist())
            
            if len(buffer) == buffer_size:
                break
        
        buffer_df = pd.DataFrame(buffer)
        
        # Apply Fourier Transform to buffer
        freq, fft_ax, fft_ay, fft_az, fft_gx, fft_gy, fft_gz, class_labels = fourier_transform(buffer_df)
        print(freq.shape)
        print(fft_ax.shape)
        print(fft_ay.shape)
        print(fft_az.shape)
        print(fft_gx.shape)
        print(fft_gy.shape)
        print(fft_gz.shape)
        print(class_labels.shape)
        print(class_labels)

        fft_data = []
        
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

        X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels = matrix_fourier_adjust(df_fft)
        print(X_acc_x.shape)
        print(X_acc_y.shape)
        print(X_acc_z.shape)
        print(X_gyro_x.shape)
        print(X_gyro_y.shape)
        print(X_gyro_z.shape)
        print(class_labels.shape)

        # predict class labels
        model_path = "./src/models/HAR.h5"
        model = load_model(model_path)

        predictions = model.predict([X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z])
        print(predictions)

        if buffer_size == 256:
            buffer = []
            break


if __name__ == '__main__':
    main()
