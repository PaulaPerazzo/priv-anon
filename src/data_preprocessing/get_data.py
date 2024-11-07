from src.data_preprocessing.fourier import matrix_fourier_adjust
import pandas as pd

def get_data(data, testing=False):
    X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels = matrix_fourier_adjust(data, testing=testing)

    return X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels
