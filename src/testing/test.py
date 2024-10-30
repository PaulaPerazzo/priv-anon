import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.fourier import matrix_fourier_adjust, fourier_transform
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np

def main():
    df_fft = pd.read_csv("processed_test_data.csv")

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

    # evaluate model with confusion matrix
    y_pred = [int(x) for x in predictions.argmax(axis=1)]
    y_true = [int(x) for x in class_labels]

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # cm to percentage matrix with 4 decimal places
    cm = np.round(cm / cm.sum(axis=1)[:, np.newaxis], 3)
    print(cm)


if __name__ == '__main__':
    main()
