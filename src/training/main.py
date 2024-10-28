import sys
import os
sys.path.append('/Users/paulaperazzo/Documents/pibic/datastream-anonymization')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from src.data_preprocessing.fourier import matrix_fourier_adjust
from src.training.branch import create_branch
import pandas as pd

# Load data
df = pd.read_csv("./data/fft_data.csv")

# Adjust data
X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z = matrix_fourier_adjust(df)

# parameters
input_shape = (256, 1) # shape of the input data
num_classes = 6 # number of classes

input_acc_x, branch_acc_x = create_branch(input_shape)
input_acc_y, branch_acc_y = create_branch(input_shape)
input_acc_z, branch_acc_z = create_branch(input_shape)
input_gyro_x, branch_gyro_x = create_branch(input_shape)
input_gyro_y, branch_gyro_y = create_branch(input_shape)
input_gyro_z, branch_gyro_z = create_branch(input_shape)

# concatenate branches
merged = Concatenate()([branch_acc_x, branch_acc_y, branch_acc_z, branch_gyro_x, branch_gyro_y, branch_gyro_z])

# dense layers for classification
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# final model
model = Model(inputs=[input_acc_x, input_acc_y, input_acc_z, input_gyro_x, input_gyro_y, input_gyro_z], outputs=output)

# optimize hyperparameters

# compile model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Treinamento do modelo
# # Ajuste X_train e y_train para serem as entradas e saídas formatadas
# # cada eixo (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) deve ser um array de forma (num_samples, 246, 1)
# history = model.fit([X_train_acc_x, X_train_acc_y, X_train_acc_z, X_train_gyro_x, X_train_gyro_y, X_train_gyro_z],
#                     y_train, epochs=20, batch_size=32, validation_split=0.2)

# # Avaliação do modelo no conjunto de teste
# test_loss, test_accuracy = model.evaluate([X_test_acc_x, X_test_acc_y, X_test_acc_z, X_test_gyro_x, X_test_gyro_y, X_test_gyro_z], y_test)
# print(f"Test Accuracy: {test_accuracy:.2f}")
