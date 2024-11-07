import sys
import os
import time
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from src.data_preprocessing.fourier import matrix_fourier_adjust
from src.training.branch import create_branch
import pandas as pd
import json
from src.training.metrics_logger import MetricsLogger
from tensorflow.keras.callbacks import EarlyStopping
from src.training.mailer import send_email

# Load data
df = pd.read_csv("processed_train_data_weight.csv")

# Adjust data
X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, class_labels = matrix_fourier_adjust(df)
# class_labels -= 1 # adjust labels

# parameters
input_shape = (256, 1) # shape of the input data
num_classes = 3 # number of classes

input_acc_x, branch_acc_x = create_branch(input_shape)
input_acc_y, branch_acc_y = create_branch(input_shape)
input_acc_z, branch_acc_z = create_branch(input_shape)
input_gyro_x, branch_gyro_x = create_branch(input_shape)
input_gyro_y, branch_gyro_y = create_branch(input_shape)
input_gyro_z, branch_gyro_z = create_branch(input_shape)

# concatenate branches
merged = Concatenate()([branch_acc_x, branch_acc_y, branch_acc_z, branch_gyro_x, branch_gyro_y, branch_gyro_z])

best_hyperparameters = "./src/training/hyperparams/best_hyperparameters_weight.json"
with open(best_hyperparameters, 'r') as f:
    best_hyperparameters = json.load(f)

# dense layers for classification
x = Dense(best_hyperparameters['dense_units1'], activation='relu')(merged)
x = Dense(best_hyperparameters['dense_units2'], activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# final model
model = Model(inputs=[input_acc_x, input_acc_y, input_acc_z, input_gyro_x, input_gyro_y, input_gyro_z], outputs=output)
model.compile(optimizer=Adam(learning_rate=best_hyperparameters['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model created")

y = tf.keras.utils.to_categorical(class_labels, num_classes=3)

# Split data train / val
X_train_acc_x, X_val_acc_x, X_train_acc_y, X_val_acc_y, X_train_acc_z, X_val_acc_z, \
X_train_gyro_x, X_val_gyro_x, X_train_gyro_y, X_val_gyro_y, X_train_gyro_z, X_val_gyro_z, \
y_train, y_val = train_test_split(X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, y, test_size=0.2)

metrics_logger = MetricsLogger(X_train=[X_train_acc_x, X_train_acc_y, X_train_acc_z, X_train_gyro_x, X_train_gyro_y, X_train_gyro_z], y_train=y_train)
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# initial time for training
initial_time = time.time()

# train model
history = model.fit(
    [X_train_acc_x, X_train_acc_y, X_train_acc_z, X_train_gyro_x, X_train_gyro_y, X_train_gyro_z], y_train, 
    epochs=50, 
    batch_size=best_hyperparameters['batch_size'], 
    validation_data=([X_val_acc_x, X_val_acc_y, X_val_acc_z, X_val_gyro_x, X_val_gyro_y, X_val_gyro_z], y_val),
    callbacks=[metrics_logger, early_stopping],
    verbose=1
)

# finish time for training
final_time = time.time()

# time spent for training
time_spent = final_time - initial_time

with open("log_time_training_weight.txt", "w") as f:
    f.write(str(time_spent))

print("Training finished")
print("Time spent for training: ")
print(final_time)

# Save model
model.save("weight_classifier.h5")

# Send email
send_email("training weight classifier model")
