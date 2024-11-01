import pickle
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from src.data_preprocessing.fourier import matrix_fourier_adjust
import pandas as pd
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json
from src.training.mailer import send_email
from tensorflow.keras.utils import to_categorical

def get_data():
    df = pd.read_csv("processed_train_data_gender.csv")
    X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, gender_labels = matrix_fourier_adjust(df)
    gender_labels = to_categorical(gender_labels, num_classes=2)

    return X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, gender_labels

def create_model(trial):
    num_filters1 = trial.suggest_int("num_filters1", 16, 64, step=16)
    num_filters2 = trial.suggest_int("num_filters2", 32, 128, step=32)
    dense_units1 = trial.suggest_int("dense_units1", 64, 256, step=64)
    dense_units2 = trial.suggest_int("dense_units2", 32, 128, step=32)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    
    input_shape = (256, 1)
    # num_classes = 6 # for activity classification
    num_classes = 2 # for gender classification
    
    def create_branch_with_params(input_shape, num_filters1, num_filters2):
        input_layer = Input(shape=input_shape)
        x = Conv1D(num_filters1, 3, activation='relu')(input_layer)
        x = MaxPooling1D(2)(x)
        x = Conv1D(num_filters2, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)

        return input_layer, x
    
    input_acc_x, branch_acc_x = create_branch_with_params(input_shape, num_filters1, num_filters2)
    input_acc_y, branch_acc_y = create_branch_with_params(input_shape, num_filters1, num_filters2)
    input_acc_z, branch_acc_z = create_branch_with_params(input_shape, num_filters1, num_filters2)
    input_gyro_x, branch_gyro_x = create_branch_with_params(input_shape, num_filters1, num_filters2)
    input_gyro_y, branch_gyro_y = create_branch_with_params(input_shape, num_filters1, num_filters2)
    input_gyro_z, branch_gyro_z = create_branch_with_params(input_shape, num_filters1, num_filters2)
    
    merged = Concatenate()([branch_acc_x, branch_acc_y, branch_acc_z, branch_gyro_x, branch_gyro_y, branch_gyro_z])
    
    x = Dense(dense_units1, activation='relu')(merged)
    x = Dense(dense_units2, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[input_acc_x, input_acc_y, input_acc_z, input_gyro_x, input_gyro_y, input_gyro_z], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def objective(trial):
    X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, gender_labels = get_data()
    
    X_train_acc_x, X_val_acc_x, X_train_acc_y, X_val_acc_y, X_train_acc_z, X_val_acc_z, \
    X_train_gyro_x, X_val_gyro_x, X_train_gyro_y, X_val_gyro_y, X_train_gyro_z, X_val_gyro_z, \
    y_train, y_val = train_test_split(X_acc_x, X_acc_y, X_acc_z, X_gyro_x, X_gyro_y, X_gyro_z, gender_labels, test_size=0.2)

    model = create_model(trial)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(
        [X_train_acc_x, X_train_acc_y, X_train_acc_z, X_train_gyro_x, X_train_gyro_y, X_train_gyro_z],
        y_train,
        validation_data=([X_val_acc_x, X_val_acc_y, X_val_acc_z, X_val_gyro_x, X_val_gyro_y, X_val_gyro_z], y_val),
        epochs=20,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        callbacks=[early_stopping],
        verbose=0
    )

    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    save_losses(loss_values, val_loss_values)
    
    val_accuracy = max(history.history['val_accuracy'])

    return val_accuracy

def save_best_hyperparameters(study, filename="best_hyperparameters_gender.json"):
    best_params = study.best_params

    with open(filename, "w") as f:
        json.dump(best_params, f)

    print(f"Best hyperparameters saved to {filename}")

def save_losses(loss, val_loss, filename="losses_gender.json"):
    losses = {
        "loss": loss, 
        "val_loss": val_loss
    }

    with open(filename, "a") as f:
        json.dump(losses, f)
    
    print(f"Losses saved to {filename}")

# begginning time
start_time = time.time()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# end time
end_time = time.time()

# total time
total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")

with open("study_gender.pkl", "wb") as f:
    pickle.dump(study, f)

with open("log_time_gender.txt", "w") as f:
    f.write(f"Total time: {total_time:.2f} seconds")

save_best_hyperparameters(study)
print("Best hyperparameters:", study.best_params)

send_email("hyperparameter gender optimization")
