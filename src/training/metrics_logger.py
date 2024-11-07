import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json
import tensorflow as tf
import numpy as np

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train):
        super(MetricsLogger, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.epoch_metrics = {
            "epoch": [],
            "accuracy": [],
            "loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc": [],
            "learning_rate": [],
            "epoch_time": []
        }

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {
            "epoch": [],
            "accuracy": [],
            "loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc": [],
            "learning_rate": [],
            "epoch_time": []
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        y_pred = np.argmax(self.model.predict(self.X_train), axis=1)
        y_true = np.argmax(self.y_train, axis=1)

        # y_pred_proba = self.model.predict(self.X_train)[:, 1]
        # y_true_binary = np.argmax(self.y_train, axis=1)
        
        # Calculate additional metrics
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        auc = roc_auc_score(y_true, self.model.predict(self.X_train), average="weighted", multi_class="ovr")
        # auc = roc_auc_score(y_true_binary, y_pred_proba)
        learning_rate = self.model.optimizer.learning_rate.numpy()

        # Store metrics
        self.epoch_metrics["epoch"].append(epoch + 1)
        self.epoch_metrics["accuracy"].append(logs.get("accuracy"))
        self.epoch_metrics["loss"].append(logs.get("loss"))
        self.epoch_metrics["val_loss"].append(logs.get("val_loss"))
        self.epoch_metrics["precision"].append(precision)
        self.epoch_metrics["recall"].append(recall)
        self.epoch_metrics["f1_score"].append(f1)
        self.epoch_metrics["auc"].append(auc)
        self.epoch_metrics["learning_rate"].append(learning_rate)
        self.epoch_metrics["epoch_time"].append(epoch_time)

    def on_train_end(self, logs=None):
        epoch_metrics_serializable = {
            key: [float(value) if isinstance(value, (np.float32, np.float64)) else value for value in values]
            for key, values in self.epoch_metrics.items()
        }

        with open("training_metrics_act_anon.json", "w") as f:
            json.dump(epoch_metrics_serializable, f)

        print("Training metrics saved to training_metrics_weight.json")
