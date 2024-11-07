import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.anonymization.filter import apply_filter

# Example usage
sensor_data = pd.read_csv("./data/motion_sense_train.csv", usecols=["rotationRate.x", "rotationRate.y", "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "act"])

buffer_size = 256  # Define window size

filtered_data = []

# Apply the filter to each window
for i in range(0, len(sensor_data), buffer_size):
    window = sensor_data.iloc[i:i + buffer_size]
    filtered_window = window.copy()

    # Apply the filter to each axis (assuming columns are labeled for sensor axes)
    filtered_window['rotationRate.x'] = apply_filter(window['rotationRate.x'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['rotationRate.y'] = apply_filter(window['rotationRate.y'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['rotationRate.z'] = apply_filter(window['rotationRate.z'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['userAcceleration.x'] = apply_filter(window['userAcceleration.x'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['userAcceleration.y'] = apply_filter(window['userAcceleration.y'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['userAcceleration.z'] = apply_filter(window['userAcceleration.z'], cutoff_low=0.3, cutoff_high=3)
    filtered_window['act'] = window['act'].mode()[0]  # Use the most common activity label in the window

    filtered_data.append(filtered_window)

# Concatenate all filtered windows
filtered_sensor_data = pd.concat(filtered_data, ignore_index=True)
filtered_sensor_data.to_csv("anonymized_sensor_data.csv", index=False)
