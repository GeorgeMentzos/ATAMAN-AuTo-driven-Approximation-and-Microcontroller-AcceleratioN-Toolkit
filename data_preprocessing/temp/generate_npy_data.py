import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Create directory if it doesn't exist
output_dir = '/home/mengeo00/Desktop/ATAMAN-AuTo-driven-Approximation-and-Microcontroller-AcceleratioN-Toolkit/data_preprocessing/cifar_data_npy'
os.makedirs(output_dir, exist_ok=True)

# Save test data and labels to .npy files in the new directory
np.save(os.path.join(output_dir, 'cifar_test_data.npy'), x_test)
np.save(os.path.join(output_dir, 'cifar_test_labels.npy'), y_test)

print("CIFAR-10 test data and labels have been saved to 'cifar_test_data.npy' and 'cifar_test_labels.npy'")