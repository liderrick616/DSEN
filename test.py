# test.py

import numpy as np
import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from scipy.signal import resample

# Import your model and other necessary classes
from main import DSENModel  # Make sure main.py is structured properly for import

# Function to preprocess EEG data
def preprocess_eeg_data(eeg_data, target_time_len=3600, target_fs=200):
    num_channels, original_time_len = eeg_data.shape
    original_fs = 1000  # Update this to your actual sampling frequency

    # Step 1: Downsample to 200Hz if necessary
    if original_fs != target_fs:
        num_samples = int(eeg_data.shape[1] * (target_fs / original_fs))
        eeg_data = resample(eeg_data, num_samples, axis=1)

    # Step 2: Extract the first 'target_time_len' time steps
    if eeg_data.shape[1] >= target_time_len:
        eeg_data = eeg_data[:, :target_time_len]
    else:
        # If the data is shorter than target_time_len, pad with zeros
        padding = np.zeros((num_channels, target_time_len - eeg_data.shape[1]))
        eeg_data = np.hstack((eeg_data, padding))

    return eeg_data

# Function to load and preprocess EEG data from .mat files
def load_and_preprocess_eeg_data(data_dir, file_names):
    data = []
    labels = []

    for file_name in file_names:
        mat_file = os.path.join(data_dir, file_name)
        mat = loadmat(mat_file)
        # Access the EEG data
        eeg_data = mat['data']  # Adjust the key if your data is stored under a different key

        # Ensure data is of type float32
        eeg_data = eeg_data.astype(np.float32)

        # Transpose if necessary
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        # Preprocess the EEG data
        eeg_data = preprocess_eeg_data(eeg_data, target_time_len=3600, target_fs=200)

        # Extract subject ID from file name
        subject_id = int(file_name.split('_')[0][3:])

        # Assign labels based on subject IDs
        if subject_id in [81, 82, 80, 61, 62, 63, 64, 65, 66, 95, 96, 97, 98, 101, 102]:
            label = 1  # Friends
        else:
            label = 0  # Stranger

        data.append(eeg_data)
        labels.append(label)

    return data, labels

# Function to create pairs for testing
def create_test_pairs(data, labels):
    pairs = []
    pair_labels = []
    num_subjects = len(data)
    for i in range(num_subjects):
        for j in range(i + 1, num_subjects):
            pairs.append((data[i], data[j]))
            if labels[i] == labels[j] == 1:
                pair_labels.append(1)  # Friends
            else:
                pair_labels.append(0)  # Strangers
    return pairs, pair_labels

# Custom Dataset for testing
class EEGTestPairDataset(Dataset):
    def __init__(self, pairs, pair_labels):
        self.pairs = pairs
        self.labels = pair_labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.labels[idx]
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        label = torch.tensor(label).long()
        return x1, x2, label

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            x1, x2, label = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            # Forward pass
            logits, _, _ = model(x1, x2)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cpu')  # Change to 'cuda' if using GPU

    # Load the saved model
    model_path = 'dsen_model.pth'  # Ensure this points to your model file
    num_channels = 30  # Update based on your data
    time_len = 3600    # Update based on your data
    model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess test data
    data_dir = '/Users/derrick/PycharmProjects/DSEN'  # Update with your test data directory
    test_files = ['sub26_1_CSD.mat', 'sub28_9_CSD.mat', 'sub97_1_CSD.mat']  # Update with your test file names

    data, labels = load_and_preprocess_eeg_data(data_dir, test_files)

    # Check data shapes
    for i, eeg_data in enumerate(data):
        print(f"Subject {test_files[i]}: EEG data shape = {eeg_data.shape}")

    # Create test pairs
    pairs, pair_labels = create_test_pairs(data, labels)

    # Create test dataset and dataloader
    test_dataset = EEGTestPairDataset(pairs, pair_labels)
    batch_size = 16  # Adjust based on your data size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    accuracy, f1 = test_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

