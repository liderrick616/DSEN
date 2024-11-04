# test.py

import numpy as np
import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F
from scipy.signal import resample

# Import your model and other necessary classes
from main import DSENModel  # Ensure main.py is structured properly for import

friend_ids = [61, 62, 63, 64, 65, 66, 80, 81, 82, 95, 96, 97, 98, 101, 102]
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
    subject_ids = []

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
        subject_ids.append(subject_id)

        # Assign labels based on subject IDs for evaluation only
        if subject_id in friend_ids:
            label = 1  # Friend
        else:
            label = 0  # Stranger

        data.append(eeg_data)
        labels.append(label)

    return data, labels, subject_ids

# Function to create pairs for testing
def create_test_pairs(data):
    pairs = []
    indices = []
    num_subjects = len(data)
    for i in range(num_subjects):
        for j in range(i + 1, num_subjects):
            pairs.append((data[i], data[j]))
            indices.append((i, j))  # Keep track of which subjects are in the pair
    return pairs, indices

# Custom Dataset for testing
class EEGTestPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        return x1, x2

# Testing function
def test_model(model, test_loader, device, actual_labels):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Forward pass
            logits, _, _ = model(x1, x2)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, all_preds)
    f1 = f1_score(actual_labels, all_preds, average='weighted')

    report = classification_report(
        actual_labels,
        all_preds,
        labels=[0, 1],
        target_names=['Stranger', 'Friend'],
        zero_division=0
    )
    return accuracy, f1, report, all_preds


if __name__ == '__main__':
    # Device configuration
    #device = torch.device('cpu')  # Change to 'cuda' if using GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define friend IDs (for evaluation purposes)

    # Load the saved model
    model_path = 'model_dsen.pth'  # Ensure this points to your model file
    num_channels = 30  # Update based on your data
    time_len = 3600    # Update based on your data
    model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess test data
    # data_dir = '/Users/derrick/PycharmProjects/DSEN'  # Update with your test data directory
    data_dir = '/home/derrick/PycharmProjects/DSEN'  # Update with your test data directory
    test_files = ['sub28_5_CSD.mat', 'sub28_6_CSD.mat', 'sub28_7_CSD.mat', 'sub28_9_CSD.mat', 'sub23_0_CSDtest(1).mat', 'sub98_4_CSD.mat', 'sub98_5_CSD.mat','sub98_6_CSD.mat','sub98_7_CSD.mat','sub98_9_CSD.mat','sub101_1_CSD.mat','sub101_4_CSD.mat','sub101_5_CSD.mat','sub101_6_CSD.mat','sub101_7_CSD.mat','sub101_9_CSD.mat','sub102_1_CSD.mat','sub102_4_CSD.mat','sub102_5_CSD.mat','sub102_6_CSD.mat','sub102_7_CSD.mat','sub102_9_CSD.mat','sub24_0_CSD.mat','sub25_0_CSD.mat']  # Update with your test file names

    data, labels, subject_ids = load_and_preprocess_eeg_data(data_dir, test_files)

    # Check data shapes and actual labels
    for i, eeg_data in enumerate(data):
        print(f"Subject {test_files[i]} (ID: {subject_ids[i]}): EEG data shape = {eeg_data.shape}, Label = {'Friend' if labels[i] == 1 else 'Stranger'}")

    # Create test pairs
    pairs, indices = create_test_pairs(data)

    # Create actual labels for the pairs (used only for evaluation)
    actual_labels = []
    for idx_pair in indices:
        i, j = idx_pair
        if labels[i] == labels[j] == 1:
            pair_label = 1  # Friends
        else:
            pair_label = 0  # Strangers
        actual_labels.append(pair_label)

    # Create test dataset and dataloader
    test_dataset = EEGTestPairDataset(pairs)
    batch_size = 1  # Use batch size of 1 to get predictions for each pair
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    accuracy, f1, report, all_preds = test_model(model, test_loader, device, actual_labels)
    print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    print('Classification Report:')
    print(report)

    # Print predictions vs actual labels
    print('Predictions vs Actual Labels:')
    for idx, pred in enumerate(all_preds):
        actual_label = actual_labels[idx]
        print(f'Pair {idx+1}: Predicted = {"Friend" if pred == 1 else "Stranger"}, Actual = {"Friend" if actual_label == 1 else "Stranger"}')


