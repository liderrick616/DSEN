import numpy as np
#from pymatreader import read_mat
from scipy.io import loadmat
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats
from main import DSENModel # Ensure main.py is structured properly for import
from itertools import combinations

# Define friend IDs (for evaluation purposes)
friend_ids = [
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 57, 58, 59, 60, 67, 68, 69, 70, 73, 74, 75, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 105, 106, 107, 108, 111, 112, 131, 132, 137, 138,
    143, 144, 153, 154, 159, 160, 161, 162, 179, 180, 42, 55, 56, 61, 62, 63,
    64, 65, 66, 71, 72, 87, 88, 89, 90, 91, 92, 93, 94, 99, 100, 109, 110, 113,
    114, 115, 116, 121, 122, 127, 128, 129, 130, 133, 134, 135, 136, 139, 140,
    141, 142, 145, 146, 147, 148, 149, 150151, 152, 155, 156, 157, 158, 163,
    164, 165, 166, 167, 168, 169, 170, 175, 176, 177, 178, 189, 190, 199, 200,
    201, 202, 203, 204, 205, 206
]


# Include the extract_subject_id function
def extract_subject_id(file_name):
    import re
    import string
    # Remove non-printable characters
    printable = set(string.printable)
    file_name_clean = ''.join(filter(lambda x: x in printable, file_name))
    # Strip whitespace
    file_name_clean = file_name_clean.strip()
    print(f"Extracting subject ID from: '{file_name_clean}'")
    match = re.search(r'obtainPLValue_(\d+)\.mat$', file_name_clean, re.IGNORECASE)
    if match:
        subject_id = int(match.group(1))
        print(f"Extracted subject ID: {subject_id}")
        return subject_id
    else:
        print("No match found.")
        return None


def load_eeg_data_mat(data_dir, file_names, friend_ids):
    data = []
    labels = []
    subjects = []
    file_names_list = []
    num_frequency_bands = 4  # Since we have PLValue1 to PLValue4

    for file_name in file_names:
        # Construct file path
        mat_file = os.path.join(data_dir, file_name)
        print(f"Processing file: '{file_name}'")
        # Load the .mat file
        try:
            mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
            print(f"Successfully loaded {mat_file}")
        except Exception as e:
            print(f"Failed to load {mat_file}: {e}")
            continue

        # Extract subject ID from file name
        subject_id = extract_subject_id(file_name)
        if subject_id is None:
            print(f"Could not extract subject ID from '{file_name}'. Skipping.")
            continue

        # Assign label based on subject IDs
        label = 1 if subject_id in friend_ids else 0

        # Collect PLV matrices for each frequency band
        plv_matrices = []
        for key in ['PLValue1', 'PLValue2', 'PLValue3', 'PLValue4']:
            if key in mat:
                plv_data = mat[key]
                # Squeeze singleton dimensions
                plv_data = np.squeeze(plv_data)

                if plv_data.ndim == 2:
                    # Single PLV matrix
                    plv_matrix = plv_data
                elif plv_data.ndim == 3:
                    # Multiple PLV matrices (trials)
                    plv_matrix = plv_data[0]  # You may need to adjust this based on your data
                else:
                    print(f"Unexpected dimension in {key} of {file_name}.")
                    continue

                # Ensure matrix is square
                num_channels = plv_matrix.shape[0]
                if plv_matrix.shape[0] != plv_matrix.shape[1]:
                    print(f"PLV matrix {key} in {file_name} is not square. Skipping.")
                    continue

                plv_matrices.append(plv_matrix)
            else:
                print(f"{key} not found in {file_name}. Skipping.")

        if len(plv_matrices) == num_frequency_bands:
            # Stack the PLV matrices to create a sample
            sample = np.stack(plv_matrices, axis=0)  # Shape: (num_frequency_bands, num_channels, num_channels)
            data.append(sample)
            labels.append(label)
            subjects.append(subject_id)
            file_names_list.append(file_name)
        else:
            print(f"Not all frequency bands found for {file_name}. Skipping.")

    return data, labels, subjects, file_names_list



# Function to create pairs for testing
def create_test_pairs(data, labels):
    pairs = []
    pair_labels = []
    indices = list(range(len(data)))

    # Create all unique pairs using combinations
    for idx1, idx2 in combinations(indices, 2):
        x1 = data[idx1]
        x2 = data[idx2]
        pairs.append((x1, x2))

        # Assign label: 1 if both are Friends, else 0
        if labels[idx1] == 1 and labels[idx2] == 1:
            pair_label = 1  # Friends
        else:
            pair_label = 0  # Strangers
        pair_labels.append(pair_label)

    return pairs, pair_labels

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

"""
def perform_t_tests( isc_features_array, actual_labels):
    # Separate features by group
    plv_friends = []
    plv_strangers = []
    isc_friends = []
    isc_strangers = []

    for idx in range(len(actual_labels)):
        label = actual_labels[idx]
        isc_feature = isc_features_array[idx]

        if label == 1:  # Friends
            isc_friends.append(isc_feature)
        else:  # Strangers
            isc_strangers.append(isc_feature)

    # Compute mean PLV and ISC for each pair
    isc_friends_mean = [np.mean(feature) for feature in isc_friends]
    isc_strangers_mean = [np.mean(feature) for feature in isc_strangers]

    # Perform t-tests
    alpha = 0.05

    # ISC t-test
    t_stat_isc, p_value_isc = stats.ttest_ind(isc_friends_mean, isc_strangers_mean, equal_var=False)
    p_value_isc_one_tailed = p_value_isc / 2 if t_stat_isc > 0 else 1 - (p_value_isc / 2)
    reject_null_isc = p_value_isc_one_tailed < alpha
    print(f"ISC t-statistic: {t_stat_isc:.4f}, p-value (one-tailed): {p_value_isc_one_tailed:.4f}, Reject null: {reject_null_isc}")
"""
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model
    model_path = 'model_dsen.pth'  # Ensure this points to your model file
    num_channels = 30  # Update based on your data
    time_len = 1000  # Update based on your data
    num_features = 128
    #model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)
    model = DSENModel(num_channels=num_channels, time_len=time_len, num_features=num_features, num_classes=2).to(device)

    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model weights from {model_path}.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        exit()

    model.eval()

    # Load and preprocess test data
    data_dir = '/home/derrick/PycharmProjects/datayx/hyperemotion9/subB'  # Update with your test data directory
    test_files = [
        'obtainPLValue_101.mat',  'obtainPLValue_137.mat',   'obtainPLValue_175.mat',   'obtainPLValue_29.mat',
        'obtainPLValue_71.mat',
        'obtainPLValue_103.mat',   'obtainPLValue_139.mat',   'obtainPLValue_177.mat',   'obtainPLValue_33.mat',
        'obtainPLValue_73.mat'
    ]  # Update with your test file names
    data, labels, subject_ids, file_names_list = load_eeg_data_mat(data_dir, test_files, friend_ids)

    # Check if data was loaded
    if not data:
        print("No data was loaded. Exiting.")
        exit()

        # Verify list lengths
    print(f"Total trials loaded: {len(data)}")
    print(f"Total labels: {len(labels)}")
    print(f"Total subject IDs: {len(subject_ids)}")

    # Add assertion to ensure lists are aligned
    assert len(data) == len(labels) == len(subject_ids), (
        f"Data length: {len(data)}, Labels length: {len(labels)}, "
        f"Subject IDs length: {len(subject_ids)}. They must be equal."
    )

    # Check data shapes and actual labels
    for i, eeg_data in enumerate(data):
        print(
            f"Subject ID: {subject_ids[i]}, EEG data shape = {eeg_data.shape}, "
            f"Label = {'Friend' if labels[i] == 1 else 'Stranger'}"
        )
    pairs, pair_labels = create_test_pairs(data, labels)

    # Compute PLV and ISC features for test pairs
    Fs = 256  # Example sampling frequency, update as needed
    LowBand = 4  # Beta band lower edge
    HighBand = 45  # Gamma band upper edge

    # Create test dataset and dataloader
    test_dataset = EEGTestPairDataset(pairs)
    batch_size = 4  # Use batch size of 1 to get predictions for each pair
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    accuracy, f1, report, all_preds = test_model(model, test_loader, device, pair_labels)
    print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    print('Classification Report:')
    print(report)

    # Print predictions vs actual labels
    print('Predictions vs Actual Labels:')
    for idx, pred in enumerate(all_preds):
        actual_label = pair_labels[idx]
        print(
            f'Pair {idx + 1}: Predicted = {"Friend" if pred == 1 else "Stranger"}, '
            f'Actual = {"Friend" if actual_label == 1 else "Stranger"}'
        )

    # Perform t-tests (if applicable)

