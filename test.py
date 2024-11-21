import numpy as np
#from pymatreader import read_mat
from scipy.io import loadmat
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats
from main import DSENModel, PLVCalculator, ISCCalculator  # Ensure main.py is structured properly for import
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
    match = re.search(r'^(\d+)_CleanData\.mat$', file_name)
    if match:
        return int(match.group(1))
    else:
        return None


def load_eeg_data_mat(data_dir, file_names, friend_ids):
    data = []
    labels = []
    subject_ids = []
    min_length = None  # Keep track of the minimum length

    for file_name in file_names:
        mat_file = os.path.join(data_dir, file_name)

        # Check if file exists
        if not os.path.isfile(mat_file):
            print(f"File not found: {mat_file}")
            continue

        # Check file size
        file_size = os.path.getsize(mat_file)
        print(f"Loading {mat_file}, file size: {file_size} bytes")
        if file_size == 0:
            print(f"File {mat_file} is empty. Skipping.")
            continue

        # Attempt to load the .mat file with enhanced parameters
        try:
            mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
            print(f"Successfully loaded {mat_file}")
        except Exception as e:
            print(f"Failed to load {mat_file} using loadmat: {e}")
            continue

        # Inspect the keys in the mat file
        print(f"Keys in {mat_file}: {mat.keys()}")

        # Check if 'data_all' exists
        if 'data_all' not in mat:
            print(f"'data_all' key not found in {mat_file}. Skipping.")
            continue

        data_all = mat['data_all']

        # Inspect the 'data_all' structure
        print(f"Type of 'data_all': {type(data_all)}")
        if isinstance(data_all, np.ndarray):
            print(f"'data_all' shape: {data_all.shape}")
        else:
            print(f"'data_all' type: {type(data_all)}")

        # Access 'trial' field
        if not hasattr(data_all, 'trial'):
            print(f"'trial' attribute not found in 'data_all' for {file_name}. Skipping.")
            continue

        trials = data_all.trial
        num_trials = trials.shape[1] if trials.ndim > 1 else 1  # Adjust based on structure
        print(f"Number of trials in {file_name}: {num_trials}")

        # Extract subject ID from file name
        subject_id = extract_subject_id(file_name)
        subject_ids.append(subject_id)

        # Assign label based on subject IDs
        label = 1 if subject_id in friend_ids else 0

        for i in range(num_trials):
            try:
                # Access each trial
                if trials.ndim == 1:
                    trial_data = trials[i]
                else:
                    trial_data = trials[0, i]

                # Ensure trial_data is a numpy array
                trial_data = np.array(trial_data)

                # Use the last 1000 ms (time points 1000 to 1999)
                if trial_data.shape[1] < 1000:
                    print(f"Trial {i + 1} in {file_name} has less than 1000 time points. Skipping.")
                    continue
                trial_data = trial_data[:, -1000:]

                # Convert to float32
                trial_data = trial_data.astype(np.float32)

                # Transpose if necessary to ensure correct EEG data shape
                if trial_data.shape[0] > trial_data.shape[1]:
                    trial_data = trial_data.T

                # Update minimum length
                if min_length is None or trial_data.shape[1] < min_length:
                    min_length = trial_data.shape[1]

                # Append the EEG data and the corresponding label to their respective lists
                data.append(trial_data)
                labels.append(label)

                # Print data shape and label for verification
                print(
                    f"Loaded {file_name}, Trial {i + 1}: Subject ID = {subject_id}, "
                    f"Assigned Label = {'Friend' if label == 1 else 'Stranger'}, EEG data shape = {trial_data.shape}"
                )
            except Exception as e:
                print(f"Error processing Trial {i + 1} in {file_name}: {e}")
                continue  # Skip this trial

    # Truncate all data to the minimum length
    if min_length is not None:
        for i in range(len(data)):
            data[i] = data[i][:, :min_length]  # Truncate to min_length

    return data, labels, subject_ids

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
    def __init__(self, pairs, plv_features, isc_features):
        self.pairs = pairs
        self.plv_features = plv_features
        self.isc_features = isc_features

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        plv = self.plv_features[idx]
        isc = self.isc_features[idx]
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        plv = torch.from_numpy(plv).float()
        isc = torch.from_numpy(isc).float()
        return x1, x2, plv, isc



# Testing function
def test_model(model, test_loader, device, actual_labels):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x1, x2, plv, isc = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            plv = plv.to(device)
            isc = isc.to(device)

            # Forward pass
            logits, _, _ = model(x1, x2, plv, isc)
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



def perform_t_tests(plv_features_array, isc_features_array, actual_labels):
    # Separate features by group
    plv_friends = []
    plv_strangers = []
    isc_friends = []
    isc_strangers = []

    for idx in range(len(actual_labels)):
        label = actual_labels[idx]
        plv_feature = plv_features_array[idx]
        isc_feature = isc_features_array[idx]

        if label == 1:  # Friends
            plv_friends.append(plv_feature)
            isc_friends.append(isc_feature)
        else:  # Strangers
            plv_strangers.append(plv_feature)
            isc_strangers.append(isc_feature)

    # Compute mean PLV and ISC for each pair
    plv_friends_mean = [np.mean(feature) for feature in plv_friends]
    plv_strangers_mean = [np.mean(feature) for feature in plv_strangers]
    isc_friends_mean = [np.mean(feature) for feature in isc_friends]
    isc_strangers_mean = [np.mean(feature) for feature in isc_strangers]

    # Perform t-tests
    alpha = 0.05

    # PLV t-test
    t_stat_plv, p_value_plv = stats.ttest_ind(plv_friends_mean, plv_strangers_mean, equal_var=False)
    p_value_plv_one_tailed = p_value_plv / 2 if t_stat_plv > 0 else 1 - (p_value_plv / 2)
    reject_null_plv = p_value_plv_one_tailed < alpha

    # ISC t-test
    t_stat_isc, p_value_isc = stats.ttest_ind(isc_friends_mean, isc_strangers_mean, equal_var=False)
    p_value_isc_one_tailed = p_value_isc / 2 if t_stat_isc > 0 else 1 - (p_value_isc / 2)
    reject_null_isc = p_value_isc_one_tailed < alpha

    print(f"PLV t-statistic: {t_stat_plv:.4f}, p-value (one-tailed): {p_value_plv_one_tailed:.4f}, Reject null: {reject_null_plv}")
    print(f"ISC t-statistic: {t_stat_isc:.4f}, p-value (one-tailed): {p_value_isc_one_tailed:.4f}, Reject null: {reject_null_isc}")

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model
    model_path = 'model_dsen.pth'  # Ensure this points to your model file
    num_channels = 30  # Update based on your data
    time_len = 1000  # Update based on your data
    model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)

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
    """
    obtainPLValue_101.mat   obtainPLValue_137.mat   obtainPLValue_175.mat   obtainPLValue_29.mat   obtainPLValue_71.mat
    obtainPLValue_103.mat   obtainPLValue_139.mat   obtainPLValue_177.mat   obtainPLValue_33.mat   obtainPLValue_73.mat
     '103_CleanData.mat', '105_CleanData.mat', '106_CleanData.mat', '107_CleanData.mat',
        '108_CleanData.mat', '109_CleanData.mat', '110_CleanData.mat', '111_CleanData.mat',
        '113_CleanData.mat', '114_CleanData.mat'
    """
    data, labels, subject_ids = load_eeg_data_mat(data_dir, test_files, friend_ids)

    # Check if data was loaded
    if not data:
        print("No data was loaded. Exiting.")
        exit()

    # Check data shapes and actual labels
    for i, eeg_data in enumerate(data):
        print(
            f"Subject ID: {subject_ids[i]}, EEG data shape = {eeg_data.shape}, "
            f"Label = {'Friend' if labels[i] == 1 else 'Stranger'}"
        )
    """
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
    """
    pairs, pair_labels = create_test_pairs(data, labels)

    # Compute PLV and ISC features for test pairs
    Fs = 256  # Example sampling frequency, update as needed
    LowBand = 13  # Beta band lower edge
    HighBand = 45  # Gamma band upper edge
    plv_calculator = PLVCalculator(Fs, LowBand, HighBand)
    isc_calculator = ISCCalculator(Fs, LowBand, HighBand)
    # Compute PLV and ISC features for each trial
    plv_features_list = []
    isc_features_list = []
    for x1, x2 in pairs:
        plv_features = plv_calculator.compute_plv_multichannel(x1, x2)
        isc_features = isc_calculator.compute_isc_multichannel(x1, x2)
        plv_features_list.append(plv_features)
        isc_features_list.append(isc_features)

    # Convert lists to numpy arrays
    plv_features_array = np.array(plv_features_list)
    isc_features_array = np.array(isc_features_list)

    # Create test dataset and dataloader
    test_dataset = EEGTestPairDataset(pairs, plv_features_array, isc_features_array)
    batch_size = 1  # Use batch size of 1 to get predictions for each pair
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
    perform_t_tests(plv_features_array, isc_features_array, pair_labels)

