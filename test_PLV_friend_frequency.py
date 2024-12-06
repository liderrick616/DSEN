import numpy as np
#from pymatreader import read_mat
from scipy.io import loadmat
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
#from scipy import stats
from main import DSENModel
from itertools import combinations

# Define friend IDs (for evaluation purposes)
friend_ids = [23,25,27,35,37,43,47,49,51,57,59,67,69,73,77,79,81,83,85,105,107,
     131,137,143,153,159]
#couples_ids = [41, 61, 63, 65, 71, 87, 109, 115, 121, 127, 129, 133, 135, 139, 141, 147, 149, 151, 155, 157, 163, 165,
               #167, 175, 177, 189, 199, 201, 203]

strangers_ids = [95,97,101,103,117,119,123,125,171,181,183,185,187,191,193,195,197]

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
        band_key = 'PLValue4'
        # Load only the specified band key
        if band_key in mat:
            plv_data = mat[band_key]
            # Squeeze singleton dimensions
            plv_data = np.squeeze(plv_data)

            # Handle dimension variations
            if plv_data.ndim == 2:
                # Single PLV matrix
                plv_matrix = plv_data
            elif plv_data.ndim == 3:
                # Multiple PLV matrices (trials), pick the first or implement your logic
                plv_matrix = plv_data[0]
            else:
                print(f"Unexpected dimension in {band_key} of {file_name}.")
                continue

            # Ensure matrix is square
            if plv_matrix.shape[0] != plv_matrix.shape[1]:
                print(f"PLV matrix {band_key} in {file_name} is not square. Skipping.")
                continue

            # Create a sample with shape (1, num_channels, num_channels)
            sample = plv_matrix[np.newaxis, ...]  # Add new axis for the band dimension
            data.append(sample.astype(np.float32))
            labels.append(label)
            subjects.append(subject_id)
            file_names_list.append(file_name)
        else:
            print(f"{band_key} not found in {file_name}. Skipping.")

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
    data_dir1 = '/home/derrick/PycharmProjects/datayx/hyperemotion1/subB'
    data_dir4 = '/home/derrick/PycharmProjects/datayx/hyperemotion4/subB'
    data_dir5 = '/home/derrick/PycharmProjects/datayx/hyperemotion5/subB'
    data_dir6 = '/home/derrick/PycharmProjects/datayx/hyperemotion6/subB'
    data_dir7 = '/home/derrick/PycharmProjects/datayx/hyperemotion7/subB'
    data_dir9 = '/home/derrick/PycharmProjects/datayx/hyperemotion9/subB'
    file_names1 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only

    ]
    file_names4 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only
    ]
    file_names5 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only
    ]
    file_names6 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only
    ]
    file_names7 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only
    ]
    file_names9 = [
        'obtainPLValue_95.mat',
        'obtainPLValue_97.mat',
        'obtainPLValue_101.mat',
        # above is stranger
        'obtainPLValue_77.mat',
        'obtainPLValue_79.mat',
        'obtainPLValue_81.mat',
        'obtainPLValue_83.mat',
        'obtainPLValue_85.mat',
        'obtainPLValue_105.mat',
        'obtainPLValue_107.mat',
        'obtainPLValue_131.mat',
        'obtainPLValue_137.mat',
        'obtainPLValue_143.mat',
        'obtainPLValue_153.mat',
        'obtainPLValue_159.mat'
        # above is friend only
    ]
    data1, labels1, subjects1, file_names_list1 = load_eeg_data_mat(data_dir1, file_names1, friend_ids)
    data4, labels4, subjects4, file_names_list4 = load_eeg_data_mat(data_dir4, file_names4, friend_ids)
    data5, labels5, subjects5, file_names_list5 = load_eeg_data_mat(data_dir5, file_names5, friend_ids)
    data6, labels6, subjects6, file_names_list6 = load_eeg_data_mat(data_dir6, file_names6, friend_ids)
    data7, labels7, subjects7, file_names_list7 = load_eeg_data_mat(data_dir7, file_names7, friend_ids)
    data9, labels9, subjects9, file_names_list9 = load_eeg_data_mat(data_dir9, file_names9, friend_ids)
    data = data1 + data4 + data5 + data6 + data7 + data9
    labels = np.concatenate([labels1, labels4, labels5, labels6, labels7, labels9])
    subject_ids = subjects1 + subjects4 + subjects5 + subjects6 + subjects7 + subjects9
    file_names_list = file_names_list1 + file_names_list4 + file_names_list5 + file_names_list6 + file_names_list7 + file_names_list9
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
    LowBand = 30  # Beta band lower edge
    HighBand = 45  # Gamma band upper edge

    # Create test dataset and dataloader
    test_dataset = EEGTestPairDataset(pairs)
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
