import numpy as np
import os
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import EdgeConv, global_max_pool
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from scipy.signal import firwin, filtfilt
import random
from scipy.signal import resample


# Slide 7
# Helper function to create a fully connected edge index
def create_fully_connected_edge_index(num_nodes):
    edge_index = torch.tensor(list(combinations(range(num_nodes), 2)), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).t()
    return edge_index


# Slide 9
# compare similar pair with dissimilar sample
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, dist_func='cosine'):
        super(TripletLoss, self).__init__()
        self.margin = margin  # positive float, define minimum difference between pos and neg distance
        self.dist_func = dist_func  # default is cosine
# Given anchor (reference sample), positive (sample similar to anchor), negative sample (sample disimilar to anchor)

    def forward(self, anchor, positive, negative):
        if self.dist_func == 'euclidean':
            distance_positive = F.pairwise_distance(anchor, positive)
            distance_negative = F.pairwise_distance(anchor, negative)
        # exception handler for dist_func: find pairwise distance between input vectors or column of input matrices.
        # dist(x, y) =∥x−y + ϵe∥p
        elif self.dist_func == 'cosine':
            distance_positive = 1.0 - F.cosine_similarity(anchor, positive)
            distance_negative = 1.0 - F.cosine_similarity(anchor, negative)
            # dist positive and negative
        else:  # error handling
            raise ValueError(f"Unsupported dist_func: {self.dist_func}")
        # find distance between anchor & pos & neg embedding
        # find triplet loss based on this distance
        losses = F.relu(distance_positive - distance_negative + self.margin)
        # eqn:11
        # apply relu to prevent negative inputs(by return 0)
        return losses.mean()  # Ltriplet loss (mean loss over batch)


# Slide 9

class CCALoss(nn.Module):
    def __init__(self):
        super(CCALoss, self).__init__()

    def forward(self, H1, H2):
        eps = 1e-10  # Small constant to prevent division by zero

        # Center the variables
        H1_centered = H1 - H1.mean(dim=0)
        H2_centered = H2 - H2.mean(dim=0)

        # Compute covariance matrices
        N = H1.size(0) - 1  # Adjust for unbiased estimate

        # Cross-covariance
        SigmaHat12 = (1.0 / N) * H1_centered.t().mm(H2_centered)

        # Variances
        SigmaHat11 = (1.0 / N) * H1_centered.t().mm(H1_centered) + eps * torch.eye(H1.size(1)).to(H1.device)
        SigmaHat22 = (1.0 / N) * H2_centered.t().mm(H2_centered) + eps * torch.eye(H2.size(1)).to(H2.device)

        # Compute the trace of covariance matrices
        trace_SigmaHat11 = torch.trace(SigmaHat11)
        trace_SigmaHat22 = torch.trace(SigmaHat22)

        # Compute the correlation coefficient
        corr_num = torch.trace(SigmaHat12)
        corr_den = torch.sqrt(trace_SigmaHat11 * trace_SigmaHat22)

        corr = corr_num / (corr_den + eps)  # Add eps to denominator to prevent division by zero

        # Negative correlation as loss
        loss = -corr
        return loss



# Slide 6,7
# DSEN Model Implementation
class DSEN(nn.Module):
    def __init__(self, num_features=128, num_channels=30, num_segments=9, num_frequency_bands=4):
        super(DSEN, self).__init__()
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_segments = num_segments
        self.num_frequency_bands = num_frequency_bands
        self.plv_feature_size = int(num_channels * (num_channels - 1) / 2) * num_frequency_bands
        self.time_len = self.plv_feature_size // self.num_channels

        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=3,
                groups=1,
                bias=False,
                padding=1
            ),
            nn.BatchNorm1d(self.num_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(100)
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=3,
                groups=1,
                bias=False,
                padding=1
            ),
            nn.BatchNorm1d(self.num_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(128)
        )

        # EdgeConv layers and linear layers remain the same
        self.conv1 = EdgeConv(Sequential(Linear(2 * 128, 128), ReLU(), Linear(128, 128), ReLU(), BatchNorm1d(128), Dropout(p=0.25)))
        self.conv2 = EdgeConv(Sequential(Linear(2 * 128, 256), ReLU(), Linear(256, 256), ReLU(), BatchNorm1d(256), Dropout(p=0.25)))
        self.conv3 = EdgeConv(Sequential(Linear(2 * 256, 512), ReLU(), Linear(512, 512), ReLU(), BatchNorm1d(512), Dropout(p=0.25)))

        self.linear1 = Linear(128 + 256 + 512, 256)
        self.linear2 = Linear(256, 128)

    def forward(self, x):
        batch_size = x.size(0)
        num_frequency_bands = x.size(1)
        num_channels = x.size(2)
        plv_features = []
        for i in range(num_frequency_bands):
            plv_matrix = x[:, i, :, :]  # Shape: (batch_size, num_channels, num_channels)
            # Extract upper triangle indices
            triu_indices = torch.triu_indices(num_channels, num_channels, offset=1)
            upper_tri = plv_matrix[:, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, num_features)
            plv_features.append(upper_tri)
        # Concatenate features from all frequency bands
        x = torch.cat(plv_features, dim=1)  # Shape: (batch_size, plv_feature_size)
        x = x.view(batch_size, self.num_channels, self.time_len)  # Shape: (batch_size, num_channels, time_len)

        x = self.block_1(x)  # x now has shape (batch_size, num_channels, 100)
        x = self.block_2(x)  # x now has shape (batch_size, num_channels, 128)

        x = x.view(batch_size * self.num_channels, -1)  # Shape: (batch_size * num_channels, 128)

        # Create edge_index for a single graph
        edge_index = create_fully_connected_edge_index(self.num_channels)  # Shape: [2, num_edges]
        # Adjust edge_index for batching
        edge_indices = []
        # Adjusts the edge indices to account for the batch dimension
        for i in range(batch_size):
            offset = i * self.num_channels
            edge_index_i = edge_index + offset
            edge_indices.append(edge_index_i)
        edge_index = torch.cat(edge_indices, dim=1).to(x.device)  # Concatenate along the second dimension
        # Create batch tensor
        batch = torch.arange(batch_size).unsqueeze(1).repeat(1, self.num_channels).view(-1).to(x.device)
        # creates tensor, Indicates which nodes belong to which sample in the batch.
        # Apply EdgeConv layers, processes node features
        x1 = self.conv1(x, edge_index)  # 128
        x2 = self.conv2(x1, edge_index)  # 256
        x3 = self.conv3(x2, edge_index)  # 512
        # Apply Global pooling, Concatenate pooled features, create a fixed-size representation for each sample
        x1_pooled = global_max_pool(x1, batch)
        x2_pooled = global_max_pool(x2, batch)
        x3_pooled = global_max_pool(x3, batch)
        out = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)
        # Combines the features from all three EdgeConv layers
        # Apply Fully connected layers
        out = F.relu(self.linear1(out))  # reduce dim to 256, apply RELu activation function
        out = F.dropout(out, p=0.25)  # prevent overfitting
        out = F.relu(self.linear2(out)) # reduce dim to 128
        # φ (x1 pooled + x2 pooled + x3 pooled)
        # φ is linear transformation
        return out  # Shape: (batch_size, 128) embedding for each sample


#Slide 8
# Relation Classifier with Attention Mechanism
class RelationClassifier(nn.Module):
    def __init__(self, num_features=128, num_classes=2):
        super(RelationClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.hidden_size = num_features  # size 128, dimensionality of the embeddings
        self.scale = self.hidden_size ** 0.5  # constant scale factor
        # Attention Mechanism weights
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for query vectors.
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for key vectors.
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for value vectors.
        #total_input_size = self.hidden_size * 2 + self.num_plv_features + self.num_isc_features
        # self.fc1 = Linear(self.hidden_size * 2, self.hidden_size)  # reduce concatenation back to 128 dim
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, num_classes)  # maps to logits (friends or strangers)

    def forward(self, x1, x2):  # equation (5) from the thesis
        # apply attention mechanism, classify relationship
        # x1 is first EEG sample, x2 is second EEG sample
        Q_x1 = self.W_q(x1).unsqueeze(1) # Shape: (batch_size, 1, hidden_size = 128)
        K_x2 = self.W_k(x2).unsqueeze(1)
        V_x2 = self.W_v(x2).unsqueeze(1)

        Q_x2 = self.W_q(x2).unsqueeze(1)
        K_x1 = self.W_k(x1).unsqueeze(1)
        V_x1 = self.W_v(x1).unsqueeze(1)
        # attention scores between embeddings. S(x,y) of equation (5)
        score_1 = torch.matmul(Q_x1, K_x2.transpose(-2, -1)) / self.scale
        score_2 = torch.matmul(Q_x2, K_x1.transpose(-2, -1)) / self.scale
        # probabilities that sums to 1, applied on last dimension
        attention_w_1 = F.softmax(score_1, dim=-1)
        attention_w_2 = F.softmax(score_2, dim=-1)
        # fused_1 = attention_w_1⋅V_x2
        # remove the 1 from (batch_size, 1, hidden_size) using squeeze(1)
        fused_1 = torch.matmul(attention_w_1, V_x2).squeeze(1)  # Shape: (batch_size, hidden_size)
        fused_2 = torch.matmul(attention_w_2, V_x1).squeeze(1)
        # Concatenate the fused features into single feature vector for classification
        #x_fused = torch.cat((fused_1, fused_2), dim=1)  # Shape: (batch_size, hidden_size * 2 = 256)
        x_fused = torch.cat((fused_1, fused_2), dim=1)
        # Classification Layers (fully connected)
        x = F.relu(self.fc1(x_fused))  # reduce dim to 128,apply ReLu activation function
        x = F.dropout(x, p=0.25)  # prevent overfitting
        x = self.fc2(x)  # reduce num_classes in x to 2 (1 friends 0 strangers)
        return x  # Logits, unnormalized scores for each class


class ISCCalculator:
    def __init__(self, Fs, LowBand, HighBand):
        self.Fs = Fs
        self.LowBand = LowBand
        self.HighBand = HighBand
        self.nyq = Fs / 2.0  # Nyquist frequency

        # Precompute the filter coefficients
        self.numtaps = int((1 / ((LowBand + HighBand) / 2) * 5 * Fs))
        if self.numtaps % 2 == 0:
            self.numtaps += 1  # Ensure numtaps is odd

        # Design the bandpass filter
        self.bfil = firwin(
            self.numtaps,
            [LowBand / self.nyq, HighBand / self.nyq],
            pass_zero='bandpass'
        )

    def compute_isc(self, sig1, sig2):
        # Convert tensors to numpy arrays if necessary
        if isinstance(sig1, torch.Tensor):
            sig1 = sig1.detach().cpu().numpy()
        if isinstance(sig2, torch.Tensor):
            sig2 = sig2.detach().cpu().numpy()
        # Bandpass filter
        X = filtfilt(self.bfil, [1.0], sig1.astype(np.float64))
        Y = filtfilt(self.bfil, [1.0], sig2.astype(np.float64))
        # Compute ISC
        corr_coef = np.corrcoef(X, Y)[0, 1]
        return corr_coef

    def compute_isc_multichannel(self, sig1, sig2):
        # Convert tensors to numpy arrays if necessary
        if isinstance(sig1, torch.Tensor):
            sig1 = sig1.detach().cpu().numpy()
        if isinstance(sig2, torch.Tensor):
            sig2 = sig2.detach().cpu().numpy()
        isc_values = []
        for ch in range(sig1.shape[0]):
            isc = self.compute_isc(sig1[ch, :], sig2[ch, :])
            isc_values.append(isc)
        return np.array(isc_values)


# Putting it all together
class DSENModel(nn.Module):
    def __init__(self, num_channels=30, time_len=3600, num_features=128,num_classes=2):
        super(DSENModel, self).__init__()
        self.encoder = DSEN(num_channels=num_channels)
        #self.num_frequency_bands = num_frequency_bands
        self.classifier = RelationClassifier(
            num_features=num_features,
            num_classes=num_classes,
            #num_plv_features=0,
            #num_isc_features=num_isc_features
        )

    def forward(self, x1, x2):
        # Encode both inputs
        h_x1 = self.encoder(x1)
        h_x2 = self.encoder(x2)
        # Classification
        logits = self.classifier(h_x1, h_x2)
        return logits, h_x1, h_x2


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

# slide 14
# Load your data from .mat files


def preprocess_eeg_data(eeg_data, target_time_len=1000, target_fs=200):
    num_channels, original_time_len = eeg_data.shape
    original_fs = 1000  # Adjust to your actual sampling frequency

    # Downsample if necessary
    if original_fs != target_fs:
        num_samples = int(eeg_data.shape[1] * (target_fs / original_fs))
        eeg_data = resample(eeg_data, num_samples, axis=1)

    # Ensure data has target_time_len time points
    if eeg_data.shape[1] >= target_time_len:
        eeg_data = eeg_data[:, :target_time_len]
    else:
        padding = np.zeros((num_channels, target_time_len - eeg_data.shape[1]))
        eeg_data = np.hstack((eeg_data, padding))

    eeg_data = eeg_data.astype(np.float32)
    return eeg_data


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




def create_pairs_and_triplets(data, labels, num_pairs=1000, num_triplets=1000):
    pairs = []
    pair_labels = []
    triplets = []

    # Convert labels to integers if they are tensors
    if isinstance(labels[0], torch.Tensor):
        labels = [label.item() for label in labels]

    # Group data indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    # Remove labels that have less than 2 samples
    labels_with_enough_samples = [label for label in label_to_indices if len(label_to_indices[label]) >= 2]
    if not labels_with_enough_samples:
        print("Not enough samples to create pairs and triplets.")
        return pairs, pair_labels, triplets

    # Generate positive and negative pairs
    for _ in range(num_pairs // 2):
        # Positive pair (same label)
        label = random.choice(labels_with_enough_samples)
        idx1, idx2 = random.sample(label_to_indices[label], 2)
        pairs.append((data[idx1], data[idx2]))
        pair_labels.append(1)

        # Negative pair (different labels)
        label1, label2 = random.sample(labels_with_enough_samples, 2)
        idx1 = random.choice(label_to_indices[label1])
        idx2 = random.choice(label_to_indices[label2])
        pairs.append((data[idx1], data[idx2]))
        pair_labels.append(0)

    # Generate triplets
    for _ in range(num_triplets):
        # Anchor and positive (same label)
        label = random.choice(labels_with_enough_samples)
        idxs = label_to_indices[label]
        if len(idxs) >= 2:
            anchor_idx, positive_idx = random.sample(idxs, 2)
            # Negative (different label)
            negative_label = random.choice([l for l in labels_with_enough_samples if l != label])
            negative_idx = random.choice(label_to_indices[negative_label])
            triplets.append((data[anchor_idx], data[positive_idx], data[negative_idx]))

    return pairs, pair_labels, triplets


# Slide 14
class EEGDataset(Dataset):
    def __init__(self, pairs, pair_labels, triplets, Fs, LowBand, HighBand):
        self.pairs = pairs
        self.pair_labels = pair_labels
        self.triplets = triplets
        self.isc_calculator = ISCCalculator(Fs, LowBand, HighBand)
        self.plv_features_list = [torch.zeros(1) for _ in self.pairs]  # Placeholder if needed
        self.isc_features_list = [torch.zeros(1) for _ in self.pairs]  # Placeholder if needed
        """
        for idx, (x1, x2) in enumerate(self.pairs):
            # Compute ISC features
            isc_features = self.isc_calculator.compute_isc_multichannel(x1, x2)
            self.isc_features_list.append(torch.from_numpy(isc_features).float())
            # Populate plv_features_list with zero tensors or appropriate values
            num_plv_features = x1.shape[1]  # Assuming x1.shape is (num_frequency_bands, num_channels, num_channels)
            plv_features = torch.zeros(num_plv_features)
            self.plv_features_list.append(plv_features)
            if idx % 100 == 0:
                print(f"Computed features for {idx + 1}/{len(self.pairs)} pairs")
        """

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.pair_labels[idx]
        # For triplet loss, pick a random triplet
        triplet_idx = random.randint(0, len(self.triplets) - 1)
        anchor, positive, negative = self.triplets[triplet_idx]

        # Convert data to torch tensors
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()
        label = torch.tensor(label).long()

        # Retrieve plv_features and isc_features
        plv_features = self.plv_features_list[1]
        isc_features = self.isc_features_list[1]

        return x1, x2, anchor, positive, negative, label, plv_features, isc_features


# slide 13
def train_model(model, train_loader, optimizer_f, optimizer_c, criterion_classification, criterion_triplet, criterion_cca, device):
    model.train()
    total_loss_combined = 0
    total_loss_triplet = 0
    all_labels = []
    all_preds = []
    for batch_idx, batch in enumerate(train_loader):
        x1, x2, anchor, positive, negative, label, _, _ = batch  # Ignore placeholder features
        x1 = x1.to(device)
        x2 = x2.to(device)
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        label = label.to(device)

        # Update encoder's parameters using triplet loss
        h_anchor = model.encoder(anchor)
        h_positive = model.encoder(positive)
        h_negative = model.encoder(negative)
        loss_triplet = criterion_triplet(h_anchor, h_positive, h_negative)
        optimizer_f.zero_grad()
        loss_triplet.backward()
        optimizer_f.step()

        # Update encoder and classifier using combined loss
        logits, h_x1, h_x2 = model(x1, x2)
        loss_classification = criterion_classification(logits, label)
        loss_cca = criterion_cca(h_x1, h_x2)
        alpha = 1.0
        beta = 1.0
        loss_combined = alpha * loss_classification - beta * loss_cca
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        loss_combined.backward()
        optimizer_f.step()
        optimizer_c.step()

        total_loss_combined += loss_combined.item()
        total_loss_triplet += loss_triplet.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss Classification: {loss_classification.item():.4f}, '
                  f'Loss CCA: {loss_cca.item():.4f}, '
                  f'Loss Triplet: {loss_triplet.item():.4f}, '
                  f'Loss Combined: {loss_combined.item():.4f}')

    avg_loss_combined = total_loss_combined / len(train_loader)
    avg_loss_triplet = total_loss_triplet / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss_combined, avg_loss_triplet, f1



if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from .mat files
    data_dir = '/home/derrick/PycharmProjects/datayx/hyperemotion4/subB'

    friend_ids = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,
     47,48,49,50,51,52,53,54,57,58,59,60,67,68,69,70,73,74,75,77,78,79,80,81,82,83,84,85,86,105,106,107,108,111,112,
     131,132,137,138,143,144,153,154,159,160,161,162,179,180,42,55,56,61,62,63,64,65,66,71,72,87,88,89,90,91,92,93,94,
                  99,100,109,110,113,114,115,116,121,122,127,128,129,130,133,134,135,136,139,140,141,142,145,146,147,
                  148,149,150151,152,155,156,157,158,163,164,165,166,167,168,169,170,175,176,177,178,189,190,199,200,
                  201,202,203,204,205,206]

    file_names = [
        'obtainPLValue_101.mat', 'obtainPLValue_137.mat', 'obtainPLValue_175.mat', 'obtainPLValue_29.mat',
        'obtainPLValue_71.mat',
        'obtainPLValue_103.mat', 'obtainPLValue_139.mat', 'obtainPLValue_177.mat',
        'obtainPLValue_73.mat'
    ]

    # Load PLV data and labels
    data, labels, subjects, file_names_list = load_eeg_data_mat(data_dir, file_names, friend_ids)
    #data = [torch.tensor(d, dtype=torch.float32) for d in data]
    data = [d.astype(np.float32) if not isinstance(d, np.ndarray) else d for d in data]
    labels = np.array(labels, dtype=np.int64)

    # Verify that lengths match
    assert len(data) == len(labels), "Mismatch in lengths of data and labels"

    # Print data information
    for i, plv_data in enumerate(data):
        file_name = file_names_list[i]
        label = labels[i]
        print(
            f"Subject {file_name}: PLV data shape = {plv_data.shape}, Label = {'Friend' if label == 1 else 'Stranger'}")

    # Create pairs and triplets
    pairs, pair_labels, triplets = create_pairs_and_triplets(data, labels)
    Fs = 256  # Example sampling frequency
    LowBand = 4  # original 13 but now 4 frequency so adjusted to 4.
    HighBand = 45  # band upper edge

    # Create dataset and dataloader
    dataset = EEGDataset(pairs, pair_labels, triplets, Fs, LowBand, HighBand)
    batch_size = 76  # Adjust based on your data size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    num_frequency_bands = 4
    num_channels = data[0].shape[1]
    num_features = 128
    #num_samples, num_features = data[0].shape
    num_plv_features = 0
    num_isc_features = num_channels
    time_len = num_channels  # Since PLV matrices are square (channels x channels)

    # Adjust DSENModel to accept PLV data
    model = DSENModel(
        num_channels=num_channels,
        time_len=time_len,
        num_features=num_features,
        num_classes=2,
    ).to(device)

    # Define loss functions
    criterion_classification = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=1.0)
    criterion_cca = CCALoss()

    # Define learning rate
    learning_rate = 1e-4

    # Initialize separate optimizers
    optimizer_f = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
    optimizer_c = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        avg_loss_combined, avg_loss_triplet, f1 = train_model(
            model,
            train_loader,
            optimizer_f,
            optimizer_c,
            criterion_classification,
            criterion_triplet,
            criterion_cca,
            device
        )
        print(f'Epoch [{epoch+1}/{num_epochs}], Combined Loss: {avg_loss_combined:.4f}, '
              f'Triplet Loss: {avg_loss_triplet:.4f}, F1 Score: {f1:.4f}')
    print('Training complete.')
    # Model saving
    torch.save(model.state_dict(), 'model_dsen.pth')
    torch.save(optimizer_f.state_dict(), 'optimizer_f.pth')
    torch.save(optimizer_c.state_dict(), 'optimizer_c.pth')
