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
from scipy.signal import firwin, filtfilt, hilbert
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
    def __init__(self, num_features=128, time_len=3600, num_channels=30, num_segments=9):
        super(DSEN, self).__init__()
        self.num_features = num_features
        self.time_len = time_len
        self.num_channels = num_channels
        self.num_segments = num_segments
# individually apply feature extraction module of DSEN to EEG data
# 2 1D conv layer, input 30 channel, kernel 64 AND 200, 9 segment for each video
        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=64,
                groups=1,  # Set to 1 for standard convolution.
                bias=False, # b/c batch normalization is used
                padding=32  # Padding of 32 time steps to maintain the size of the output
            ),
            nn.BatchNorm1d(self.num_channels),  # batch normalization
            nn.ELU(),  # activation function Exponential linear units, introduce non linearity
            nn.AdaptiveAvgPool1d(100)  # pooling to size of 100 for each local feature
        )
        # same as above
        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=200,
                groups=1,
                bias=False,
                padding=100
            ),
            nn.BatchNorm1d(self.num_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(128)
        )
        # Spatial Feature Extraction (DGCNN)
        # Edge conv layer 1, 2, 3, pooled as global max, output increased to 512 dimension.
        self.conv1 = EdgeConv(Sequential(Linear(2 * 128, 128), ReLU(), Linear(128, 128), ReLU(), BatchNorm1d(128), Dropout(p=0.25)))
        self.conv2 = EdgeConv(Sequential(Linear(2 * 128, 256), ReLU(), Linear(256, 256), ReLU(), BatchNorm1d(256), Dropout(p=0.25)))
        self.conv3 = EdgeConv(Sequential(Linear(2 * 256, 512), ReLU(), Linear(512, 512), ReLU(), BatchNorm1d(512), Dropout(p=0.25)))
        # Fully connected layers to combine features and reduce dimensions to 128
        self.linear1 = Linear(128 + 256 + 512, 256)
        self.linear2 = Linear(256, 128)

# define the input data flow
    def forward(self, x):
        batch_size = x.size(0)
        # reshapes the dimensions
        x = x.view(batch_size, self.num_channels, self.time_len)
        segment_len = self.time_len // self.num_segments
        # divide EEG data 9 equal parts, each with dimension of 2
        # "2-second sliding window to obtain segments from 9 EEG recordings"
        segments = torch.split(x, segment_len, dim=2)
        segment_features = []
        for idx, segment in enumerate(segments):
            out = self.block_1(segment)
            segment_features.append(out)
        # all out has shape (batch_size, num_channels, 100) due to adapative pooling
        x = torch.cat(segment_features, dim=2)
        # Concatenates the processed segments along the time dimension
        x = self.block_2(x)
        # Flatten batch and channel dimensions for graph processing
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
    def __init__(self, num_features=128, num_classes=2, num_plv_features=30, num_isc_features=30):
        super(RelationClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_plv_features = num_plv_features
        self.num_isc_features = num_isc_features

        self.hidden_size = num_features  # size 128, dimensionality of the embeddings
        self.scale = self.hidden_size ** 0.5  # constant scale factor
        # Attention Mechanism weights
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for query vectors.
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for key vectors.
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)  # linear transformation for value vectors.

        # self.fc1 = Linear(self.hidden_size * 2, self.hidden_size)  # reduce concatenation back to 128 dim
        self.fc1 = nn.Linear(self.hidden_size * 2 + self.num_plv_features + self.num_isc_features, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, num_classes)  # maps to logits (friends or strangers)

    def forward(self, x1, x2, plv_features, isc_features):  # equation (5) from the thesis
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
        x_fused = torch.cat((fused_1, fused_2, plv_features, isc_features), dim=1)
        # Classification Layers (fully connected)
        x = F.relu(self.fc1(x_fused))  # reduce dim to 128,apply ReLu activation function
        x = F.dropout(x, p=0.25)  # prevent overfitting
        x = self.fc2(x)  # reduce num_classes in x to 2 (1 friends 0 strangers)
        return x  # Logits, unnormalized scores for each class



class PLVCalculator:
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
            pass_zero='bandpass'  # or false, depend on the version
        )

    def compute_plv(self, sig1, sig2):
        # Apply the filter
        X = filtfilt(self.bfil, [1.0], sig1.astype(np.float64))
        Y = filtfilt(self.bfil, [1.0], sig2.astype(np.float64))

        # Compute the analytic signals
        hX = hilbert(X)
        hY = hilbert(Y)

        # Extract instantaneous phases
        angleX = np.angle(hX)
        angleY = np.angle(hY)

        # Compute phase differences
        deltaXY = np.unwrap(angleX) - np.unwrap(angleY)

        # Compute PLV
        expdelta = np.exp(1j * deltaXY)
        plv = np.abs(np.sum(expdelta)) / len(deltaXY)

        return plv

    def compute_plv_multichannel(self, eeg1, eeg2):
        num_channels = eeg1.shape[0]
        plv_features = []
        for ch in range(num_channels):
            sig1 = eeg1[ch, :]
            sig2 = eeg2[ch, :]
            plv = self.compute_plv(sig1, sig2)
            plv_features.append(plv)
        return np.array(plv_features)


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
        # Apply the filter
        X = filtfilt(self.bfil, [1.0], sig1.astype(np.float64))
        Y = filtfilt(self.bfil, [1.0], sig2.astype(np.float64))

        # Compute the analytic signals
        hX = hilbert(X)
        hY = hilbert(Y)
        # Extract instantaneous amplitudes
        amplitudeX = np.abs(hX)
        amplitudeY = np.abs(hY)
        # Compute ISC (Pearson correlation coefficient)
        if np.std(amplitudeX) == 0 or np.std(amplitudeY) == 0:
            isc = 0.0  # Avoid division by zero
        else:
            isc = np.corrcoef(amplitudeX, amplitudeY)[0, 1]
        return isc
    def compute_isc_multichannel(self, eeg1, eeg2):
        num_channels = eeg1.shape[0]
        isc_features = []
        for ch in range(num_channels):
            sig1 = eeg1[ch, :]
            sig2 = eeg2[ch, :]
            isc = self.compute_isc(sig1, sig2)
            isc_features.append(isc)
        return np.array(isc_features)


# Putting it all together
class DSENModel(nn.Module):

    # def __init__(self, num_channels=30, time_len=3600):
        # super(DSENModel, self).__init__()
        # self.encoder = DSEN(num_channels=num_channels, time_len=time_len)
        # self.classifier = RelationClassifier()

    def __init__(self, num_channels=30, time_len=3600, num_features=128, num_classes=2, num_plv_features=30, num_isc_features=30):
        super(DSENModel, self).__init__()
        self.encoder = DSEN(num_channels=num_channels, time_len=time_len)
        self.classifier = RelationClassifier(
            num_features=num_features,
            num_classes=num_classes,
            num_plv_features=num_plv_features,
            num_isc_features=num_isc_features
        )

    def forward(self, x1, x2, plv_features, isc_features):
        # Encode both inputs
        h_x1 = self.encoder(x1)
        h_x2 = self.encoder(x2)
        # Classification
        logits = self.classifier(h_x1, h_x2, plv_features, isc_features)
        return logits, h_x1, h_x2


def extract_subject_id(file_name):
    import re
    match = re.search(r'^(\d+)_CleanData\.mat$', file_name)
    if match:
        return int(match.group(1))
    else:
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
    min_length = None  # Keep track of the minimum length
    for file_name in file_names:
        # Construct file path
        mat_file = os.path.join(data_dir, file_name)

        # Load the .mat file
        mat = loadmat(mat_file)

        # Extract 'data_all' structure from the .mat file
        data_all = mat['data_all']

        # Access 'trial' field
        trials = data_all['trial'][0, 0]  # Adjust indices as needed
        num_trials = trials.shape[1]
        # Extract subject ID from file name
        subject_id = extract_subject_id(file_name)

        # Assign label based on subject IDs
        if subject_id is not None and subject_id in friend_ids:
            label = 1  # Friend
        else:
            label = 0  # Stranger

        for i in range(num_trials):
            # Access each trial
            trial_data = trials[0, i]

            # Use the last 1000 ms (time points 1000 to 1999)
            trial_data = trial_data[:, 999:]  # Adjust indices for zero-based indexing

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
                f"Loaded {file_name}, Trial {i + 1}: Subject ID = {subject_id}, Assigned Label = {'Friend' if label == 1 else 'Stranger'}, EEG data shape = {trial_data.shape}")

    # Truncate all data to the minimum length
    for i in range(len(data)):
        data[i] = data[i][:, :min_length]  # Truncate to min_length

    return data, labels


# Slide 14
# Create pairs and triplets
"""
def create_pairs_and_triplets(data, labels):
    # Create pairs for classification
    pairs = []  # store pairs of EEG data samples.
    pair_labels = []  # store labels for each pair (1 for friends, 0 for strangers).
    num_subjects = len(data)
    # pair creation, add to pair_labels list
    for i in range(num_subjects):
        for j in range(i + 1, num_subjects):
            pairs.append((data[i], data[j]))
            if labels[i] == labels[j] == 1:
                pair_labels.append(1)  # Friends
            else:
                pair_labels.append(0)  # Strangers

    # Create triplets for triplet loss
    triplets = []  # store triplets of EEG data samples.
    for i in range(num_subjects):
        anchor = data[i]  # current data sample
        label = labels[i]  # label of the anchor sample

        # Find positive samples (same label)
        positive_indices = [idx for idx, l in enumerate(labels) if l == label and idx != i]
        if not positive_indices:
            continue  # Skip if no positive sample

        positive = data[random.choice(positive_indices)]
        # Creates a list of indices where the label is different from the anchor's label.
        # Find negative samples (different label)
        negative_indices = [idx for idx, l in enumerate(labels) if l != label]
        if not negative_indices:
            continue  # Skip if no negative sample

        negative = data[random.choice(negative_indices)]  # Retrieves the negative sample using the selected index.

        triplets.append((anchor, positive, negative))

    return pairs, pair_labels, triplets
"""


def create_pairs_and_triplets(data, labels, num_pairs=1000, num_triplets=1000): # 6000, 6000 to get batch = 76
    pairs = []
    pair_labels = []
    triplets = []

    # Group data indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    # Generate positive and negative pairs
    for _ in range(num_pairs // 2):
        # Positive pair (same label)
        label = random.choice(list(label_to_indices.keys()))
        idx1, idx2 = random.sample(label_to_indices[label], 2)
        pairs.append((data[idx1], data[idx2]))
        pair_labels.append(1)

        # Negative pair (different labels)
        label1, label2 = random.sample(list(label_to_indices.keys()), 2)
        idx1 = random.choice(label_to_indices[label1])
        idx2 = random.choice(label_to_indices[label2])
        pairs.append((data[idx1], data[idx2]))
        pair_labels.append(0)

    # Generate triplets
    for _ in range(num_triplets):
        # Anchor and positive (same label)
        label = random.choice(list(label_to_indices.keys()))
        anchor_idx, positive_idx = random.sample(label_to_indices[label], 2)
        # Negative (different label)
        negative_label = random.choice([l for l in label_to_indices.keys() if l != label])
        negative_idx = random.choice(label_to_indices[negative_label])
        triplets.append((data[anchor_idx], data[positive_idx], data[negative_idx]))

    return pairs, pair_labels, triplets
# Slide 14
class EEGDataset(Dataset):
    def __init__(self, pairs, pair_labels, triplets, Fs, LowBand, HighBand):
        self.pairs = pairs
        self.pair_labels = pair_labels
        self.triplets = triplets
        # Initialize the PLVCalculator
        self.plv_calculator = PLVCalculator(Fs, LowBand, HighBand)
        self.isc_calculator = ISCCalculator(Fs, LowBand, HighBand)
        self.plv_features_list = []
        self.isc_features_list = []
        for idx, (x1, x2) in enumerate(self.pairs):
            plv_features = self.plv_calculator.compute_plv_multichannel(x1, x2)
            isc_features = self.isc_calculator.compute_isc_multichannel(x1, x2)
            self.plv_features_list.append(torch.from_numpy(plv_features).float())
            self.isc_features_list.append(torch.from_numpy(isc_features).float())
            if idx % 100 == 0:
                print(f"Computed features for {idx + 1}/{len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.pair_labels[idx]
        # For triplet loss, pick a random triplet
        triplet_idx = random.randint(0, len(self.triplets) - 1)
        anchor, positive, negative = self.triplets[triplet_idx]

        # Compute PLV features
        #plv_features = self.plv_calculator.compute_plv_multichannel(x1, x2)
        #isc_features = self.isc_calculator.compute_isc_multichannel(x1, x2)
        #plv_features = torch.from_numpy(plv_features).float()
        #isc_features = torch.from_numpy(isc_features).float()
        #self.plv_features_list.append(plv_features)
        #self.isc_features_list.append(isc_features)

        # Convert data to torch tensors and flatten
        plv_features = self.plv_features_list[idx]
        isc_features = self.isc_features_list[idx]
        x1 = torch.from_numpy(x1).float()  # Shape: (num_channels, time_len)
        x2 = torch.from_numpy(x2).float()
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()
        label = torch.tensor(label).long()

        return x1, x2, anchor, positive, negative, label, plv_features, isc_features


# slide 13
def train_model(model, train_loader, optimizer_f, optimizer_c, criterion_classification, criterion_triplet, criterion_cca, device):
    model.train()
    total_loss_combined = 0
    total_loss_triplet = 0
    all_labels = []
    all_preds = []


    plv_features_list = []
    isc_features_list = []


    for batch_idx, batch in enumerate(train_loader):
        # Unpack the batch
        #x1, x2, anchor, positive, negative, label = batch
        x1, x2, anchor, positive, negative, label, plv_features, isc_features = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        label = label.to(device)

        plv_features = plv_features.to(device)
        isc_features = isc_features.to(device)


        plv_features_list.extend(plv_features.cpu().numpy())
        isc_features_list.extend(isc_features.cpu().numpy())


        # Update encoder's parameter θ_f using triplet loss L_triplet, generate embedding from input data
        h_anchor = model.encoder(anchor)
        h_positive = model.encoder(positive)
        h_negative = model.encoder(negative)
        # Compute Triplet Loss using criterion
        loss_triplet = criterion_triplet(h_anchor, h_positive, h_negative)
        # eqn for criterion L(h_anchor, h_positive, h_negative) = max{d(h_anchor(i), h_negative(i)) - d(h_anchor(i),h_negative(i)) + margin,0}
        # Updates the encoder parameters θ_f to minimize triplet loss
        optimizer_f.zero_grad()  # Clears gradients of encoder's parameters
        loss_triplet.backward()  # Computes gradients of triplet loss with respect to encoder's parameters
        optimizer_f.step()  # Updates encoder's parameters based on computed gradients.
        # Update θ_f and θ_c using L_combined
        #logits, h_x1, h_x2 = model(x1, x2)
        logits, h_x1, h_x2 = model(x1, x2, plv_features, isc_features)
        # Compute Classification Loss
        loss_classification = criterion_classification(logits, label)
        # Compute CCA Loss
        loss_cca = criterion_cca(h_x1, h_x2)
        # slide 9
        # Compute Combined Loss
        alpha = 1.0  # Weight for classification loss
        beta = 1.0  # Weight for CCA loss
        loss_combined = alpha * loss_classification - beta * loss_cca  # Subtract if loss_cca is negative
        # Backward and optimize θ_f and θ_c, similar as above
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        loss_combined.backward()
        optimizer_f.step()
        optimizer_c.step()
        # Adds current batch's losses to total losses for epoch.
        total_loss_combined += loss_combined.item()
        total_loss_triplet += loss_triplet.item()
        # Collect predictions and labels for F1 score
        # Determines predicted class by selecting index with highest logit value
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Print individual losses
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss Classification: {loss_classification.item():.4f}, '
                  f'Loss CCA: {loss_cca.item():.4f}, '
                  f'Loss Triplet: {loss_triplet.item():.4f}, '
                  f'Loss Combined: {loss_combined.item():.4f}')
    # average combined and triplet losses over entire epoch
    avg_loss_combined = total_loss_combined / len(train_loader)
    avg_loss_triplet = total_loss_triplet / len(train_loader)
    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    np.save('plv_features_train.npy', np.array(plv_features_list))
    np.save('isc_features_train.npy', np.array(isc_features_list))

    return avg_loss_combined, avg_loss_triplet, f1


"""
def collect_files(file_names, data_directories):
    full_paths = []
    for dir_path in data_directories:
        for file_name in file_names:
            # Construct the full search pattern
            search_pattern = os.path.join(dir_path, file_name)
            # Find all matching files
            matched_files = glob.glob(search_pattern)
            full_paths.extend(matched_files)
    return full_paths
"""



if __name__ == '__main__':
    # Device configuration
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data from .mat files
    data_dir = '/home/derrick/PycharmProjects/datayx/hyperemotion1/subA'

    """
        '/home/derrick/PycharmProjects/datayx/hyperemotion4/subA',
        '/home/derrick/PycharmProjects/datayx/hyperemotion5/subA',
        '/home/derrick/PycharmProjects/datayx/hyperemotion6/subA',
        '/home/derrick/PycharmProjects/datayx/hyperemotion7/subA',
        '/home/derrick/PycharmProjects/datayx/hyperemotion9/subA'
    """
    #data_dir = '/home/derrick/PycharmProjects/DSEN'

    friend_ids = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,
     47,48,49,50,51,52,53,54,57,58,59,60,67,68,69,70,73,74,75,77,78,79,80,81,82,83,84,85,86,105,106,107,108,111,112,
     131,132,137,138,143,144,153,154,159,160,161,162,179,180,42,55,56,61,62,63,64,65,66,71,72,87,88,89,90,91,92,93,94,
                  99,100,109,110,113,114,115,116,121,122,127,128,129,130,133,134,135,136,139,140,141,142,145,146,147,
                  148,149,150151,152,155,156,157,158,163,164,165,166,167,168,169,170,175,176,177,178,189,190,199,200,
                  201,202,203,204,205,206]
    Fs = 256  # Example sampling frequency
    LowBand = 13  # Beta band lower edge
    HighBand = 45  # gamma band upper edge

    file_names = [
        '101_CleanData.mat', '102_CleanData.mat', '103_CleanData.mat', '105_CleanData.mat',
        '106_CleanData.mat', '107_CleanData.mat', '108_CleanData.mat', '109_CleanData.mat', '110_CleanData.mat',
        '111_CleanData.mat', '113_CleanData.mat', '114_CleanData.mat', '115_CleanData.mat', '116_CleanData.mat',
        '117_CleanData.mat', '118_CleanData.mat', '119_CleanData.mat', '120_CleanData.mat', '121_CleanData.mat',
        '122_CleanData.mat', '123_CleanData.mat', '124_CleanData.mat', '125_CleanData.mat', '126_CleanData.mat',
        '127_CleanData.mat', '128_CleanData.mat', '129_CleanData.mat', '130_CleanData.mat', '131_CleanData.mat',
        '132_CleanData.mat', '133_CleanData.mat', '134_CleanData.mat', '135_CleanData.mat', '136_CleanData.mat',
        '137_CleanData.mat', '138_CleanData.mat', '139_CleanData.mat', '140_CleanData.mat', '141_CleanData.mat',
        '142_CleanData.mat', '143_CleanData.mat', '144_CleanData.mat', '145_CleanData.mat', '146_CleanData.mat',
        '147_CleanData.mat', '148_CleanData.mat', '149_CleanData.mat', '150_CleanData.mat', '151_CleanData.mat',
        '152_CleanData.mat', '153_CleanData.mat', '154_CleanData.mat', '155_CleanData.mat', '156_CleanData.mat',
        '157_CleanData.mat', '158_CleanData.mat', '159_CleanData.mat', '160_CleanData.mat', '161_CleanData.mat',
        '162_CleanData.mat', '163_CleanData.mat', '164_CleanData.mat', '165_CleanData.mat', '166_CleanData.mat',
        '167_CleanData.mat', '168_CleanData.mat', '169_CleanData.mat', '170_CleanData.mat', '171_CleanData.mat',
        '172_CleanData.mat', '175_CleanData.mat', '176_CleanData.mat', '177_CleanData.mat', '178_CleanData.mat',
        '179_CleanData.mat', '180_CleanData.mat', '181_CleanData.mat', '182_CleanData.mat', '183_CleanData.mat',
        '184_CleanData.mat', '185_CleanData.mat', '186_CleanData.mat', '187_CleanData.mat', '188_CleanData.mat',
        '189_CleanData.mat', '190_CleanData.mat', '191_CleanData.mat', '192_CleanData.mat', '193_CleanData.mat',
        '194_CleanData.mat', '195_CleanData.mat', '196_CleanData.mat', '197_CleanData.mat', '198_CleanData.mat',
        '199_CleanData.mat', '200_CleanData.mat', '201_CleanData.mat', '202_CleanData.mat', '203_CleanData.mat',
        '204_CleanData.mat', '23_CleanData.mat', '24_CleanData.mat', '25_CleanData.mat', '26_CleanData.mat',
        '27_CleanData.mat', '28_CleanData.mat', '29_CleanData.mat', '30_CleanData.mat', '31_CleanData.mat',
        '34_CleanData.mat', '35_CleanData.mat', '36_CleanData.mat', '37_CleanData.mat', '38_CleanData.mat',
        '39_CleanData.mat', '41_CleanData.mat', '42_CleanData.mat', '43_CleanData.mat', '44_CleanData.mat',
        '47_CleanData.mat', '48_CleanData.mat', '49_CleanData.mat', '50_CleanData.mat', '51_CleanData.mat',
        '52_CleanData.mat', '53_CleanData.mat', '54_CleanData.mat', '55_CleanData.mat', '56_CleanData.mat',
        '57_CleanData.mat', '58_CleanData.mat', '59_CleanData.mat', '60_CleanData.mat', '61_CleanData.mat',
        '62_CleanData.mat', '63_CleanData.mat', '64_CleanData.mat', '65_CleanData.mat', '66_CleanData.mat',
        '67_CleanData.mat', '68_CleanData.mat', '69_CleanData.mat', '70_CleanData.mat', '71_CleanData.mat',
        '72_CleanData.mat', '73_CleanData.mat', '74_CleanData.mat', '75_CleanData.mat', '76_CleanData.mat',
        '77_CleanData.mat', '78_CleanData.mat', '79_CleanData.mat', '80_CleanData.mat', '81_CleanData.mat',
        '82_CleanData.mat', '83_CleanData.mat', '84_CleanData.mat', '85_CleanData.mat', '86_CleanData.mat',
        '87_CleanData.mat', '88_CleanData.mat', '92_CleanData.mat', '93_CleanData.mat', '95_CleanData.mat',
        '96_CleanData.mat', '97_CleanData.mat', '98_CleanData.mat'
    ]

    #file_names = collect_files(file_name, data_dir)

    # Load EEG data and labels
    #data, labels = load_eeg_data_mat(data_dir, file_names, friend_ids)
    data, labels = load_eeg_data_mat(data_dir, file_names, friend_ids)
    # Check data shapes and labels
    #for i, eeg_data in enumerate(data):
        #print(
            #f"Subject {file_names[i]}: EEG data shape = {eeg_data.shape}, Label = {'Friend' if labels[i] == 1 else 'Stranger'}")
    # Create pairs and triplets
    file_names_per_trial = []
    for file_name in file_names:
        mat_file = os.path.join(data_dir, file_name)
        mat = loadmat(mat_file)
        data_all = mat['data_all']
        trials = data_all['trial'][0, 0]
        num_trials = trials.shape[1]
        # Append the file name for each trial
        file_names_per_trial.extend([file_name] * num_trials)

    # Verify that lengths match
    assert len(data) == len(labels) == len(
        file_names_per_trial), "Mismatch in lengths of data, labels, and file_names_per_trial"

    # Print data information
    for i, eeg_data in enumerate(data):
        file_name = file_names_per_trial[i]
        label = labels[i]
        print(
            f"Subject {file_name}: EEG data shape = {eeg_data.shape}, Label = {'Friend' if label == 1 else 'Stranger'}")

    pairs, pair_labels, triplets = create_pairs_and_triplets(data, labels)

    # Create dataset and dataloader
    dataset = EEGDataset(pairs, pair_labels, triplets, Fs, LowBand, HighBand)
    batch_size = 79  # Adjust based on your data size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    # num_channels, time_len = data[0].shape
    num_channels, time_len = data[0].shape
    num_plv_features = num_channels
    num_isc_features = num_channels
    model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)

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
    # Model saving slide 14
    torch.save(model.state_dict(), 'model_dsen.pth')
    torch.save(optimizer_f.state_dict(), 'optimizer_f.pth')
    torch.save(optimizer_c.state_dict(), 'optimizer_c.pth')
    # separate into 6 different .pth file to store
