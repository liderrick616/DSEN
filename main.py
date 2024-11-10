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
import random


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
"""
class CCALoss(nn.Module):
    def __init__(self):
        super(CCALoss, self).__init__()

    def forward(self, H1, H2):
        # r1, r2 are regularization parameters to ensure positive definiteness of matrix
        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-10
        # to center the data
        H1_centered = H1 - H1.mean(dim=0)
        H2_centered = H2 - H2.mean(dim=0)
        # below is Hx adpated and Hy adapted from thesis
        # Cross-covariance matrix between H1 and H2.
        SigmaHat12 = H1_centered.t().mm(H2_centered)
        # Auto-covariance matrix of H1, Σ11 = H1^T * H1 + r1*Identity matrix
        SigmaHat11 = H1_centered.t().mm(H1_centered) + r1 * torch.eye(H1.size(1)).to(H1.device) # 2-D tensor, ones on diagonal, zeros elsewhere
        # same for H2
        SigmaHat22 = H2_centered.t().mm(H2_centered) + r2 * torch.eye(H2.size(1)).to(H2.device)
        # use Cholesky decomposition to perform similar eqn as:
        # max(f) corr(f(Ax), f(Ay)) (eqn: 7)
        try:
            D1 = torch.cholesky(SigmaHat11)
            D2 = torch.cholesky(SigmaHat22)
        except RuntimeError as e:
            print(f"Cholesky decomposition error: {e}")
            D1 = torch.linalg.cholesky(SigmaHat11 + eps * torch.eye(SigmaHat11.size(0)).to(H1.device))
            D2 = torch.linalg.cholesky(SigmaHat22 + eps * torch.eye(SigmaHat22.size(0)).to(H2.device))
        # D1 is Rx, D2 is Ry, sigmaHat12 is RXY (eqn: 8)
        T = torch.inverse(D1).mm(SigmaHat12).mm(torch.inverse(D2).t()) # T is E
        U, S, V = torch.svd(T)  # T = UΣV^T, s is singular value of E derive from SVD of E
        # (eqn: 9)
        corr = torch.sum(S) # Sum of the canonical correlations.
        loss = -corr  # Negative of the total correlation.
        return loss
"""

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

        self.fc1 = Linear(self.hidden_size * 2, self.hidden_size)  # reduce concatenation back to 128 dim
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
        x_fused = torch.cat((fused_1, fused_2), dim=1)  # Shape: (batch_size, hidden_size * 2 = 256)
        # Classification Layers (fully connected)
        x = F.relu(self.fc1(x_fused))  # reduce dim to 128,apply ReLu activation function
        x = F.dropout(x, p=0.25)  # prevent overfitting
        x = self.fc2(x)  # reduce num_classes in x to 2 (1 friends 0 strangers)
        return x  # Logits, unnormalized scores for each class


# Putting it all together
class DSENModel(nn.Module):
    def __init__(self, num_channels=30, time_len=3600):
        super(DSENModel, self).__init__()
        self.encoder = DSEN(num_channels=num_channels, time_len=time_len)
        self.classifier = RelationClassifier()

    def forward(self, x1, x2):
        # Encode both inputs
        h_x1 = self.encoder(x1)
        h_x2 = self.encoder(x2)

        # Classification
        logits = self.classifier(h_x1, h_x2)

        return logits, h_x1, h_x2


# slide 14
# Load your data from .mat files
def load_eeg_data_mat(data_dir, file_names, friend_ids):
    data = []
    labels = []
    min_length = None  # Keep track of the minimum length

    for file_name in file_names:
        # loop and load files and construct file path
        mat_file = os.path.join(data_dir, file_name)
        mat = loadmat(mat_file)
        # extract EEG data
        eeg_data = mat['data']

        # convert to float32
        eeg_data = eeg_data.astype(np.float32)

        # Transpose so correct EGG data shape
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        # Update minimum length
        if min_length is None or eeg_data.shape[1] < min_length:
            min_length = eeg_data.shape[1]

        # Extract subject ID from file name
        subject_id = int(file_name.split('_')[0][3:])

        # Assign labels based on subject IDs
        if subject_id in friend_ids:
            label = 1  # Friend
        else:
            label = 0  # Stranger
        # Appends the EEG data and the corresponding label to their respective lists.
        data.append(eeg_data)
        labels.append(label)

        # Print data shape and label for verification
        print(f"Loaded {file_name}: Subject ID = {subject_id}, Assigned Label = {'Friend' if label == 1 else 'Stranger'}, EEG data shape = {eeg_data.shape}")

    # Truncate all data to the minimum length
    for i in range(len(data)):
        data[i] = data[i][:, :min_length]  # Truncate to min_length

    return data, labels


# Slide 14
# Create pairs and triplets
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


# Slide 14
class EEGDataset(Dataset):
    def __init__(self, pairs, pair_labels, triplets):
        self.pairs = pairs
        self.pair_labels = pair_labels
        self.triplets = triplets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.pair_labels[idx]
        # For triplet loss, pick a random triplet
        triplet_idx = random.randint(0, len(self.triplets) - 1)
        anchor, positive, negative = self.triplets[triplet_idx]

        # Convert data to torch tensors and flatten
        x1 = torch.from_numpy(x1).float()  # Shape: (num_channels, time_len)
        x2 = torch.from_numpy(x2).float()
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()

        label = torch.tensor(label).long()

        return x1, x2, anchor, positive, negative, label


# slide 13
def train_model(model, train_loader, optimizer_f, optimizer_c, criterion_classification, criterion_triplet, criterion_cca, device):
    model.train()
    total_loss_combined = 0
    total_loss_triplet = 0
    all_labels = []
    all_preds = []
    for batch in train_loader:
        # Unpack the batch
        x1, x2, anchor, positive, negative, label = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        label = label.to(device)
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
        logits, h_x1, h_x2 = model(x1, x2)
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
        print(f'Loss Classification: {loss_classification.item():.4f}, '
              f'Loss CCA: {loss_cca.item():.4f}, '
              f'Loss Triplet: {loss_triplet.item():.4f}, '
              f'Loss Combined: {loss_combined.item():.4f}')
    # average combined and triplet losses over entire epoch
    avg_loss_combined = total_loss_combined / len(train_loader)
    avg_loss_triplet = total_loss_triplet / len(train_loader)
    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss_combined, avg_loss_triplet, f1


if __name__ == '__main__':
    # Device configuration
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data from .mat files
    #data_dir = '/Users/derrick/PycharmProjects/DSEN'
    data_dir = '/home/derrick/PycharmProjects/DSEN'

    # Define friend IDs
    friend_ids = [55, 61, 62, 63, 64, 65, 66, 80, 81, 82, 95, 96, 97, 98, 101, 102]

    # List of files
    friend_files = ['sub61_1_CSD.mat','sub61_4_CSD.mat','sub61_5_CSD.mat','sub61_6_CSD.mat','sub61_7_CSD.mat','sub61_9_CSD.mat','sub62_1_CSD.mat','sub62_4_CSD.mat','sub62_5_CSD.mat','sub62_6_CSD.mat','sub62_7_CSD.mat','sub62_9_CSD.mat','sub63_1_CSD.mat','sub63_4_CSD.mat','sub63_5_CSD.mat','sub63_6_CSD.mat','sub63_7_CSD.mat','sub63_9_CSD.mat','sub64_1_CSD.mat','sub64_4_CSD.mat','sub64_5_CSD.mat','sub64_6_CSD.mat','sub64_7_CSD.mat','sub64_9_CSD.mat','sub65_1_CSD.mat','sub65_4_CSD.mat','sub65_5_CSD.mat','sub65_6_CSD.mat','sub65_7_CSD.mat','sub65_9_CSD.mat','sub66_1_CSD.mat','sub66_4_CSD.mat','sub66_5_CSD.mat','sub66_6_CSD.mat','sub66_7_CSD.mat','sub66_9_CSD.mat','sub80_0_CSD.mat','sub81_0_CSD.mat','sub82_0_CSD.mat','sub95_1_CSD.mat','sub95_4_CSD.mat','sub95_5_CSD.mat','sub95_6_CSD.mat','sub95_7_CSD.mat','sub95_9_CSD.mat','sub96_1_CSD.mat','sub96_4_CSD.mat','sub96_5_CSD.mat','sub96_6_CSD.mat','sub96_7_CSD.mat','sub96_9_CSD.mat','sub97_1_CSD.mat','sub97_4_CSD.mat','sub97_5_CSD.mat','sub97_6_CSD.mat','sub97_7_CSD.mat','sub97_9_CSD.mat', 'sub98_1_CSD.mat']
    stranger_files = ['sub24_1_CSD.mat','sub24_4_CSD.mat','sub24_5_CSD.mat','sub24_6_CSD.mat','sub24_7_CSD.mat','sub24_9_CSD.mat','sub25_1_CSD.mat','sub25_4_CSD.mat','sub25_5_CSD.mat','sub25_6_CSD.mat','sub25_7_CSD.mat','sub25_9_CSD.mat','sub26_1_CSD.mat','sub26_4_CSD.mat','sub26_5_CSD.mat','sub26_6_CSD.mat','sub26_7_CSD.mat','sub26_9_CSD.mat','sub27_1_CSD.mat','sub27_4_CSD.mat','sub27_5_CSD.mat','sub27_6_CSD.mat','sub27_7_CSD.mat','sub27_9_CSD.mat', 'sub28_1_CSD.mat','sub28_4_CSD.mat']
    file_names = friend_files + stranger_files

    # Load EEG data and labels
    data, labels = load_eeg_data_mat(data_dir, file_names, friend_ids)

    # Check data shapes and labels
    for i, eeg_data in enumerate(data):
        print(f"Subject {file_names[i]}: EEG data shape = {eeg_data.shape}, Label = {'Friend' if labels[i] == 1 else 'Stranger'}")

    # Create pairs and triplets
    pairs, pair_labels, triplets = create_pairs_and_triplets(data, labels)

    # Create dataset and dataloader
    dataset = EEGDataset(pairs, pair_labels, triplets)
    batch_size = 79  # Adjust based on your data size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    num_channels, time_len = data[0].shape
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
