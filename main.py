import numpy as np
import os
from scipy.io import loadmat
#from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import EdgeConv, global_max_pool
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import random

# Helper function to create a fully connected edge index
def create_fully_connected_edge_index(num_nodes):
    edge_index = torch.tensor(list(combinations(range(num_nodes), 2)), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).t()
    return edge_index

# Triplet Loss Implementation
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, dist_func='cosine'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.dist_func = dist_func

    def forward(self, anchor, positive, negative):
        if self.dist_func == 'euclidean':
            distance_positive = F.pairwise_distance(anchor, positive)
            distance_negative = F.pairwise_distance(anchor, negative)
        elif self.dist_func == 'cosine':
            distance_positive = 1.0 - F.cosine_similarity(anchor, positive)
            distance_negative = 1.0 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported dist_func: {self.dist_func}")
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# CCA Loss Implementation
class CCALoss(nn.Module):
    def __init__(self):
        super(CCALoss, self).__init__()

    def forward(self, H1, H2):
        o1 = H1.size(0)
        o2 = H2.size(0)

        H1 = H1 - H1.mean(dim=0)
        H2 = H2 - H2.mean(dim=0)

        S12 = torch.mm(H1.t(), H2) / (o1 - 1)
        S11 = torch.mm(H1.t(), H1) / (o1 - 1) + 1e-4 * torch.eye(H1.size(1)).to(H1.device)
        S22 = torch.mm(H2.t(), H2) / (o2 - 1) + 1e-4 * torch.eye(H2.size(1)).to(H2.device)

        S11_inv = torch.inverse(S11)
        S22_inv = torch.inverse(S22)

        T = torch.mm(torch.mm(S11_inv, S12), S22_inv)
        corr = torch.trace(torch.mm(T, S12.t()))
        loss = -corr  # We want to maximize correlation, so minimize negative correlation
        return loss

# DSEN Model Implementation
class DSEN(nn.Module):
    def __init__(self, num_features=128, time_len=3600, num_channels=30, num_segments=9):
        super(DSEN, self).__init__()
        self.num_features = num_features
        self.time_len = time_len
        self.num_channels = num_channels
        self.num_segments = num_segments

        # Temporal Feature Extraction
        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,  # Changed from 1 to self.num_channels
                out_channels=self.num_channels,  # You can adjust out_channels as needed
                kernel_size=64,
                groups=1,
                bias=False,
                padding=32
            ),
            nn.BatchNorm1d(self.num_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(100)
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_channels,  # Changed from 1 to self.num_channels
                out_channels=self.num_channels,  # Adjust out_channels as needed
                kernel_size=200,
                groups=1,
                bias=False,
                padding=100
            ),
            nn.BatchNorm1d(self.num_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(128)
        )

        # Rest of the model remains the same...


        # Spatial Feature Extraction (DGCNN)
        self.conv1 = EdgeConv(Sequential(Linear(2 * 128, 128), ReLU(), Linear(128, 128), ReLU(), BatchNorm1d(128), Dropout(p=0.25)))
        self.conv2 = EdgeConv(Sequential(Linear(2 * 128, 256), ReLU(), Linear(256, 256), ReLU(), BatchNorm1d(256), Dropout(p=0.25)))
        self.conv3 = EdgeConv(Sequential(Linear(2 * 256, 512), ReLU(), Linear(512, 512), ReLU(), BatchNorm1d(512), Dropout(p=0.25)))

        self.linear1 = Linear(128 + 256 + 512, 256)
        self.linear2 = Linear(256, 128)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_channels, self.time_len)
        #print(f"Input shape after reshape: {x.shape}")  # Should be (batch_size, 30, 3600)
        segment_len = self.time_len // self.num_segments  # Should be 400
        #print(f"Segment length: {segment_len}")
        segments = torch.split(x, segment_len, dim=2)
        #print(f"Number of segments: {len(segments)}")  # Should be 9

        segment_features = []
        for idx, segment in enumerate(segments):
            #print(f"Segment {idx} shape before block_1: {segment.shape}")  # (batch_size, 30, 400)
            out = self.block_1(segment)
            #print(f"Segment {idx} shape after block_1: {out.shape}")  # (batch_size, 30, 100)
            segment_features.append(out)

        x = torch.cat(segment_features, dim=2)
        #print(f"Shape after concatenation: {x.shape}")  # (batch_size, 30, 900)

        x = self.block_2(x)
        #print(f"Shape after block_2: {x.shape}")  # (batch_size, 30, 128)

        # Flatten batch and channel dimensions for graph processing
        x = x.view(batch_size * self.num_channels, -1)  # Shape: (batch_size * num_channels, 128)

        # Create edge_index for a single graph
        edge_index = create_fully_connected_edge_index(self.num_channels)  # Shape: [2, num_edges]

        # Adjust edge_index for batching
        edge_indices = []
        for i in range(batch_size):
            offset = i * self.num_channels
            edge_index_i = edge_index + offset
            edge_indices.append(edge_index_i)

        edge_index = torch.cat(edge_indices, dim=1).to(x.device)  # Concatenate along the second dimension

        # Create batch tensor
        batch = torch.arange(batch_size).unsqueeze(1).repeat(1, self.num_channels).view(-1).to(x.device)

        # EdgeConv layers
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Global pooling
        x1_pooled = global_max_pool(x1, batch)
        x2_pooled = global_max_pool(x2, batch)
        x3_pooled = global_max_pool(x3, batch)

        # Concatenate pooled features
        out = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)

        # Fully connected layers
        out = F.relu(self.linear1(out))
        out = F.dropout(out, p=0.25)
        out = F.relu(self.linear2(out))

        return out  # Shape: (batch_size, 128)


# Relation Classifier with Attention Mechanism
class RelationClassifier(nn.Module):
    def __init__(self, num_features=128, num_classes=2):
        super(RelationClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.hidden_size = num_features
        self.scale = self.hidden_size ** 0.5

        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc1 = Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, num_classes)

    def forward(self, x1, x2):
        # Attention Mechanism
        Q_x1 = self.W_q(x1).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        K_x2 = self.W_k(x2).unsqueeze(1)
        V_x2 = self.W_v(x2).unsqueeze(1)

        Q_x2 = self.W_q(x2).unsqueeze(1)
        K_x1 = self.W_k(x1).unsqueeze(1)
        V_x1 = self.W_v(x1).unsqueeze(1)

        score_1 = torch.matmul(Q_x1, K_x2.transpose(-2, -1)) / self.scale  # Shape: (batch_size, 1, 1)
        score_2 = torch.matmul(Q_x2, K_x1.transpose(-2, -1)) / self.scale

        attention_w_1 = F.softmax(score_1, dim=-1)
        attention_w_2 = F.softmax(score_2, dim=-1)

        fused_1 = torch.matmul(attention_w_1, V_x2).squeeze(1)  # Shape: (batch_size, hidden_size)
        fused_2 = torch.matmul(attention_w_2, V_x1).squeeze(1)

        # Concatenate the fused features
        x_fused = torch.cat((fused_1, fused_2), dim=1)  # Shape: (batch_size, hidden_size * 2)

        # Classification Layers
        x = F.relu(self.fc1(x_fused))
        x = F.dropout(x, p=0.25)
        x = self.fc2(x)

        return x  # Logits

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

# Load your data from .mat files
def load_eeg_data_mat(data_dir, file_names):
    data = []
    labels = []
    min_length = None  # Keep track of the minimum length

    for file_name in file_names:
        mat_file = os.path.join(data_dir, file_name)
        mat = loadmat(mat_file)
        keys = mat.keys()
        print(f"Keys in {file_name}: {mat.keys()}")  # For debugging
        # Access the EEG data
        eeg_data = mat['data']  # Use the appropriate key for your EEG data

        # Ensure data is of type float32
        eeg_data = eeg_data.astype(np.float32)

        # Transpose if necessary
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        # Update minimum length
        if min_length is None or eeg_data.shape[1] < min_length:
            min_length = eeg_data.shape[1]

        # Extract subject ID from file name
        subject_id = int(file_name.split('_')[0][3:])

        # Assign labels based on subject IDs
        if subject_id in [81, 82, 80]:
            label = 1  # Friends
        else:
            label = 0  # Stranger

        data.append(eeg_data)
        labels.append(label)

        # Print data shape for verification
        #print(f"Loaded {file_name}: EEG data shape = {eeg_data.shape}")
    # Truncate all data to the minimum length
    for i in range(len(data)):
        data[i] = data[i][:, :min_length]  # Truncate to min_length

    return data, labels


# Create pairs and triplets
def create_pairs_and_triplets(data, labels):
    # Create pairs for classification
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

    # Create triplets for triplet loss
    triplets = []
    for i in range(num_subjects):
        anchor = data[i]
        label = labels[i]

        # Find positive samples (same label)
        positive_indices = [idx for idx, l in enumerate(labels) if l == label and idx != i]
        if not positive_indices:
            continue  # Skip if no positive sample

        positive = data[random.choice(positive_indices)]

        # Find negative samples (different label)
        negative_indices = [idx for idx, l in enumerate(labels) if l != label]
        if not negative_indices:
            continue  # Skip if no negative sample

        negative = data[random.choice(negative_indices)]

        triplets.append((anchor, positive, negative))

    return pairs, pair_labels, triplets


# Custom Dataset
class EEGDataset(Dataset):
    def __init__(self, pairs, pair_labels, triplets):
        self.pairs = pairs
        self.pair_labels = pair_labels
        self.triplets = triplets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        #print(f"x1 shape: {x1.shape}")
        #print(f"x2 shape: {x2.shape}")
        label = self.pair_labels[idx]
        #print(f"x1 shape: {x1.shape}")
        #print(f"x2 shape: {x2.shape}")
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


# Training function
def train_model(model, train_loader, optimizer, criterion_classification, criterion_triplet, criterion_cca, device):
    model.train()
    total_loss = 0
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

        # Forward pass
        logits, h_anchor, h_positive = model(x1, x2)
        h_negative = model.encoder(negative)

        # Compute losses
        loss_classification = criterion_classification(logits, label)
        loss_triplet = criterion_triplet(h_anchor, h_positive, h_negative)
        loss_cca = criterion_cca(h_anchor, h_positive)
        alpha = 1.0  # Weight for classification loss
        beta = 1.0  # Weight for triplet loss
        gamma = 1.0  # Weight for CCA loss

        loss = alpha * loss_classification + beta * loss_triplet + gamma * loss_cca

        # Combined loss
        #loss = loss_classification + loss_triplet + loss_cca

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect predictions and labels for F1 score
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(train_loader)
    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1
"""
def train_model(model, train_loader, optimizer, criterion_classification, criterion_triplet, criterion_cca, device):
    model.train()
    total_loss = 0
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

        # Forward pass through the feature extractor
        h_anchor = model.encoder(anchor)
        h_positive = model.encoder(positive)
        h_negative = model.encoder(negative)

        # Compute triplet loss and update feature extractor parameters
        loss_triplet = criterion_triplet(h_anchor, h_positive, h_negative)
        optimizer.zero_grad()
        loss_triplet.backward()
        optimizer.step()

        # Forward pass for classification and CCA loss
        h_x1 = model.encoder(x1)
        h_x2 = model.encoder(x2)

        # Compute CCA loss
        loss_cca = criterion_cca(h_x1, h_x2)

        # Attention fusion and classification
        logits = model.classifier(h_x1, h_x2)

        # Compute classification loss
        loss_classification = criterion_classification(logits, label)

        # Compute combined loss
        alpha = 1.0
        beta = 1.0
        loss_combined = alpha * loss_classification + beta * loss_cca

        # Backward and optimize
        optimizer.zero_grad()
        loss_combined.backward()
        optimizer.step()

        total_loss += loss_combined.item()

        # Collect predictions and labels for F1 score
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(train_loader)
    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1
"""
# Main training loop
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cpu')
    # Load data from .mat files
    data_dir = '/Users/derrick/PycharmProjects/DSEN'

    friend_files = ['sub80_0_CSD.mat', 'sub81_0_CSD.mat', 'sub82_0_CSD.mat']
    stranger_files = ['sub23_0_CSDtest(1).mat', 'sub24_0_CSD.mat', 'sub25_0_CSD.mat','sub27_1_CSD.mat','sub27_4_CSD.mat']
    file_names = friend_files + stranger_files

    #file_names = ['sub81_0_CSD.mat', 'sub82_0_CSD.mat', 'sub24_0_CSD.mat']
    data, labels = load_eeg_data_mat(data_dir, file_names)

    # Check data shapes
    for i, eeg_data in enumerate(data):
        print(f"Subject {file_names[i]}: EEG data shape = {eeg_data.shape}")

    # Create pairs and triplets
    pairs, pair_labels, triplets = create_pairs_and_triplets(data, labels)

    # Create dataset and dataloader
    dataset = EEGDataset(pairs, pair_labels, triplets)
    batch_size = 79  # Adjust based on your data size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """
    num_classes = 2
    labels = pair_labels  # Use the labels from your pairs
    class_counts = Counter(labels)
    print(f"Class counts: {class_counts}")
    total_samples = sum(class_counts.values())
    class_weights = []
    for class_index in range(num_classes):
        class_count = class_counts[class_index]
        weight = total_samples / (num_classes * class_count)
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    criterion_classification = nn.CrossEntropyLoss(weight=class_weights)
    """
    num_channels, time_len = data[0].shape
    model = DSENModel(num_channels=num_channels, time_len=time_len).to(device)
    # Define loss functions and optimizer
    criterion_classification = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=1.0)
    criterion_cca = CCALoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 100  # Adjust as needed

    # Training loop
    for epoch in range(num_epochs):
        avg_loss, f1 = train_model(model, train_loader, optimizer, criterion_classification, criterion_triplet, criterion_cca, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}')

    print('Training complete.')
