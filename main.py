import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.signal import hilbert
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import os


class CNN1Dlocal(nn.Module):
    def __init__(self):
        super(CNN1Dlocal, self).__init__()
        self.conv = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=64)
        self.bn = nn.BatchNorm1d(30)
        self.elu = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(100)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN1Dglobal(nn.Module):
    def __init__(self):
        super(CNN1Dglobal, self).__init__()
        self.conv = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=200)
        self.bn = nn.BatchNorm1d(30)
        self.elu = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(128)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels * 2, out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        batch_size, num_channels, num_steps = x.shape
        x_expanded = x.unsqueeze(2).expand(-1, -1, num_channels, -1)
        x_neighbors = x.unsqueeze(1).expand(-1, num_channels, -1, -1)
        x_concat = torch.cat([x_expanded, x_neighbors], dim=3)
        x_concat = x_concat.view(batch_size * num_channels * num_steps, -1)

        if x_concat.dim() == 3: #modified
            x_concat = x_concat.transpose(1, 2).contiguous()
        x_concat = x_concat.view(-1, num_channels * 2)

        out = self.linear1(x_concat)
        out = self.relu(out)
        out = self.linear2(out)

        out = out.view(batch_size, num_channels, num_steps, -1)
        out = out.max(dim=2)[0]

        return out


class DSENFeatureExtractor(nn.Module):
    def __init__(self):
        super(DSENFeatureExtractor, self).__init__()
        self.cnn_local = CNN1Dlocal()
        self.cnn_global = CNN1Dglobal()

        # Second convolution to reduce concatenated local features to (30, 128)
        self.second_conv = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=3, padding=1)
        self.pool_to_128 = nn.AdaptiveAvgPool1d(128)

        self.edge_conv1 = EdgeConvBlock(2*128, 128)
        self.edge_conv2 = EdgeConvBlock(128, 256)
        self.edge_conv3 = EdgeConvBlock(256, 512)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128 + 256 + 512, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        # Local feature extraction
        local_features = []
        for i in range(9):
            segment = x[:, :, i * 400:(i + 1) * 400]
            local_feat = self.cnn_local(segment)
            local_features.append(local_feat)
        x_local = torch.cat(local_features, dim=2)  # Concatenate 9 segments, shape is [batch_size, 30, 900]

        # Apply second 1D convolution to reduce to (30, 128)
        x_local = self.second_conv(x_local)
        x_local = self.pool_to_128(x_local)  # Now shape is [batch_size, 30, 128]
        print(f"x_local shape after reduction: {x_local.shape}")

        # Global feature extraction
        x_global = self.cnn_global(x)  # x_global is already [batch_size, 30, 128]
        print(f"x_global shape: {x_global.shape}")

        # Concatenate local and global features
        x_concat = torch.cat((x_local, x_global), dim=1)  # Now shapes match for concatenation
        print(f"x_concat shape: {x_concat.shape}")

        # EdgeConv blocks
        h1 = self.edge_conv1(x_concat)
        h1_pool = self.global_maxpool(h1).squeeze(-1)

        h2 = self.edge_conv2(h1)
        h2_pool = self.global_maxpool(h2).squeeze(-1)

        h3 = self.edge_conv3(h2)
        h3_pool = self.global_maxpool(h3).squeeze(-1)

        # Concatenate pooled outputs
        h_concat = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)

        h_concat = h_concat.view(h_concat.size(0), -1) #modified

        # Final fully connected layers
        h_fc1 = self.fc1(h_concat)
        h_final = self.fc2(h_fc1)

        return h_final


class AttentionFusionClassifier(nn.Module):
    def __init__(self):
        super(AttentionFusionClassifier, self).__init__()
        self.W_q = nn.Linear(128, 128)
        self.W_k = nn.Linear(128, 128)
        self.W_v = nn.Linear(128, 128)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, H_X, H_Y):
        Q_X, K_X, V_X = self.W_q(H_X), self.W_k(H_X), self.W_v(H_X)
        Q_Y, K_Y, V_Y = self.W_q(H_Y), self.W_k(H_Y), self.W_v(H_Y)

        attention_X = torch.softmax(torch.matmul(Q_X, K_Y.transpose(-1, -2)) / (128 ** 0.5), dim=-1)
        attention_Y = torch.softmax(torch.matmul(Q_Y, K_X.transpose(-1, -2)) / (128 ** 0.5), dim=-1)

        attended_X = torch.matmul(attention_X, V_Y)
        attended_Y = torch.matmul(attention_Y, V_X)

        fused_features = torch.cat((attended_X, attended_Y), dim=-1)

        out = self.fc1(fused_features)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


"""
class DSEN(nn.Module):
    def __init__(self):
        super(DSEN, self).__init__()
        self.feature_extractor = DSENFeatureExtractor()
        self.classifier = AttentionFusionClassifier()

    def forward(self, X, Y, Z=None):
        H_X = self.feature_extractor(X)
        H_Y = self.feature_extractor(Y)
        preds = self.classifier(H_X, H_Y)

        if Z is not None:
            H_Z = self.feature_extractor(Z)
            return preds, H_X, H_Y, H_Z
        else:
            return preds, H_X, H_Y
"""


class DSEN(nn.Module):
    def __init__(self):
        super(DSEN, self).__init__()
        self.feature_extractor = DSENFeatureExtractor()  # Extract features for X and Y
        self.classifier = AttentionFusionClassifier()  # Classify relationship

    def forward(self, X, Y):
        # Extract features for both friend subjects X and Y
        H_X = self.feature_extractor(X)
        H_Y = self.feature_extractor(Y)

        # Classify based on fused features from X and Y
        preds = self.classifier(H_X, H_Y)

        # No need for H_Z (stranger) for now
        return preds, H_X, H_Y


def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=1)


class EEGDataset(Dataset):
    def __init__(self, mat_files, mat_files_path, n_emotions=9):
        self.data_X = []
        self.data_Y = []
        self.labels = []
        self.n_emotions = n_emotions

        # Load the friend data pair (sub81_0_CSD.mat and sub82_0_CSD.mat)
        for mat_file in mat_files:
            data_path = os.path.join(mat_files_path, mat_file)
            mat_data = loadmat(data_path)

            data = mat_data['data']  # Shape should be (30, time_steps)

            # Split data into X and Y (for the two friends)
            mid_point = data.shape[1] // 2
            data_X = data[:, :mid_point]  # First half for subject X
            data_Y = data[:, mid_point:]  # Second half for subject Y

            # Compute instantaneous amplitude using Hilbert transform
            analytic_signal_X = hilbert(data_X)
            analytic_signal_Y = hilbert(data_Y)

            amplitude_X = np.abs(analytic_signal_X)
            amplitude_Y = np.abs(analytic_signal_Y)

            # Segment the data into n_emotions segments
            segment_length = amplitude_X.shape[1] // self.n_emotions
            for i in range(self.n_emotions):
                start = i * segment_length
                end = (i + 1) * segment_length
                self.data_X.append(torch.FloatTensor(amplitude_X[:, start:end]))
                self.data_Y.append(torch.FloatTensor(amplitude_Y[:, start:end]))

            # Assuming you know the labels beforehand for training (e.g., 1 for friends)
            label = 1  # Since 81 and 82 are friends
            self.labels.extend([label] * self.n_emotions)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        X = self.data_X[idx]
        Y = self.data_Y[idx]
        label = torch.LongTensor([self.labels[idx]])
        return X, Y, label


"""
class EEGDataset(Dataset):
    def __init__(self, mat_files, mat_files_path, n_emotions=9):
        self.data_X = []
        self.data_Y = []
        self.data_Z = []  # Z represents the stranger data
        self.labels = []
        self.n_emotions = n_emotions

        # Load the friend data pair (sub81_0_CSD.mat and sub82_0_CSD.mat)
        all_data = []
        for mat_file in mat_files:
            data_path = os.path.join(mat_files_path, mat_file)
            mat_data = loadmat(data_path)

            data = mat_data['data']  # Shape should be (30, time_steps)
            all_data.append(data)

        # Assuming first two files are friends data (81, 82)
        data_X = all_data[0]  # sub81_0_CSD.mat
        data_Y = all_data[1]  # sub82_0_CSD.mat

        # Compute instantaneous amplitude using Hilbert transform for X and Y
        analytic_signal_X = hilbert(data_X)
        analytic_signal_Y = hilbert(data_Y)
        amplitude_X = np.abs(analytic_signal_X)
        amplitude_Y = np.abs(analytic_signal_Y)

        # Segment the data into n_emotions segments for X and Y
        segment_length = amplitude_X.shape[1] // self.n_emotions
        for i in range(self.n_emotions):
            start = i * segment_length
            end = (i + 1) * segment_length
            self.data_X.append(torch.FloatTensor(amplitude_X[:, start:end]))
            self.data_Y.append(torch.FloatTensor(amplitude_Y[:, start:end]))

        # Assign friendship labels
        label = 1  # Since 81 and 82 are friends
        self.labels.extend([label] * self.n_emotions)

        # Create Z data (stranger) randomly from the data not in (X, Y)
        self.data_Z = self.create_Z_data(all_data)

    def create_Z_data(self, all_data):
        data_Z = []
        for i in range(len(self.data_X)):
            # Randomly select a sample from the data (stranger) not from X or Y
            while True:
                idx = np.random.randint(2, len(all_data))  # Assuming the first two are friends
                data_Z_full = all_data[idx]
                analytic_signal_Z = hilbert(data_Z_full)
                amplitude_Z = np.abs(analytic_signal_Z)

                # Segment the Z data in the same way as X and Y
                segment_length = amplitude_Z.shape[1] // self.n_emotions
                start = (i % self.n_emotions) * segment_length
                end = ((i % self.n_emotions) + 1) * segment_length

                if data_Z_full.shape == self.data_X[i].shape:
                    data_Z.append(torch.FloatTensor(amplitude_Z[:, start:end]))
                    break  # Ensure Z is selected and properly segmented
        return data_Z

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        X = self.data_X[idx]
        Y = self.data_Y[idx]
        Z = self.data_Z[idx]  # Fetch the stranger data (Z)
        label = torch.LongTensor([self.labels[idx]])
        return X, Y, Z, label

"""


def triplet_loss(anchor, positive, negative, margin=0.2):
    dist_p = 1.0 - cosine_similarity(anchor, positive)
    dist_n = 1.0 - cosine_similarity(anchor, negative)
    losses = F.relu(dist_p - dist_n + margin)
    return losses.mean()


# CCA loss function (simplified version)
def cca_loss(H_X, H_Y):
    # Center the matrices
    H_X = H_X - H_X.mean(dim=0)
    H_Y = H_Y - H_Y.mean(dim=0)

    # Compute covariance matrices
    cov_X = torch.mm(H_X.t(), H_X) / (H_X.size(0) - 1)
    cov_Y = torch.mm(H_Y.t(), H_Y) / (H_Y.size(0) - 1)
    cov_XY = torch.mm(H_X.t(), H_Y) / (H_X.size(0) - 1)

    # Compute E matrix
    E = torch.mm(torch.mm(torch.inverse(cov_X.pow(0.5)), cov_XY), torch.inverse(cov_Y.pow(0.5)))

    # Compute CCA loss
    _, s, _ = torch.svd(E)
    return -torch.sum(s)


# Combined loss function
def combined_loss(preds, labels, H_X, H_Y, alpha=1, beta=1, gamma=1):
    classification_loss = F.cross_entropy(preds, labels)
    cca_loss_val = cca_loss(H_X, H_Y)
    #triplet_loss_val = triplet_loss(H_X, H_Y, H_Z)
    return alpha * classification_loss + beta * cca_loss_val #+ gamma * triplet_loss_val


"""
def train_DSEN(data_loader, model, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for X, Y, Z, labels in data_loader:
            optimizer.zero_grad()

            # Forward pass for X, Y, and Z
            preds, H_X, H_Y, H_Z = model(X, Y, Z)

            # Compute triplet loss
            triplet_loss_val = triplet_loss(H_X, H_Y, H_Z)

            # Compute classification loss
            classification_loss = F.cross_entropy(preds, labels)

            # Combine losses (weights alpha, beta can be adjusted)
            loss = classification_loss + triplet_loss_val
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
"""


# Training function
def train_DSEN(data_loader, model, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for X, Y, labels in data_loader:
            optimizer.zero_grad()
            preds, H_X, H_Y = model(X, Y)  # No H_Z
            loss = combined_loss(preds, labels, H_X, H_Y)  # Updated loss function
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Initialize dataset and DataLoader
mat_files_path = '/Users/derrick/PycharmProjects/DSEN'
mat_files = ['sub81_0_CSD.mat', 'sub82_0_CSD.mat']
dataset = EEGDataset(mat_files, mat_files_path)
data_loader = DataLoader(dataset, batch_size=79, shuffle=True)

# Initialize the DSEN model
model = DSEN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train_DSEN(data_loader, model, optimizer, epochs=100)
