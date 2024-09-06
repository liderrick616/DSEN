import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import os


# Define the CNN layers for temporal feature extraction
class CNN1Dlocal(nn.Module):
    def __init__(self):
        super(CNN1Dlocal, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=64)
        self.bn1 = nn.BatchNorm1d(30)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool1d(kernel_size=64)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        return x


# Global 1D convolutional layer
class CNN1Dglobal(nn.Module):
    def __init__(self):
        super(CNN1Dglobal, self).__init__()
        self.conv = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=200)
        self.bn = nn.BatchNorm1d(30)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool1d(kernel_size=200)
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
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # x has shape (batch_size, channels, time_steps)
        batch_size, num_channels, num_steps = x.shape

        # Reshape for Linear layer
        x = x.transpose(1, 2).contiguous()  # Shape becomes (batch_size, time_steps, channels)
        #x = x.view(-1, num_channels)  # Flatten to (batch_size * time_steps, channels)
        x = x.view(batch_size * num_steps, num_channels)  # Flatten to (batch_size * time_steps, channels)

        # Apply Linear transformations
        out = self.linear1(x)  # Linear expects 2D input
        out = self.relu(out)
        out = self.linear2(out)

        # Reshape back to 3D (batch_size, out_channels, num_steps)
        out = out.view(batch_size, num_steps, -1)
        out = out.transpose(1, 2).contiguous()  # Return shape (batch_size, out_channels, num_steps)

        return out


# DSEN feature extractor
class DSENFeatureExtractor(nn.Module):
    def __init__(self):
        super(DSENFeatureExtractor, self).__init__()

        # Use CNN1Dlocal and CNN1Dglobal for temporal feature extraction
        self.cnn_local = CNN1Dlocal()
        self.cnn_global = CNN1Dglobal()

        # EdgeConv blocks with the correct input/output sizes as per the diagram
        self.edge_conv1 = EdgeConvBlock(2*128, 128)  # First block
        self.edge_conv2 = EdgeConvBlock(128, 256)  # Second block
        self.edge_conv3 = EdgeConvBlock(256, 512)  # Third block

        # Global max pooling to ensure the features are reduced to a fixed size
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)  # Pooling to a single value per channel

        # Final linear layers for feature concatenation
        self.fc1 = nn.Linear(128 + 256 + 512, 256)  # Concatenation of pooled outputs
        self.fc2 = nn.Linear(256, 128)  # Reducing to final feature size (128)

        # Adaptive pooling to make sure x_local and x_global have the same time dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)  # Fixed size (e.g., 128)

    def forward(self, x):
        # Apply CNN1Dlocal for local feature extraction
        x_local = self.cnn_local(x)
        print(f"x_local shape: {x_local.shape}")  # Debugging

        # Apply CNN1Dglobal for global feature extraction
        x_global = self.cnn_global(x)
        print(f"x_global shape: {x_global.shape}")  # Debugging

        # Apply adaptive pooling to ensure the time dimensions match
        x_local = self.adaptive_pool(x_local)
        x_global = self.adaptive_pool(x_global)
        print(f"x_local shape after pooling: {x_local.shape}")  # Debugging
        print(f"x_global shape after pooling: {x_global.shape}")  # Debugging

        # Concatenate local and global features for the EdgeConv input
        x_concat = torch.cat((x_local, x_global), dim=1)  # Concatenate along the channel dimension
        print(f"x_concat shape: {x_concat.shape}")  # Debugging

        # Apply EdgeConv blocks and global pooling
        h1 = self.edge_conv1(x_concat)
        print(f"h1 shape: {h1.shape}")  # Debugging
        h1_pool = self.global_maxpool(h1)  # First pooled output
        print(f"h1_pool shape: {h1_pool.shape}")  # Debugging

        h2 = self.edge_conv2(h1)
        print(f"h2 shape: {h2.shape}")  # Debugging
        h2_pool = self.global_maxpool(h2)  # Second pooled output
        print(f"h2_pool shape: {h2_pool.shape}")  # Debugging

        h3 = self.edge_conv3(h2)
        print(f"h3 shape: {h3.shape}")  # Debugging
        h3_pool = self.global_maxpool(h3)  # Third pooled output
        print(f"h3_pool shape: {h3_pool.shape}")  # Debugging
        # Concatenate the outputs from the three blocks
        h_concat = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)

        # Flatten the tensor correctly before passing to Linear layers
        h_concat = h_concat.flatten(1)  # Flatten from dimension 1 onwards

        # Debugging output to check shape before passing to fc1
        print(f"h_concat shape (after flattening): {h_concat.shape}")

        # Pass through fully connected layers
        h_fc1 = self.fc1(h_concat)
        h_final = self.fc2(h_fc1)

        return h_final


# Attention mechanism for the classifier
class AttentionFusionClassifier(nn.Module):
    def __init__(self):
        super(AttentionFusionClassifier, self).__init__()
        self.W_q = nn.Linear(128, 128)
        self.W_k = nn.Linear(128, 128)
        self.W_v = nn.Linear(128, 128)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 2)  # 2 classes (stranger or friend)
        )

    def forward(self, H_X, H_Y):
        # Attention mechanism
        Q_X = self.W_q(H_X)
        K_Y = self.W_k(H_Y)
        V_Y = self.W_v(H_Y)
        attention_weights_X = torch.softmax(torch.matmul(Q_X, K_Y.transpose(-1, -2)) / 128 ** 0.5, dim=-1)
        attended_X = torch.matmul(attention_weights_X, V_Y)

        # Reciprocal for Y
        Q_Y = self.W_q(H_Y)
        K_X = self.W_k(H_X)
        V_X = self.W_v(H_X)
        attention_weights_Y = torch.softmax(torch.matmul(Q_Y, K_X.transpose(-1, -2)) / 128 ** 0.5, dim=-1)
        attended_Y = torch.matmul(attention_weights_Y, V_X)

        # Concatenate attended features
        fused_features = torch.cat((attended_X, attended_Y), dim=-1)

        # Classification
        output = self.fc(fused_features)
        return output


# Define the full DSEN model
class DSEN(nn.Module):
    def __init__(self):
        super(DSEN, self).__init__()
        self.feature_extractor = DSENFeatureExtractor()
        self.classifier = AttentionFusionClassifier()

    def forward(self, X, Y):
        H_X = self.feature_extractor(X)
        H_Y = self.feature_extractor(Y)
        preds = self.classifier(H_X, H_Y)
        return preds


class EEGDataset(Dataset):
    def __init__(self, mat_files, mat_files_path):
        self.data_X = []
        self.data_Y = []
        self.labels = []  # Assuming you have labels somewhere

        for mat_file in mat_files:
            data_path = os.path.join(mat_files_path, mat_file)
            mat_data = loadmat(data_path)

            data = mat_data['data']  # Assuming 'data' contains (30, time_steps)

            # Ensure that both X and Y have the full 30 channels
            half = data.shape[1] // 2  # Split based on time steps or samples, NOT channels
            data_X = data[:, :half]  # First half of time steps for X
            data_Y = data[:, half:]  # Second half of time steps for Y

            self.data_X.append(data_X)
            self.data_Y.append(data_Y)

            # You would need to populate 'labels' from the .mat files if available
            self.labels.append(1)  # Modify this as per your actual label data

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        X = self.data_X[idx]
        Y = self.data_Y[idx]
        label = self.labels[idx]  # Assuming you have labels
        return X, Y, label


# Define the CCA loss function
def cca_loss(H_X, H_Y):
    # Mean center the features
    H_X_centered = H_X - H_X.mean(dim=0)
    H_Y_centered = H_Y - H_Y.mean(dim=0)

    # Covariance matrices
    cov_X = torch.mm(H_X_centered.T, H_X_centered) / H_X.size(0)
    cov_Y = torch.mm(H_Y_centered.T, H_Y_centered) / H_Y.size(0)
    cross_cov_XY = torch.mm(H_X_centered.T, H_Y_centered) / H_X.size(0)

    # Calculate CCA loss based on trace of canonical correlations
    E = torch.mm(torch.mm(torch.inverse(cov_X), cross_cov_XY), torch.inverse(cov_Y))
    s = torch.svd(E)[1]  # Singular values
    L_cca = -torch.trace(torch.mm(s.T, s)) ** 0.5
    return L_cca


# Define the combined loss function
def combined_loss(preds, labels, H_X, H_Y, alpha=1, beta=1):
    criterion = nn.CrossEntropyLoss()
    L_classification = criterion(preds, labels)
    L_cca = cca_loss(H_X, H_Y)
    L_combined = alpha * L_classification + beta * L_cca
    return L_combined


# Example of initializing and training the DSEN model
def train_DDSN(data_loader, model, optimizer, epochs=100):
    for epoch in range(epochs):
        for X, Y, labels in data_loader:
            preds = model(X, Y)
            loss = combined_loss(preds, labels, X, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Initialize dataset and DataLoader
mat_files_path = '/Users/derrick/PycharmProjects/DSEN'
mat_files = ['sub81_0_CSD.mat', 'sub82_0_CSD.mat']
dataset = EEGDataset(mat_files, mat_files_path)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the DSEN model
model = DSEN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train_DDSN(data_loader, model, optimizer)
