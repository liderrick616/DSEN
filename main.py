import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mne
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
mat_files_path = '/Users/derrick/PycharmProjects/DSEN'
mat_files = ['sub24_0_CSD.mat', 'sub25_0_CSD.mat']
datasets = {}

for mat_file in mat_files:
    data_path = os.path.join(mat_files_path, mat_file)
    mat_data = loadmat(data_path)
    dataset_name = os.path.splitext(mat_file)[0]
    datasets[dataset_name] = {
        'data': mat_data['data'],
        'times': mat_data['times'],
        'chanlocs': mat_data['chanlocs'],
        'srate': mat_data['srate'],
        'events': mat_data.get('events', None)  # Use .get() to handle optional event data
    }
for dataset_name, dataset in datasets.items():
    print(f"Dataset: {dataset_name}")
    data_shape = dataset['data'].shape
    print(f"  data (shape: {data_shape}):")
    if len(data_shape) == 2:
        print(dataset['data'][:, :10])  # Print first 10 time points for all channels
    elif len(data_shape) == 3:
        print(dataset['data'][:, :10, 0])  # Print first 10 time points of the first trial for all channels
    else:
        print("Unexpected data shape.")
    # Print time points
    print(f"  times (shape: {dataset['times'].shape}):")
    print(dataset['times'][:10])  # Print the first 10 time points
    # Print channel locations (names)
    print(f"  chanlocs (number of channels: {len(dataset['chanlocs'])}):")
    print(dataset['chanlocs'])  # Print all channel names
    # Print sampling rate
    print(f"  srate: {dataset['srate']}")
    if dataset['events'] is not None:
        print(f"  events:")
        for event in dataset['events']:
            print(event)  # Print each event
    else:
        print("  events: None")
    print("\n")


class CNN1Dlocal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNN1Dlocal, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv(x)))
        return x


class CNN1Dglobal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNN1Dglobal, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv(x)))
        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        # Adjust the Linear layer input size to match the actual size after concatenation
        self.sMLP = nn.Sequential(
            nn.Linear(200, out_channels),  # Use 200 to match the actual concatenated edge_features size
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, adj):
        batch_size, num_vertices, num_features = x.size()
        print(f"Input x size: {x.size()}")  # Debug print

        x_i = x.unsqueeze(2).repeat(1, 1, num_vertices, 1)
        x_j = x.unsqueeze(1).repeat(1, num_vertices, 1, 1)
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)  # Concatenate along the feature dimension

        print(f"Concatenated edge_features size: {edge_features.size()}")  # Debug print

        # Flatten the edge features to (batch_size * num_vertices * num_vertices, concatenated_features)
        edge_features = edge_features.view(-1, edge_features.size(-1))

        print(f"Flattened edge_features size: {edge_features.size()}")  # Debug print

        # Apply the MLP with corrected input size
        edge_features = self.sMLP(edge_features)

        print(f"MLP output size: {edge_features.size()}")  # Debug print

        # Reshape back to (batch_size, num_vertices, num_vertices, out_channels)
        edge_features = edge_features.view(batch_size, num_vertices, num_vertices, -1)

        print(f"Reshaped edge_features size: {edge_features.size()}")  # Debug print

        # Multiply by the adjacency matrix and aggregate
        edge_features = edge_features * adj.unsqueeze(-1)
        H = edge_features.max(dim=2)[0]  # Global max pooling over neighbors
        return H



class EdgeConv1(EdgeConv):
    def __init__(self, in_channels):
        super(EdgeConv1, self).__init__(in_channels, 128)
        #super().__init__(in_channels, 128)


class EdgeConv2(EdgeConv):
    def __init__(self, in_channels):
        super(EdgeConv2, self).__init__(in_channels, 256)
        # Adjust the sMLP input size to 256
        self.sMLP = nn.Sequential(
            nn.Linear(256, 256),  # Match 256, the size of the input features after EdgeConv1
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )


class EdgeConv3(EdgeConv):
    def __init__(self, in_channels):
        super(EdgeConv3, self).__init__(in_channels, 512)
        # Adjust the sMLP input size to 512
        self.sMLP = nn.Sequential(
            nn.Linear(512, 512),  # Match 512, the size of the input features after EdgeConv2
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )


class DSEN(nn.Module):
    def __init__(self):
        super(DSEN, self).__init__()
        self.local_cnn = CNN1Dlocal(in_channels=32, out_channels=64)
        self.global_cnn = CNN1Dglobal(in_channels=64, out_channels=128)

        # EdgeConv layers
        self.edge_conv1 = EdgeConv1(in_channels=128)
        self.edge_conv2 = EdgeConv2(in_channels=128)
        self.edge_conv3 = EdgeConv3(in_channels=128)

        self.linear_projector = nn.Linear(512, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x, adj):
        fea_local = self.local_cnn(x)
        TX = self.global_cnn(fea_local)

        # Adjust TX to match the adjacency matrix size
        TX = TX.permute(0, 2, 1)  # Switch to (batch_size, num_vertices, num_features)

        if TX.size(1) != adj.size(1):
            TX = F.interpolate(TX, size=adj.size(1), mode='nearest')

        TX = TX.permute(0, 2, 1)  # Switch back to (batch_size, num_features, num_vertices)

        H1X = self.edge_conv1(TX, adj)
        H2X = self.edge_conv2(H1X, adj)
        H3X = self.edge_conv3(H2X, adj)

        HX = torch.cat([H1X, H2X, H3X], dim=-1)
        HX_projected = self.linear_projector(HX)

        return HX_projected, HX_projected, HX_projected


def triplet_loss(anchor, positive, negative, margin=1.0):
    dist_p = 1.0 - F.cosine_similarity(anchor, positive)
    dist_n = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(dist_p - dist_n + margin)


def cca_loss(HX, HY):
    HX_centered = HX - HX.mean(dim=0)
    HY_centered = HY - HY.mean(dim=0)

    covariance_matrix = torch.matmul(HX_centered.t(), HY_centered) / (HX.size(0) - 1)
    Rx = torch.matmul(HX_centered.t(), HX_centered) / (HX.size(0) - 1)
    Ry = torch.matmul(HY_centered.t(), HY_centered) / (HX.size(0) - 1)

    Rx_inv = torch.inverse(Rx + 1e-5 * torch.eye(Rx.size(0)))  # Regularization
    Ry_inv = torch.inverse(Ry + 1e-5 * torch.eye(Ry.size(0)))  # Regularization

    E = torch.matmul(torch.matmul(Rx_inv, covariance_matrix), Ry_inv)
    _, S, _ = torch.svd(E)

    return -torch.sum(S)


def combined_loss(HX, HY, HZ, preds, labels, alpha=1.0, beta=1.0):
    l_triplet = triplet_loss(HX, HY, HZ)
    l_cca = cca_loss(HX, HY)
    l_classification = F.cross_entropy(preds, labels)
    return alpha * l_classification + beta * l_cca, l_triplet


def train_dsen(dsen_model, optimizer, data_loader, max_epochs=100):
    for epoch in range(max_epochs):
        for (AX, AY, AZ, adj, labels) in data_loader:  # Unpack 5 values
            optimizer.zero_grad()

            # Pass the adjacency matrix `adj` to the DSEN model
            HX, HY, HZ = dsen_model(AX, adj)

            # Triplet Loss
            l_triplet = triplet_loss(HX, HY, HZ)

            # Classification
            preds = dsen_model.fc(HX + HY)  # Example fusion for classification
            l_combined, _ = combined_loss(HX, HY, HZ, preds, labels)

            # Total Loss
            total_loss = l_triplet + l_combined
            total_loss.backward()
            optimizer.step()


class EEGDataset(Dataset):
    def __init__(self, eeg_data, adjacency_matrices, labels):
        self.eeg_data = eeg_data
        self.adjacency_matrices = adjacency_matrices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        AX = self.eeg_data[idx]
        AY = AX  # Replace with actual AY data if different
        AZ = torch.zeros_like(AX)  # Replace None with a tensor of zeros or use actual AZ data if available
        adj = self.adjacency_matrices[idx]
        label = self.labels[idx]
        return AX, AY, AZ, adj, label  # Return 5 items


# Example data (replace with actual data)
num_samples = 100
num_channels = 32
num_timesteps = 100

eeg_data = torch.randn(num_samples, num_channels, num_timesteps)  # EEG data, 3 parameter, return tensor
adjacency_matrices = torch.eye(num_channels).unsqueeze(0).repeat(num_samples, 1, 1)  # adjacency matrices
labels = torch.randint(0, 2, (num_samples,))  # binary labels, if exist return 1, else 0.

# Create dataset and DataLoader
dataset = EEGDataset(eeg_data, adjacency_matrices, labels) # slice into individual slices, training individually.
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage with the DSEN model
dsen_model = DSEN()
optimizer = torch.optim.Adam(dsen_model.parameters(), lr=1e-3)

# Training loop
train_dsen(dsen_model, optimizer, data_loader)
