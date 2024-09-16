import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.signal import hilbert
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import EdgeConv, global_max_pool
from sklearn.metrics import f1_score
import os


class GateFusion(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(GateFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f = x1 * x2
        gate = self.gate(x1)
        out = gate * f
        return out


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
            print(f"Segment shape: {segment.shape}")  # Debugging
            local_feat = self.cnn_local(segment)
            print(f"Local feature shape: {local_feat.shape}")  # Debugging
            local_features.append(local_feat)
        x_local = torch.cat(local_features, dim=2)  # Concatenate along the time dimension
        if x_local.size(2) < 128:
            # Pad to make the time dimension match 128
            padding = 128 - x_local.size(2)
            x_local = F.pad(x_local, (0, padding), "constant", 0)
        print(f"x_local shape after concatenation: {x_local.shape}")  # Debugging

        # Global feature extraction
        x_global = self.cnn_global(x)
        print(f"x_global shape: {x_global.shape}")  # Debugging

        # Concatenate local and global features
        x_concat = torch.cat((x_local, x_global), dim=1)  # Concatenate along the channel dimension
        print(f"x_concat shape: {x_concat.shape}")  # Debugging

        # EdgeConv blocks
        h1 = self.edge_conv1(x_concat)
        h1_pool = self.global_maxpool(h1).squeeze(-1)

        h2 = self.edge_conv2(h1)
        h2_pool = self.global_maxpool(h2).squeeze(-1)

        h3 = self.edge_conv3(h2)
        h3_pool = self.global_maxpool(h3).squeeze(-1)

        # Concatenate pooled outputs
        h_concat = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)

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


class DSEN(nn.Module):
    def __init__(self, num_features=128, time_len=3600, num_channels=30, num_segments=9):
        super(DSEN, self).__init__()
        self.num_features = num_features
        self.time_len = time_len
        self.num_channels = num_channels
        self.num_segments = num_segments

        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=30,
                out_channels=30,
                kernel_size=32,
                groups=30,
                bias=False,
                padding=(32 - 1) // 2
            ),
            nn.BatchNorm1d(30),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(100)
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=30,
                out_channels=30,
                kernel_size=100,
                bias=False,
                groups=30,
            ),
            nn.BatchNorm1d(30),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(128)
        )

        self.conv1 = EdgeConv(Sequential(Linear(2 * 128, 128), ReLU(), Linear(128, 128), ReLU(), BatchNorm1d(128), Dropout(p=0.2)))
        self.conv2 = EdgeConv(Sequential(Linear(2 * 128, 256), ReLU(), Linear(256, 256), ReLU(), BatchNorm1d(256), Dropout(p=0.2)))
        self.conv3 = EdgeConv(Sequential(Linear(2 * 256, 512), ReLU(), Linear(512, 512), ReLU(), BatchNorm1d(512), Dropout(p=0.2)))

        self.linear1 = Linear(128 + 256 + 512, 256)
        self.linear2 = Linear(256, 128)

    def forward(self, x, edge_index, batch):
        x = x.view(-1, self.num_channels, self.time_len)

        # Process each of the 9 segments (from 3600 time points, split into 9)
        segments = torch.split(x, 200, dim=2)
        out_seg_list = []
        for segment_x in segments:
            out_seg_list.append(self.block_1(segment_x))

        # Concatenate and apply global block
        x = torch.cat(out_seg_list, dim=2)
        x = self.block_2(x)
        x = F.dropout(x, p=0.25)

        # Apply EdgeConv
        x1 = self.conv1(x, edge_index)
        x1_pooled = global_max_pool(x1, batch)
        x2 = self.conv2(x1, edge_index)
        x2_pooled = global_max_pool(x2, batch)
        x3 = self.conv3(x2, edge_index)
        x3_pooled = global_max_pool(x3, batch)

        # Fully connected layers
        out = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        return out


class RelationClassifier(nn.Module):
    def __init__(self, encoder, num_channels=30, num_classes=2, num_features=448):
        super(RelationClassifier, self).__init__()
        self.encoder = encoder
        self.num_features = num_features
        self.num_channels = num_channels
        self.input_size = 256
        self.hidden_size = self.input_size // 2
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        self.relation_conv = EdgeConv(Sequential(Linear(2 * self.num_features, self.input_size), ReLU()))

    def forward(self, x1, x2, edge_index=None, batch=None, c2c_index=None):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x1 = x1.view(-1, self.num_channels, self.num_features)
        x2 = x2.view(-1, self.num_channels, self.num_features)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(-1, self.num_features)
        x = self.relation_conv(x, c2c_index)
        new_batch = torch.repeat_interleave(batch, 2)
        x = global_max_pool(x, new_batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=1)


class EEGDataset(Dataset):
    def __init__(self, friend_files, stranger_files, mat_files_path, n_emotions=9):
        self.data_X = []
        self.data_Y = []
        self.data_Z = []  # For triplet loss (stranger data)
        self.labels = []
        self.n_emotions = n_emotions

        # Load friend data
        for mat_file in friend_files:
            data_path = os.path.join(mat_files_path, mat_file)
            mat_data = loadmat(data_path)
            data = mat_data['data']  # Shape should be (30, time_steps)
            mid_point = data.shape[1] // 2
            data_X = data[:, :mid_point]
            data_Y = data[:, mid_point:]

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

            # Add friend label (assuming 1 means friends)
            self.labels.extend([1] * self.n_emotions)

        # Load stranger data (negative samples)
        self.data_Z = []
        for mat_file in stranger_files:
            data_path = os.path.join(mat_files_path, mat_file)
            mat_data = loadmat(data_path)
            data = mat_data['data']  # Shape should be (30, time_steps)
            amplitude_Z = np.abs(hilbert(data))

            # Segment the data into n_emotions segments for the stranger
            segment_length = amplitude_Z.shape[1] // self.n_emotions
            for i in range(self.n_emotions):
                start = i * segment_length
                end = (i + 1) * segment_length
                self.data_Z.append(torch.FloatTensor(amplitude_Z[:, start:end]))

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        X = self.data_X[idx]  # Anchor
        Y = self.data_Y[idx]  # Positive (Friend)
        Z = self.data_Z[idx % len(self.data_Z)]  # Negative (Stranger, randomly selected)
        label = torch.LongTensor([self.labels[idx]])  # Label for anchor-positive pair (e.g., 1 for friends)
        return X, Y, Z, label


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
def combined_loss(preds, labels, H_X, H_Y,H_Z=None, alpha=1, beta=1, gamma=1):
    classification_loss = F.cross_entropy(preds, labels)
    cca_loss_val = cca_loss(H_X, H_Y)
    if H_Z is not None:
        triplet_loss_val = triplet_loss(H_X, H_Y, H_Z)
    else:
        triplet_loss_val = 0  # No triplet loss if H_Z is not provided
    return alpha * classification_loss + beta * cca_loss_val + gamma * triplet_loss_val


def train_DSEN(data_loader, model, optimizer, epochs=100):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_classification_loss = 0
        total_cca_loss = 0
        total_triplet_loss = 0
        all_preds = []
        all_labels = []

        for X, Y, Z, labels in data_loader:
            optimizer.zero_grad()

            # Forward pass
            preds, H_X, H_Y, H_Z = model(X, Y, Z)

            # Compute the combined loss
            classification_loss = F.cross_entropy(preds, labels)
            cca_loss_val = cca_loss(H_X, H_Y)
            triplet_loss_val = triplet_loss(H_X, H_Y, H_Z)
            loss = classification_loss + cca_loss_val + triplet_loss_val

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Aggregate losses for the epoch
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_cca_loss += cca_loss_val.item()
            total_triplet_loss += triplet_loss_val.item()

            # Convert predicted probabilities to class labels
            predicted_labels = torch.argmax(preds, dim=1)
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute the average losses for the epoch
        avg_loss = total_loss / len(data_loader)
        avg_classification_loss = total_classification_loss / len(data_loader)
        avg_cca_loss = total_cca_loss / len(data_loader)
        avg_triplet_loss = total_triplet_loss / len(data_loader)

        # Calculate F1-score for the epoch
        f1 = f1_score(all_labels, all_preds, average='weighted')  # 'weighted' accounts for label imbalance

        # Log the results for this epoch
        print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.3f}, "
              f"Classification Loss: {avg_classification_loss:.3f}, "
              f"CCA Loss: {avg_cca_loss:.3f}, Triplet Loss: {avg_triplet_loss:.3f}, "
              f"F1-Score: {f1:.3f}")


# Initialize dataset and DataLoader
mat_files_path = '/Users/derrick/PycharmProjects/DSEN'
mat_files = ['sub81_0_CSD.mat', 'sub82_0_CSD.mat', 'sub34_0_CSD.mat']
dataset = EEGDataset(mat_files, mat_files_path)
data_loader = DataLoader(dataset, batch_size=79, shuffle=True)

# Initialize the DSEN model
model = DSEN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train_DSEN(data_loader, model, optimizer, epochs=100)
