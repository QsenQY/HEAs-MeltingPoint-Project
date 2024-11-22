import torch
import torch.nn as nn

import torch.nn as nn

# Model_1
# class PredictFeatureModel(nn.Module):
#     def __init__(self, num_element, embedding_dim, n_features, padding_value):
#         super(PredictFeatureModel, self).__init__()
#
#         self.embedding = nn.Embedding(num_element, embedding_dim, padding_idx=padding_value)
#         self.flatten = nn.Flatten()
#
#         self.fc1 = nn.Linear(embedding_dim * 5, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.leaky_relu1 = nn.LeakyReLU()
#         self.dropout1 = nn.Dropout(0.5)
#
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.leaky_relu2 = nn.LeakyReLU()
#         self.dropout2 = nn.Dropout(0.5)
#
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.leaky_relu3 = nn.LeakyReLU()
#         self.dropout3 = nn.Dropout(0.5)
#
#         self.fc4 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.leaky_relu4 = nn.LeakyReLU()
#         self.fc5 = nn.Linear(64, n_features)
#
#     def forward(self, element_ids, element_ratios, additional_features=None):
#         embeds = self.embedding(element_ids)
#         combined = embeds * element_ratios.unsqueeze(-1)
#         flattened = self.flatten(combined)
#
#         x = self.fc1(flattened)
#         x = self.bn1(x)
#         x = self.leaky_relu1(x)
#         x = self.dropout1(x)
#
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.leaky_relu2(x)
#         x = self.dropout2(x)
#
#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = self.leaky_relu3(x)
#         x = self.dropout3(x)
#
#         x = self.fc4(x)
#         x = self.bn4(x)
#         x = self.leaky_relu4(x)
#
#         return self.fc5(x)


# Model_2
class PredictEletronegativity(nn.Module):
    def __init__(self, num_elements, embedding_dim, additional_features_dim, padding_value, dropout=0.4327,
                 num_neurons=[60, 252, 73, 81]):
        super(PredictEletronegativity, self).__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim, padding_idx=padding_value)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(embedding_dim * 5, int(num_neurons[0]))
        self.bn1 = nn.BatchNorm1d(int(num_neurons[0]))
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(int(num_neurons[0]), int(num_neurons[1]))
        self.bn2 = nn.BatchNorm1d(int(num_neurons[1]))
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(int(num_neurons[1]), int(num_neurons[2]))
        self.bn3 = nn.BatchNorm1d(int(num_neurons[2]))
        self.fc4 = nn.Linear(int(num_neurons[2]), 1)

    def forward(self, element_ids, element_ratios, additional_features=None):
        embeds = self.embedding(element_ids)
        combined = embeds * element_ratios.unsqueeze(-1)
        flattened = self.flatten(combined)

        x = self.fc1(flattened)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        return self.fc4(x)
#



# Model_3

# class MultiTaskModel(nn.Module):
#     def __init__(self, num_element, embedding_dim, padding_value, num_tasks):
#         super(MultiTaskModel, self).__init__()
#
#         self.embedding = nn.Embedding(num_element, embedding_dim, padding_idx=padding_value)
#         self.flatten = nn.Flatten()
#
#         # Shared layers
#         self.fc1 = nn.Linear(embedding_dim * 5, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(0.3)
#
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.3)
#
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.dropout3 = nn.Dropout(0.3)
#
#         self.fc4 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.dropout4 = nn.Dropout(0.3)
#
#         # Output layers for each task
#         self.output_layers = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])
#
#     def forward(self, element_ids, element_ratios, additional_features=None):
#         embeds = self.embedding(element_ids)
#         combined = embeds * element_ratios.unsqueeze(-1)
#         flattened = self.flatten(combined)
#
#         x = self.fc1(flattened)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.dropout1(x)
#
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
#         x = self.dropout2(x)
#
#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = torch.relu(x)
#         x = self.dropout3(x)
#
#         x = self.fc4(x)
#         x = self.bn4(x)
#         x = torch.relu(x)
#         x = self.dropout4(x)
#
#         # Get outputs for each task
#         outputs = [output_layer(x) for output_layer in self.output_layers]
#
#         return outputs