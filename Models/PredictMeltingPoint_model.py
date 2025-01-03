import torch.nn as nn
import torch
import torch.nn.functional as F



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class ExtendedAlloyModel(nn.Module):
    def __init__(self, num_elements, embedding_dim, additional_features_dim, padding_value, dropout=0.45396618,
                 num_neurons=[267, 178, 290, 98]):
        super(ExtendedAlloyModel, self).__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim, padding_idx=padding_value)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * 5 + additional_features_dim, int(num_neurons[0]))
        self.bn1 = nn.BatchNorm1d(int(num_neurons[0]))
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(int(num_neurons[0]), int(num_neurons[1]))
        self.bn2 = nn.BatchNorm1d(int(num_neurons[1]))
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(int(num_neurons[1]), int(num_neurons[2]))
        self.bn3 = nn.BatchNorm1d(int(num_neurons[2]))
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear((int(num_neurons[2])), int(num_neurons[3]))
        self.bn4 = nn.BatchNorm1d(int(num_neurons[3]))

        self.fc5 = nn.Linear(int(num_neurons[3]), 1)

    def forward(self, element_ids, element_ratios, additional_features):
        embeds = self.embedding(element_ids)
        combined = embeds * element_ratios.unsqueeze(-1)
        flattened = self.flatten(combined)

        x = torch.cat((flattened, additional_features), dim=1)
        x = Swish()(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = Swish()(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = Swish()(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = Swish()(self.bn4(self.fc4(x)))

        return self.fc5(x)

