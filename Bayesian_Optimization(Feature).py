# 1. 导入所需的库
import torch
from Models.PredictVEC_model import PredictVEC
from Models.PredictEletronegativity_model import PredictEletronegativity
from Models.PredictTC_model import PredictTC
from Models.PredictTEC_model import PredictTEC
from Models.PredictDensity_model import PredictDensity
from Models.PredictCohesive_model import PredictCohesive
from bayes_opt import BayesianOptimization
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from Dataset.AlloysDataset import AlloysDataset2
from torch.utils.data import DataLoader ,random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AlloysDataset2('/path', target_feature='density')

batch_size = 64
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

EMBEDDING_DIM = 8
num_elements = len(dataset.element_to_id) + 1
additional_features_dim = dataset.additional_features.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(lr, dropout, num_neurons, epochs=20):
    model = PredictDensity(num_elements=num_elements,
                       embedding_dim=EMBEDDING_DIM,
                       additional_features_dim=additional_features_dim,
                       padding_value=dataset.padding_value,
                       dropout=dropout,
                       num_neurons=num_neurons)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    no_improvement_count = 0
    patience = 3
    min_delta = 0.001

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for batch_element_ids, batch_element_ratios, batch_additional_features, batch_targets in train_loader:
            batch_element_ids = batch_element_ids.to(device)
            batch_element_ratios = batch_element_ratios.to(device)
            batch_additional_features = batch_additional_features.to(device)
            batch_targets = batch_targets.to(device).view(-1, 1)  # Ensure it's [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_targets.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_r2_score = r2_score(all_labels, all_preds)

        model.eval()
        total_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch_element_ids, batch_element_ratios, batch_additional_features, batch_targets in val_loader:
                batch_element_ids = batch_element_ids.to(device)
                batch_element_ratios = batch_element_ratios.to(device)
                batch_additional_features = batch_additional_features.to(device)
                batch_targets = batch_targets.to(device).view(-1, 1)  # Ensure it's [batch_size, 1]

                outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
                loss = criterion(outputs, batch_targets)
                total_val_loss += loss.item()
                all_val_preds.extend(outputs.detach().cpu().numpy())
                all_val_labels.extend(batch_targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_r2_score = r2_score(all_val_labels, all_val_preds)
        scheduler.step(avg_val_loss)

        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count > patience:
            break

    return -best_val_loss

# 5. 目标函数
def objective(lr, dropout, num_neurons_1, num_neurons_2, num_neurons_3, num_neurons_4):
    return train_model(lr, dropout, [num_neurons_1, num_neurons_2, num_neurons_3, num_neurons_4])

# 6. 超参数的范围定义
param_bounds = {
    'lr': (0.00001, 0.1),
    'dropout': (0.1, 0.5),
    'num_neurons_1': (50, 300),
    'num_neurons_2': (50, 300),
    'num_neurons_3': (50, 300),
    'num_neurons_4': (20, 100)
}


optimizer = BayesianOptimization(f=objective, pbounds=param_bounds, random_state=1)
optimizer.maximize(init_points=5, n_iter=25)

print(optimizer.max)
