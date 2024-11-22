# 1. 导入所需的库
import torch
from Models.PredictMeltingPoint_model import ExtendedAlloyModel
from bayes_opt import BayesianOptimization
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from Dataset.AlloysDataset import AlloysDataset
from torch.utils.data import DataLoader ,random_split
import pandas as pd
feature_cols = ['VEC', 'electronegativity', 'cohesive energy', 'density', 'radius', 'heat of fusion']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 3. 数据加载和预处理
raw_data = pd.read_excel(r"C:\Users\PC\Desktop\train_data(add_feature)test.xlsx")
dataset = AlloysDataset(r"C:\Users\PC\Desktop\element_data.xlsx",additional_features=raw_data[feature_cols].values, data=raw_data)

batch_size = 128
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
    model = ExtendedAlloyModel(num_elements, EMBEDDING_DIM, additional_features_dim, dataset.padding_value,
                               dropout=dropout, num_neurons=num_neurons)
    # ... [其他代码保持不变]
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    no_improvement_count = 0
    patience = 5
    min_delta = 0.001

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for batch_element_ids, batch_element_ratios, batch_additional_features, batch_melting_point_targets, _ in train_loader:
            batch_element_ids = batch_element_ids.to(device)
            batch_element_ratios = batch_element_ratios.to(device)
            batch_additional_features = batch_additional_features.to(device)
            batch_melting_point_targets = batch_melting_point_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
            loss = criterion(outputs, batch_melting_point_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_melting_point_targets.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_r2_score = r2_score(all_labels, all_preds)

        model.eval()
        total_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch_element_ids, batch_element_ratios, batch_additional_features, batch_melting_point_targets, _ in val_loader:
                batch_element_ids = batch_element_ids.to(device)
                batch_element_ratios = batch_element_ratios.to(device)
                batch_additional_features = batch_additional_features.to(device)
                batch_melting_point_targets = batch_melting_point_targets.to(device)

                outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
                loss = criterion(outputs, batch_melting_point_targets)
                total_val_loss += loss.item()
                all_val_preds.extend(outputs.detach().cpu().numpy())
                all_val_labels.extend(batch_melting_point_targets.cpu().numpy())

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

    return -best_val_loss  # We return negative loss because BayesianOptimization maximizes the objective function

# 5. 目标函数
def objective(lr, dropout, num_neurons_1, num_neurons_2, num_neurons_3, num_neurons_4,num_neurons_5):
    return train_model(lr, dropout, [num_neurons_1, num_neurons_2, num_neurons_3, num_neurons_4,num_neurons_5])

# 6. 超参数的范围定义
param_bounds = {
    'lr': (0.000001, 1),
    'dropout': (0.1, 0.5),
    'num_neurons_1': (50, 300),
    'num_neurons_2': (50, 300),
    'num_neurons_3': (50, 300),
    'num_neurons_4': (50, 300),
    'num_neurons_5':(50,300)

}

# 7. 贝叶斯优化过程
optimizer = BayesianOptimization(f=objective, pbounds=param_bounds, random_state=1)
optimizer.maximize(init_points=5, n_iter=95)

print(optimizer.max)