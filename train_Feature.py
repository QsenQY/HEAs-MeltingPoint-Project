from Dataset.AlloysDataset import AlloysDataset2
from Models.PredictTC_model import PredictTC
from Models.PredictDensity_model import PredictDensity
from Models.PredictTEC_model import PredictTEC
from Models.PredictCohesive_model import PredictCohesive
from Models.PredictEletronegativity_model import PredictEletronegativity
from Models.PredictVEC_model import PredictVEC
from Utils.EarlyStopping import EarlyStopping
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from math import sqrt

# 设置随机种子数
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def safe_format(value, format_spec):
    try:
        return format_spec.format(value)
    except:
        return 'N/A'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_epoch_results(epoch, EPOCHS, feature_names, train_losses, train_r2s, val_losses, val_r2s):
    print(f"Epoch {epoch}/{EPOCHS}:")
    print("---------------------------------------------------------")
    header = "Metric/Feature   | " + " | ".join([f"{feature}" for feature in feature_names])
    print(header)
    print("---------------------------------------------------------")
    train_loss_str = "Train Loss       | " + " | ".join([safe_format(train_losses[feature], "{:.4f}") for feature in feature_names])
    train_r2_str = "Train R2         | " + " | ".join([safe_format(train_r2s[feature], "{:.4f}") for feature in feature_names])
    val_loss_str = "Validation Loss  | " + " | ".join([safe_format(val_losses[feature], "{:.4f}") for feature in feature_names])
    val_r2_str = "Validation R2    | " + " | ".join([safe_format(val_r2s[feature], "{:.4f}") for feature in feature_names])

    print(train_loss_str)
    print(train_r2_str)
    print(val_loss_str)
    print(val_r2_str)
    print("---------------------------------------------------------")
    print("\n")



def main(best_lrs):
    feature_names = ['VEC', 'electronegativity', 'cohesive energy', 'TEC', 'TC', 'density']

    dataset = {feature: AlloysDataset2(
        data_path=r"C:\Users\PC\Desktop\Final_data\Final_data\Newest_data\Final_train_data.xlsx",
        element_path=r"C:\Users\PC\Desktop\element_data.xlsx", target_feature=feature) for feature in feature_names}
    test_datasets = {
        feature: AlloysDataset2(r"C:\Users\PC\Desktop\Final_data\Final_data\测试集2\test_data_chunk_1.xlsx",
                                element_path=r"C:\Users\PC\Desktop\element_data.xlsx",
                                target_feature=feature) for feature in feature_names}


    batch_size = 32
    train_datasets = {}
    val_datasets = {}
    writers = {feature: SummaryWriter(log_dir=f"./logs/{feature}") for feature in feature_names}

    # 划分训练和验证数据集
    for feature in feature_names:
        dataset_size = len(dataset[feature])
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset[feature], [train_size, val_size])
        train_datasets[feature] = train_dataset
        val_datasets[feature] = val_dataset

    # 数据加载器
    train_loaders = {feature: DataLoader(train_datasets[feature], batch_size=batch_size, shuffle=True, num_workers=12) for feature in
                     feature_names}
    val_loaders = {feature: DataLoader(val_datasets[feature], batch_size=batch_size, shuffle=False) for feature in
                   feature_names}
    test_loaders = {feature: DataLoader(test_datasets[feature], batch_size=batch_size, shuffle=False) for feature in feature_names}

    # 模型参数
    EMBEDDING_DIM = 8
    additional_features_dims = {feature: dataset[feature].additional_features.shape[1] for feature in feature_names}
    num_elements = len(next(iter(dataset.values())).element_to_id) + 1  # 取第一个数据集的属性作为参考
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数、优化器和早停
    models = {
        'VEC': PredictVEC(num_elements, EMBEDDING_DIM, additional_features_dims['VEC'],
                          dataset['VEC'].padding_value).to(device),
        'electronegativity': PredictEletronegativity(num_elements, EMBEDDING_DIM,
                                                     additional_features_dims['electronegativity'],
                                                     dataset['electronegativity'].padding_value).to(device),
        'cohesive energy': PredictCohesive(num_elements, EMBEDDING_DIM, additional_features_dims['cohesive energy'],
                                           dataset['cohesive energy'].padding_value).to(device),
        'TEC': PredictTEC(num_elements, EMBEDDING_DIM, additional_features_dims['TEC'],
                          dataset['TEC'].padding_value).to(device),
        'TC': PredictTC(num_elements, EMBEDDING_DIM, additional_features_dims['TC'], dataset['TC'].padding_value).to(
            device),
        'density': PredictDensity(num_elements, EMBEDDING_DIM, additional_features_dims['density'],
                                  dataset['density'].padding_value).to(device)
    }

    criterions = {feature: nn.MSELoss().to(device) for feature in feature_names}
    optimizers = {feature: optim.Adam(models[feature].parameters(), lr=best_lrs[feature], weight_decay=0.1) for feature in feature_names}

    early_stoppings = {feature: EarlyStopping(patience=5, verbose=True) for feature in feature_names}

    # 训练和验证模型
    EPOCHS = 50
    for epoch in range(EPOCHS):
        for feature in feature_names:
            model = models[feature]
            criterion = criterions[feature]
            optimizer = optimizers[feature]
            early_stopping = early_stoppings[feature]

            train_one_epoch(model, train_loaders[feature], criterion, optimizer, device, feature_names, feature)
            train_loss, train_r2 = train_one_epoch(model, train_loaders[feature], criterion, optimizer, device,
                                                   feature_names, feature)
            val_loss, val_r2 = validate_one_epoch(model, val_loaders[feature], criterion, device, feature_names,
                                                  feature)

            print(f"Epoch {epoch}/{EPOCHS} - Feature {feature}")
            print(f"Training Loss: {train_loss:.4f}, Training R2: {train_r2:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation R2: {val_r2:.4f}")
            writers[feature].add_scalar('Loss/train', train_loss, epoch)
            writers[feature].add_scalar('R2/train', train_r2, epoch)
            writers[feature].add_scalar('Loss/val', val_loss, epoch)
            writers[feature].add_scalar('R2/val', val_r2, epoch)

            early_stopping(val_loss, model)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping for feature: {feature}")
                early_stopping.early_stop = False  # reset for the next feature

    for writer in writers.values():
        writer.close()

    return models, test_loader, criterions, device, feature_names

def train_one_epoch(model, train_loader, criterion, optimizer, device, feature_names, feature):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch_element_ids, batch_element_ratios, _, batch_targets in train_loader:
        batch_element_ids = batch_element_ids.to(device)
        batch_element_ratios = batch_element_ratios.to(device)
        # batch_target = batch_targets[:, feature_names.index(feature)].unsqueeze(1).to(device)
        batch_target = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_element_ids, batch_element_ratios)
        loss = criterion(outputs, batch_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_target.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(all_labels, all_preds)
    return avg_loss, r2

def validate_one_epoch(model, val_loader, criterion, device, feature_names, feature):
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_element_ids, batch_element_ratios, _, batch_targets in val_loader:
            batch_element_ids = batch_element_ids.to(device)
            batch_element_ratios = batch_element_ratios.to(device)
            batch_target = batch_targets.to(device)

            outputs = model(batch_element_ids, batch_element_ratios)
            loss = criterion(outputs, batch_target)
            total_val_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_target.cpu().numpy())
    avg_loss = total_val_loss / len(val_loader)
    r2 = r2_score(all_labels, all_preds)
    return avg_loss, r2

def evaluate_single_feature_model(model, test_loader, criterion, device, feature_name):
    model.eval()

    all_test_preds = []
    all_test_labels = []
    total_test_loss = 0.0

    with torch.no_grad():
        for batch_element_ids, batch_element_ratios, _, batch_feature_targets in test_loader:
            batch_element_ids = batch_element_ids.to(device)
            batch_element_ratios = batch_element_ratios.to(device)
            batch_target = batch_feature_targets.to(device)

            outputs = model(batch_element_ids, batch_element_ratios)

            feature_loss = criterion(outputs, batch_target)
            total_test_loss += feature_loss.item()

            all_test_preds.extend(outputs.detach().cpu().numpy())
            all_test_labels.extend(batch_target.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_r2_score = r2_score(all_test_labels, all_test_preds)
    test_mse = avg_test_loss
    test_rmse = sqrt(avg_test_loss)

    # 输出每个特征的结果
    print(f"Feature: {feature_name}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test R2 Score: {test_r2_score:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


def evaluate_all_features(models, test_loaders, criterions, device, feature_names):
    for feature in feature_names:
        evaluate_single_feature_model(models[feature], test_loaders[feature], criterions[feature], device, feature)



if __name__ == "__main__":
    best_learning_rates = {
        'VEC': 0.01894,
        'electronegativity': 0.0007462,
        'cohesive energy': 0.001157,
        'TEC': 0.0007462,
        'TC': 0.00251,
        'density': 0.000746
    }
    feature_models, test_loader, feature_criterions, device, feature_names = main(best_learning_rates)
    evaluate_all_features(feature_models, test_loader, feature_criterions, device, feature_names)

