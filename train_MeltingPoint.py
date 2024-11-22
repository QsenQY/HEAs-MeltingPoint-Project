from Dataset.AlloysDataset import AlloysDataset
from Models.PredictMeltingPoint_model import ExtendedAlloyModel
from Utils import utils
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from math import sqrt
import pickle
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from sklearn.metrics import mean_absolute_error


# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
feature_cols = ['VEC', 'electronegativity', 'cohesive energy', 'density', 'radius', 'heat of fusion']



def split_and_scale_data(data, feature_cols, test_size=0.2, random_state=None):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    train_features, scaler = utils.additional_features_scaling(train_data, feature_cols)
    val_features, _ = utils.additional_features_scaling(val_data, feature_cols, scaler)

    return train_data, val_data, train_features, val_features, scaler


def prepare_data(data, feature_cols, element_path):
    train_data, val_data, train_features, val_features, scaler = split_and_scale_data(data, feature_cols)

    train_dataset = AlloysDataset(element_path, train_features, data=train_data)
    val_dataset = AlloysDataset(element_path, val_features, data=val_data)

    return train_dataset, val_dataset, scaler

def train_model(train_loader, val_loader, model, optimizer, criterion, scheduler, device, writer, EPOCHS=20, patience=3, min_delta=0.001, fold=0, save_path=r"C:\Users\PC\Desktop\Final_data\model_weight\MeltingPointModel.pt"):
    best_val_loss = float('inf')
    best_model = None
    no_improvement_count = 0

    for epoch in range(EPOCHS):
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

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train R2: {train_r2_score:.4f}, Validation R2: {val_r2_score:.4f}",
            flush=True)
        val_mae = mean_absolute_error(all_val_labels, all_val_preds)
        print(f"Validation MAE: {val_mae}")

        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            no_improvement_count = 0  # Reset counter
            best_model = deepcopy(model)  # Save the best model
            torch.save(best_model.state_dict(),
                       f'C:\\Users\\PC\\Desktop\\Final_data\\model_weight\\4hiddenlayer\\MeltingPointModel_fold_{fold}.pt')
            print("Save best model")
        else:
            no_improvement_count += 1

        if no_improvement_count > patience:
            break

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('R2/train', train_r2_score, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('R2/val', val_r2_score, epoch)

    writer.close()


    return best_model, best_val_loss

def k_fold_cross_validation(dataset, model_init_func, criterion, scheduler, device, writer,  k=5, EPOCHS=20, patience=3, min_delta=0.001, lr=0.040383):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold = 0
    best_global_val_loss = float('inf')
    best_global_model = None

    for train_idx, val_idx in kf.split(dataset.data):  # 注意这里我们分割的是dataset.data
        print(f"Starting fold {fold + 1}/{k}")
        train_data_raw = dataset.data.iloc[train_idx]
        val_data_raw = dataset.data.iloc[val_idx]

        train_features, fold_scaler = utils.additional_features_scaling(train_data_raw, feature_cols)


        val_features, _ = utils.additional_features_scaling(val_data_raw, feature_cols, fold_scaler)

        # 创建每个折叠的训练和验证数据集
        train_dataset = AlloysDataset(
            data=train_data_raw,
            element_path=r"C:\Users\PC\Desktop\element_data.xlsx",  # 正确的element_path
            additional_features=train_features # 使用已标准化的特征
        )
        val_dataset = AlloysDataset(
            data=val_data_raw,
            element_path=r"C:\Users\PC\Desktop\element_data.xlsx",  # 正确的element_path
            additional_features=val_features  # 使用已标准化的特征
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Reset model using the provided initialization function
        model = model_init_func().to(device)

        # Directly create a new optimizer instance for each fold
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train model on this fold
        model, best_val_loss = train_model(train_loader, val_loader, model, optimizer, criterion, scheduler, device,
                                           writer, EPOCHS, patience, min_delta, fold)

        # Check if this fold's model is the best among all folds
        if best_val_loss < best_global_val_loss:
            best_global_val_loss = best_val_loss
            best_global_model = model

        fold += 1



    entire_features, _ = utils.additional_features_scaling(dataset.data, feature_cols)
    entire_dataset = AlloysDataset(
        data=dataset.data,
        element_path=r"C:\Users\PC\Desktop\element_data.xlsx",
        additional_features = entire_features
    )

    entire_loader = DataLoader(entire_dataset, batch_size=64, shuffle=False)
    all_ensemble_predictions = []

    # Use each fold's best model to predict on the entire dataset
    for fold in range(k):
        fold_model_path = f'C:\\Users\\PC\\Desktop\\Final_data\\model_weight\\4hiddenlayer\\MeltingPointModel_fold_{fold}.pt'
        best_global_model.load_state_dict(torch.load(fold_model_path))
        fold_predictions = []
        with torch.no_grad():
            for batch_element_ids, batch_element_ratios, batch_additional_features, _, _ in entire_loader:
                batch_element_ids = batch_element_ids.to(device)
                batch_element_ratios = batch_element_ratios.to(device)
                batch_additional_features = batch_additional_features.to(device)
                outputs = best_global_model(batch_element_ids, batch_element_ratios, batch_additional_features)
                fold_predictions.extend(outputs.cpu().numpy())

        all_ensemble_predictions.append(fold_predictions)

    # Calculate mean and standard deviation for ensemble predictions
    mean_predictions = np.mean(all_ensemble_predictions, axis=0)
    std_predictions = np.std(all_ensemble_predictions, axis=0)

    # Save the predictions to an Excel file
    df = pd.DataFrame({
        'Mean Prediction': mean_predictions.flatten(),
        'Std. Deviation': std_predictions.flatten()
    })
    df.to_excel('ensemble_predictions_after_kfold.xlsx', index=False)

    return best_global_model


def model_init_func(dataset):
    EMBEDDING_DIM = 8
    num_elements = len(dataset.element_to_id) + 1
    additional_features_dim = dataset.additional_features.shape[1]
    return ExtendedAlloyModel(num_elements, EMBEDDING_DIM, additional_features_dim, dataset.padding_value)


def main():
    # Load dataset
    raw_data = pd.read_excel(r"C:\Users\PC\Desktop\train_data(add_feature)test.xlsx")

    # Prepare the data using the split_and_scale_data function
    # train_data, val_data, train_features, val_features, scaler = split_and_scale_data(
    #     raw_data, feature_cols
    # )

    # Create datasets for training and validation
    train_dataset = AlloysDataset(r"C:\Users\PC\Desktop\element_data.xlsx", additional_features=raw_data[feature_cols].values, data=raw_data)

    test_data = pd.read_excel(r"C:\Users\PC\Desktop\Final_data\Final_data\测试集3\test_subset_3(add_feature).xlsx")
    # Create the test dataset
    global_scaler = StandardScaler()
    global_scaler.fit(raw_data[feature_cols])

    test_additional_features, _ = utils.additional_features_scaling(test_data, feature_cols, global_scaler)
    test_dataset = AlloysDataset(
        element_path=r"C:\Users\PC\Desktop\element_data.xlsx",
        additional_features=test_additional_features,
        data=test_data,
        predict_mode=False
    )

    # Scale the test data
    # test_dataset.data[feature_cols], _ = utils.additional_features_scaling(test_dataset.data, feature_cols, scaler)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    EMBEDDING_DIM = 8
    num_elements = len(train_dataset.element_to_id) + 1
    additional_features_dim = train_dataset.additional_features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExtendedAlloyModel(num_elements, EMBEDDING_DIM, additional_features_dim, train_dataset.padding_value)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0681997, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    model = model.to(device)


    element_to_id = train_dataset.element_to_id
    with open("element_to_id.pkl", "wb") as f:
        pickle.dump(element_to_id, f)

    writer = SummaryWriter()

    trained_model = k_fold_cross_validation(train_dataset, lambda: model_init_func(train_dataset), criterion, scheduler, device, writer,  k=5, lr=0.040383)

    train_features = global_scaler.transform(raw_data[feature_cols])
    train_dataset.additional_features = train_features


    test_dataset.data[feature_cols], _ = utils.additional_features_scaling(test_dataset.data, feature_cols, global_scaler)

    return trained_model, test_loader, criterion, device, global_scaler


def predict_with_ensemble(input_data, model_instance, n_models=5, save_path='ensemble_predictions.xlsx'):
    all_predictions = []

    # Load each model and make predictions
    for ensemble_idx in range(n_models):
        model_path = f'C:\\Users\\PC\\Desktop\\Final_data\\model_weight\\4hiddenlayer\\MeltingPointModel_fold_{ensemble_idx}.pt'
        model_instance.load_state_dict(torch.load(model_path))
        model_instance.eval()

        with torch.no_grad():
            element_ids, element_ratios, additional_features = input_data
            predictions = model_instance(element_ids, element_ratios, additional_features)
            all_predictions.append(predictions)

    # Calculate mean and confidence interval
    mean_prediction = torch.mean(torch.stack(all_predictions), dim=0)
    std_prediction = torch.std(torch.stack(all_predictions), dim=0)
    df = pd.DataFrame({
        'Mean Prediction': mean_prediction.cpu().numpy().flatten(),
        'Std. Deviation': std_prediction.cpu().numpy().flatten()
    })
    df.to_excel(save_path, index=False)
    return mean_prediction, std_prediction

def evaluate_MeltingPoint_Model(model, test_loader, criterion, device):

    save_path = r"C:\Users\PC\Desktop\Final_data\model_weight\4hiddenlayer\MeltingPointModel_fold_1.pt"
    model.load_state_dict(torch.load(save_path))
    model.eval()

    all_test_preds = []
    all_test_labels = []
    total_test_loss = 0.0
    with torch.no_grad():
        for batch_element_ids, batch_element_ratios, batch_additional_features, batch_melting_point_targets, _ in test_loader:
            batch_element_ids = batch_element_ids.to(device)
            batch_element_ratios = batch_element_ratios.to(device)
            batch_additional_features = batch_additional_features.to(device)
            batch_melting_point_targets = batch_melting_point_targets.to(device)

            outputs, _ = predict_with_ensemble((batch_element_ids, batch_element_ratios, batch_additional_features),
                                               model)
            loss = criterion(outputs, batch_melting_point_targets)
            total_test_loss += loss.item()
            all_test_preds.extend(outputs.detach().cpu().numpy())
            all_test_labels.extend(batch_melting_point_targets.cpu().numpy())
    avg_test_loss = total_test_loss / len(test_loader)
    test_r2_score = r2_score(all_test_labels, all_test_preds)
    test_mse = avg_test_loss
    test_rmse = sqrt(avg_test_loss)
    test_mae = mean_absolute_error(all_test_labels, all_test_preds)
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Test R2 Score: {test_r2_score}")
    print(f"Test MSE: {test_mse}")
    print(f"Test RMSE: {test_rmse}")



if __name__ == "__main__":
    best_model, test_loader, criterion, device, global_scaler = main()
    evaluate_MeltingPoint_Model(best_model, test_loader, criterion, device)

