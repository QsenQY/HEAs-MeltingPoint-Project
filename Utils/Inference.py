#
import torch
from utils import element_to_id_conversion
from Utils.process_input_data import preprocess_user_input
from sklearn.metrics import r2_score
from math import sqrt
from Models.PredictMeltingPoint_model import ExtendedAlloyModel
from Dataset.AlloysDataset import AlloysDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from Utils.utils import element_to_id_conversion

from Dataset.AlloysDataset import AlloysDataset
import pickle
import torch
import random

# def set_seeds(seed_value=42):
#     """Set seeds for reproducibility."""
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     random.seed(seed_value)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
# set_seeds()
#
# feature_cols = ['VEC', 'electronegativity', 'cohesive energy', 'TEC', 'TC', 'density', 'radius', 'heat of fusion']
# def process_new_data(data_path, element_data_path, element_to_id=None, additional_features = None):
#     data = pd.read_excel(data_path)
#     print(data.columns)
#     if additional_features is None:
#         additional_features = data[feature_cols]
#
#     dataset = AlloysDataset(data, element_data_path, additional_features, predict_mode=True, element_to_id=element_to_id)
#     loader = DataLoader(dataset, batch_size=64, shuffle=False)
#     print(dataset.element_to_id)
#     return loader
#
# with open(r"C:\Users\PC\Desktop\HEA_MeltingPoint\element_to_id.pkl", "rb") as f:
#     element_to_id = pickle.load(f)
# new_data_path = r"C:\Users\PC\Desktop\all_alloy_properties_with_Pt_Ru_updated 2.xlsx"
# element_data_path =r"C:\Users\PC\Desktop\element_data.xlsx"
# new_data_loader = process_new_data(new_data_path, element_data_path, element_to_id=element_to_id)
#
# EMBEDDING_DIM = 8
# num_elements = 36
# additional_features_dim = 53
# model = ExtendedAlloyModel(num_elements, EMBEDDING_DIM, additional_features_dim, padding_value=35)
# model_path = r"C:\Users\PC\Desktop\Final_data\model_weight\MeltingPointModel.pt"
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# predictions_list = []
# with torch.no_grad():
#     for batch_element_ids, batch_element_ratios, batch_additional_features,  _ in new_data_loader:
#
#         outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
#         predictions = outputs.cpu().numpy()
#         predictions_list.append(predictions)
#
# all_predictions = np.concatenate(predictions_list, axis=0)
# predictions_df = pd.DataFrame(all_predictions, columns=["Predicted Melting Point"])
# save_path = r"C:\Users\PC\Desktop\HEA_MeltingPoint\Utils\predicted_melting_points1.xlsx"
# predictions_df.to_excel(save_path, index=False)


import torch
import pandas as pd
from torch.utils.data import DataLoader
import pickle

import torch
import pandas as pd
from Models.PredictMeltingPoint_model import ExtendedAlloyModel  # 根据您的模型路径进行调整
from Utils import utils  # 根据您的 utils 路径进行调整
from Dataset.AlloysDataset import AlloysDataset  # 根据您的数据集路径进行调整
from torch.utils.data import DataLoader
from joblib import load

def predict_melting_point(data_path, element_data_path, weights_path, save_path, scaler_path):
    # Step 1: Data Preprocessing
    scaler = load(scaler_path)
    data = pd.read_excel(data_path)

    feature_cols = ['VEC', 'electronegativity', 'cohesive energy',  'density', 'radius', 'heat of fusion']
    additional_features = data[feature_cols]
    scaled_additional_features = scaler.transform(additional_features)
    dataset = AlloysDataset(element_data_path, scaled_additional_features, data, predict_mode=True)

    # Step 3: Initialize Model
    # Now you can use values from dataset object to initialize the model
    EMBEDDING_DIM = 8  # Update as per your training configuration
    num_elements = 36  # Update as per your training configuration
    additional_features_dim = dataset.additional_features.shape[1]
    model = ExtendedAlloyModel(num_elements, EMBEDDING_DIM, additional_features_dim, padding_value=35)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Step 4: Create DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Step 5: Model Inference
    predictions_list = []
    with torch.no_grad():
        for batch_element_ids, batch_element_ratios, batch_additional_features, batch_features in loader:
            outputs = model(batch_element_ids, batch_element_ratios, batch_additional_features)
            predictions = outputs.cpu().numpy()
            predictions_list.append(predictions)

    # Step 6: Output Results
    all_predictions = np.concatenate(predictions_list, axis=0)
    predictions_df = pd.DataFrame(all_predictions, columns=["Predicted Melting Point"])
    predictions_df.to_excel(save_path, index=False)


predict_melting_point(
    data_path=r"C:\Users\PC\Desktop\绘图代码  数据处理代码\ORR HEA catalyst.xlsx",
    element_data_path=r"C:\Users\PC\Desktop\element_data.xlsx",
    weights_path=r"C:\Users\PC\Desktop\Final_data\model_weight\4hiddenlayer\MeltingPointModel_fold_1.pt",
    save_path=r"C:\Users\PC\Desktop\HEA_MeltingPoint\Utils\predictions4.xlsx",
    scaler_path=r"C:\Users\PC\Desktop\HEA_MeltingPoint\scaler.pkl"
)
