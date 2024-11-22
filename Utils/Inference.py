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
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pickle

import torch
import pandas as pd
from Models.PredictMeltingPoint_model import ExtendedAlloyModel  
from Utils import utils  
from Dataset.AlloysDataset import AlloysDataset  
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
    data_path='/path',
    element_data_path='/path',
    weights_path='path',
    save_path='/path',
    scaler_path='/path'
)
