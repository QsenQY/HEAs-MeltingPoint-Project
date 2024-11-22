import pandas as pd
import torch
from torch.utils.data import Dataset
from Utils import utils


# Melting_point_dataset
class AlloysDataset(Dataset):
    def __init__(self, element_path, additional_features, data=None, predict_mode=False, element_to_id=None):
        # print("Entering __init__ method.")
        # print("Columns at the start of AlloysDataset:", data.columns)
        self.data = data
        # print("Columns before element_to_id_conversion:", self.data.columns)
        self.element_ids, self.element_ratios, self.element_to_id, self.padding_value = utils.element_to_id_conversion(
            self.data, element_to_id=element_to_id)
        # print("Columns after element_to_id_conversion:", self.data.columns)
        self.padding_value = len(self.element_to_id)
        self.element_dict = self.create_element_dict(element_path)
        self.data = self.extend_feature(self.data)
        feature_columns = ['VEC', 'electronegativity', 'cohesive energy', 'density', 'radius', 'heat of fusion']
        # 从 element_dict 中获取第一个元素的属性名
        sample_element_props = list(next(iter(self.element_dict.values())).keys())
        extended_feature_columns = feature_columns + [f'element{i}_{prop}' for i in range(1, 6) for prop in sample_element_props]

        self.additional_features = additional_features

        # 当 predict_mode 为 False 时，读取 "melting point" 列
        if not predict_mode:
            self.melting_point = torch.tensor(self.data["melting point"].values, dtype=torch.float32).unsqueeze(1)
        else:
            self.melting_point = None

        self.features = torch.tensor(self.data[feature_columns].values, dtype=torch.float32)
        self.predict_mode = predict_mode


    def __len__(self):
        if self.predict_mode:
            return len(self.features)
        return len(self.melting_point)

    def __getitem__(self, idx):
        # print("Columns in __getitem__:", self.data.columns)
        element_ids_tensor = torch.tensor(self.element_ids[idx], dtype=torch.long)
        element_ratios_tensor = torch.tensor(self.element_ratios[idx], dtype=torch.float32)
        if self.additional_features is None:
            raise ValueError("Additional features is None. Please provide valid additional features.")
        additional_features_tensor = torch.tensor(self.additional_features[idx], dtype=torch.float32)
        # additional_features_tensor = torch.tensor(self.additional_features.iloc[idx], dtype=torch.float32)

        # 当 predict_mode 为 True 时，不返回 melting_point
        if self.predict_mode:
            return element_ids_tensor, element_ratios_tensor, additional_features_tensor, self.features[idx]
        else:
            return element_ids_tensor, element_ratios_tensor, additional_features_tensor, self.melting_point[idx], self.features[idx]


    def create_element_dict(self, element_data_path):
        element_df = pd.read_excel(element_data_path)
        # print("Length of element_df:", len(element_df))
        # print("Last few rows of element_df:\n", element_df.tail())
        element_dict = {}
        for _, row in element_df.iterrows():
                if 'Element' not in row:
                    raise ValueError("Row without 'Element' found:\n" + str(row))
        return {row['Element']: row.drop('Element').to_dict() for _, row in element_df.iterrows()}

    def extend_feature(self, data):
        # print("First few entries of element_dict:")
        # print({k: self.element_dict[k] for k in list(self.element_dict)[:3]})
        data = data.copy(deep=True)
        for inx, row in data.iterrows():
            alloys = row['Alloys'].split('-')
            for i, element_name in enumerate(alloys, 1):
                for prop, value in self.element_dict.get(element_name, {}).items():
                    column_name = f'element{i}_{prop}'
                    if column_name not in data.columns:
                        data[column_name] = 0
                    data.at[inx, column_name] = value
        return data


# Feature_dataset
# class AlloysDataset2(Dataset):
#     def __init__(self, data_path, element_path, target_feature):
#         self.data = pd.read_excel(data_path)
#         self.element_ids, self.element_ratios, self.element_to_id, self.padding_value = utils.element_to_id_conversion(self.data)
#         self.padding_value = len(self.element_to_id)
#         feature_columns = ['VEC', 'electronegativity', 'cohesive energy', 'TEC', 'TC', 'density']
#         assert target_feature in feature_columns, f"{target_feature} not in available features"
#         self.additional_features = utils.additional_features_scaling(self.data, feature_columns)
#         self.element_dict = self.create_element_dict(element_path)
#         self.data = self.extend_feature(self.data)
#         self.target = self.data[target_feature].values
#
#     def __len__(self):
#         return len(self.target)
#
#     def __getitem__(self, idx):
#         element_ids_tensor = torch.tensor(self.element_ids[idx], dtype=torch.long)
#         element_ratios_tensor = torch.tensor(self.element_ratios[idx], dtype=torch.float32)
#         additional_features_tensor = torch.tensor(self.additional_features[idx], dtype=torch.float32)
#         target_value = torch.tensor(self.target[idx], dtype=torch.float32)
#         target_tensor = target_value.unsqueeze(0) if len(target_value.shape) == 0 else target_value
#
#         return element_ids_tensor, element_ratios_tensor, additional_features_tensor, target_tensor
#
#     def create_element_dict(self, element_data_path):
#         element_df = pd.read_excel(element_data_path)
#         return {row['Element']: row.drop('Element').to_dict() for _, row in element_df.iterrows()}
#
#     def extend_feature(self, data):
#         for inx, row in self.data.iterrows():
#             alloys = row['Alloys'].split('-')
#             for i, element_name in enumerate(alloys, 1):
#                 for prop, value in self.element_dict.get(element_name, {}).items():
#                     column_name = f'element{i}_{prop}'
#                     if column_name not in data.columns:
#                         data[column_name] = 0
#                     data.at[inx, column_name] = value
#         print(data.head())
#         return data
