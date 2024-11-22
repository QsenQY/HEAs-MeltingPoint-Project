import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

additional_features_cols = [ 'VEC', 'electronegativity', 'cohesive energy',  'density', 'radius', 'heat of fusion']
def element_to_id_conversion(data, element_to_id=None):
    # print("Type of data in conversion function:", type(data))
    # print('Alloys' in data.columns)
    # print("First few rows of data in conversion function:", data.head())

    # if isinstance(data, pd.Series):
    #     print("Data is a Series.")
    # elif isinstance(data, pd.DataFrame):
    #     print("Data is a DataFrame.")
    # else:
    #     print("Data is an unknown type.")
    if element_to_id is None:
        all_elements = set()
        try:
            for alloy in data['Alloys']:
                if not isinstance(alloy, str):  # 检查是否为字符串
                    alloy = str(alloy)          # 如果不是字符串，转换为字符串
                elements = alloy.split('-')
                for element in elements:
                    all_elements.add(element)
        except KeyError:
            print("Key 'Alloys' not found in data.")
            return None, None, None, None
        # print(f"All elements found: {all_elements}")

        element_to_id = {element: idx for idx, element in enumerate(sorted(all_elements))}

        # 新添加的检查和打印语句
        # print(f"Generated element_to_id: {element_to_id}")

    padding_value = len(element_to_id)
    # Create element IDs and ratios arrays
    element_ids = []
    element_ratios = []

    try:
        for _, row in data.iterrows():
            ratios = [row['element1'], row['element2'], row['element3'], row['element4'], row['element5']]
            elements = row['Alloys'].split('-')
            current_ids = [element_to_id[element] for element in elements if element in element_to_id]
            current_ratios = [ratio for ratio in ratios if ratio > 0]
            while len(current_ids) < 5:
                current_ids.append(padding_value)
                current_ratios.append(0.0)
            element_ids.append(current_ids[:5])
            element_ratios.append(current_ratios[:5])
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        print(f"Row data: {row}")

    return element_ids, element_ratios, element_to_id, padding_value


def additional_features_scaling(data, feature_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_additional_features = scaler.fit_transform(data[feature_cols])
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaled_additional_features = scaler.transform(data[feature_cols])

    return scaled_additional_features, scaler

