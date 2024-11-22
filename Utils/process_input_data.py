import torch
from utils import element_to_id_conversion

def preprocess_user_input(user_input, element_to_id, padding_value):
    # 提取用户输入的元素和比例
    elements = user_input['Alloys']
    ratios = user_input['Ratios']

    # 转换元素为对应的ID
    element_ids = [element_to_id[element] for element in elements if element in element_to_id]

    # 填充到固定长度
    while len(element_ids) < 5:
        element_ids.append(padding_value)
        ratios.append(0.0)

    # 将列表转换为PyTorch tensors
    element_ids_tensor = torch.tensor(element_ids[:5], dtype=torch.long)
    ratios_tensor = torch.tensor(ratios[:5], dtype=torch.float32)

    return element_ids_tensor, ratios_tensor