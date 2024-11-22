import pandas as pd
import numpy as np
import itertools

# Load the data
element_data = pd.read_excel(r"C:\Users\PC\Desktop\element_data.xlsx", sheet_name="Sheet1")
# binary_hmix_data = pd.read_excel(r"C:\Users\PC\Desktop\element_data.xlsx", sheet_name="Sheet2")

# Define the transition metals excluding Pt and Ru
# transition_metals = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
#                      "Y", "Zr", "Nb", "Mo", "Tc", "Rh", "Pd", "Ag", "Cd",
#                      "Hf", "Ta", "W", "Re", "Os", "Ir","Au"]
#
# def calculate_properties(metals):
#     selected_data = element_data[element_data['Element'].isin(metals)]
#     alloy_properties = selected_data.select_dtypes(include=[np.number]).mean()
#
#     # mixing_entropy = -8.314 * 5 * 0.2 * np.log(0.2)
#     # mixing_enthalpy = 0
#     # for combo in itertools.combinations(metals, 2):
#     #     ci, cj = 0.2, 0.2
#     #     try:
#     #         Hmix = binary_hmix_data.set_index("Unnamed: 0").loc[combo[0], combo[1]]
#     #     except KeyError:
#     #         Hmix = binary_hmix_data.set_index("Unnamed: 0").loc[combo[1], combo[0]]
#     #     mixing_enthalpy += 4 * ci * cj * Hmix
#
#     return {
#         "Alloys": "-".join(metals),
#         # "Δmixed entropy": mixing_entropy,
#         # "Δmixed enthalpy": mixing_enthalpy,
#         "VEC": alloy_properties["VEC"],
#         "electronegativity": alloy_properties["electronegativity"],
#         "cohesive energy": alloy_properties["cohesive energy"],
#         "TEC": alloy_properties["TEC"],
#         "TC": alloy_properties["TC"],
#         "density": alloy_properties["density"]
#     }
#
# # Generate all possible combinations of 3 metals from the transition metals
# all_combinations = list(itertools.combinations(transition_metals, 3))
#
# # Calculate properties for all combinations, including Pt and Ru
# results = []
# for combo in all_combinations:
#     combo_with_pt_ru = ["Pt", "Ru"] + list(combo)
#     results.append(calculate_properties(combo_with_pt_ru))
#
# # Convert the results to a DataFrame and save as Excel
# all_alloys_df = pd.DataFrame(results)
# all_alloys_df.to_excel("all_alloy_properties_with_Pt_Ru_updated.xlsx", index=False)

import pandas as pd
import numpy as np
import itertools
import random

# Load the element data from the provided Excel file
element_data = pd.read_excel(r"C:\Users\PC\Desktop\element_data.xlsx", sheet_name="Sheet1")

# Define the list of elements including Pt and Ru
elements = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Au", "Pt", "Ru"]


def calculate_properties(metals):
    selected_data = element_data[element_data['Element'].isin(metals)]
    alloy_properties = selected_data.select_dtypes(include=[np.number]).mean()

    return {
        "Alloys": "-".join(metals),
        "VEC": alloy_properties["VEC"],
        "electronegativity": alloy_properties["electronegativity"],
        "cohesive energy": alloy_properties["cohesive energy"],
        "TEC": alloy_properties["TEC"],
        "TC": alloy_properties["TC"],
        "density": alloy_properties["density"]
    }


# Generate random combinations of 5 metals from the element list
num_combinations = 10000  # You can change this to the number of combinations you want
results = []
for _ in range(num_combinations):
    random_metals = random.sample(elements, 5)
    results.append(calculate_properties(random_metals))

# Convert the results to a DataFrame and save as Excel
random_alloys_df = pd.DataFrame(results)
random_alloys_df.to_excel("random_alloy_properties.xlsx", index=False)
