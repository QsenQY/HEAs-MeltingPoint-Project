import pandas as pd
import numpy as np
import itertools


element_data = pd.read_excel('/path')


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

