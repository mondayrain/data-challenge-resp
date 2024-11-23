import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


from pathlib import Path

# Set directory path
INPUT_PATH = './data'
TEXT_FILE_PATH = INPUT_PATH + "/events/"

# Set file headers as constants
DEMOGRAPHIC_INFO_HEADERS = ['PatientNumber', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)']
FILENAME_INFO_HEADERS = ["PatientNumber", "RecIndex", "ChestLocation", "AquisitionMode", "RecEquipment"]


# =============================================
# === READING DEMOGRAPHIC INFO
# =============================================
# region
print("\n==== DEMOGRAPHIC INFO ====")

df_demographic_info = pd.read_csv(INPUT_PATH + '/demographic_info.txt',
                                  names = DEMOGRAPHIC_INFO_HEADERS,
                                  delimiter = "\t")

print("\n Figuring out if we're missing demographic data:")
rows_missing_all_except_patient = df_demographic_info.drop(columns=['PatientNumber']).isnull().all(axis=1)
print(f"\nRows missing all data except PatientNumber: {rows_missing_all_except_patient.sum}")
print(rows_missing_all_except_patient)

# How many and which rows with age < 15 are missing Child Weight (kg) or Child Height (cm)?
rows_missing_child_data = df_demographic_info[
    (df_demographic_info['Age'] < 15) & 
    (df_demographic_info[['Child Weight (kg)', 'Child Height (cm)']].isnull().any(axis=1))
]
print(f"\nRows with age < 15 missing Child Weight or Child Height: {len(rows_missing_child_data)}\n")
print(rows_missing_child_data)

# 3. How many and which rows with Age > 15 that are missing Adult BMI (kg/m2)?
adult_rows_missing_bmi = df_demographic_info[
    (df_demographic_info['Age'] >= 15) & (df_demographic_info[['Adult BMI (kg/m2)']].isnull().any(axis=1))
]
print(f"\nRows with Age >= 15 Adult and missing BMI (kg/m2): {len(adult_rows_missing_bmi)}\n")
print(adult_rows_missing_bmi)

# endregion

# =============================================
# === IMPUTE MISSING DEMOGRAPHIC INFO
# =============================================
# region
print("\n==== TODO: IMPUTE MISSING DEMOGRAPHIC INFO ====")

# endregion
# =============================================
# === READING DIAGNOSIS INFO
# =============================================
# region
print("\n==== DIAGNOSIS INFO ====")
df_diagnoses_info = pd.read_csv(INPUT_PATH + '/patient_diagnosis.csv',
                               names = ["PatientNumber", "Diagnosis"],
                               delimiter = "\t")

# print(df_diagnoses_info)
# endregion

# =============================================
# === JOINING DEMOGRAPHIC & DIAGNOSIS INFO
# =============================================
# region
print("\n==== JOINING DEMOGRAPHIC & DIAGNOSIS INFO ====")

joined = pd.merge(df_demographic_info, df_diagnoses_info, on='PatientNumber', how='outer')
# print(joined.head())
# endregion

# =============================================
# === SPLIT ADULTS & CHILDREN INTO TWO TABLES
# =============================================
# region
print("\n==== SPLIT ADULTS & CHILDREN INTO TWO TABLES ====")
df_adults = joined[pd.notna(joined["Adult BMI (kg/m2)"])]
df_adults = df_adults.drop(["Child Weight (kg)", "Child Height (cm)"], axis=1)

# One for children
df_children = joined[pd.isna(joined["Adult BMI (kg/m2)"])]
df_children = df_children.drop(["Adult BMI (kg/m2)"], axis=1)

total_child_records = df_children.PatientNumber.count()
print("Number of Children: {}".format(total_child_records))
# print(df_children.head())

total_adult_records = df_adults.PatientNumber.count()
print("Number of Adults: {}".format(total_adult_records))
# print(df_adults.head())

print("Total {} records".format(total_adult_records + total_child_records))
# endregion

# =============================================
# === READ IN FILENAME & FILE INFO
# =============================================
# region
FILENAME_INFO_HEADERS = ["PatientNumber", "RecIndex", "ChestLocation", "AquisitionMode", "RecEquipment"]

data = []
for dirname, _, filenames in os.walk(TEXT_FILE_PATH):
    for filename in filenames[:100]:
        if filename.endswith('.txt'):  # Process only .txt files
            # Extract metadata from the filename
            metadata_values = Path(filename).stem.split("_")
            metadata_dict = dict(zip(FILENAME_INFO_HEADERS, metadata_values))

            # Read the file contents
            file_path = os.path.join(dirname, filename)
            file_data = pd.read_csv(
                file_path,
                sep="\t",  # Tab-separated values
                header=None,  # No header in the file
                names=["CycleStart", "CycleEnd", "Crackles", "Wheezes"]  # Column names
            )

            # Add metadata to each row of file_data
            for _, row in file_data.iterrows():
                full_row = {
                    **metadata_dict,  # Add metadata from filename
                    **row.to_dict()   # Add file data
                }
                data.append(full_row)

# Combine all rows into a single DataFrame
df_respiratory_data = pd.DataFrame(data)

# Ensure PatientNumber is of type string in all DataFrames
df_respiratory_data["PatientNumber"] = df_respiratory_data["PatientNumber"].astype(str)
df_adults["PatientNumber"] = df_adults["PatientNumber"].astype(str)
df_children["PatientNumber"] = df_children["PatientNumber"].astype(str)

# Ensure Crackles and Wheezes are booleans
df_respiratory_data["Crackles"] = df_respiratory_data["Crackles"].astype(bool)
df_respiratory_data["Wheezes"] = df_respiratory_data["Wheezes"].astype(bool)

# Set 'PatientNumber' as the index
df_adults = df_adults.set_index("PatientNumber")
df_children = df_children.set_index("PatientNumber")
df_respiratory_data = df_respiratory_data.set_index("PatientNumber")

# # Merge tables with respiratory data
df_adults_with_resp_data = pd.merge(df_adults, df_respiratory_data, on='PatientNumber', how='inner').sort_index()
df_children_with_resp_data = pd.merge(df_children, df_respiratory_data, on='PatientNumber', how='inner').sort_index()

# print("\nDataFrames are indexed and sorted by PatientNumber:\n")
# print(df_adults_with_resp_data.head())
# print(df_children_with_resp_data.head())

# # Now the DataFrames are indexed and sorted by 'PatientNumber'. This lets us use .loc["PatientNumber"] to filter rows.
# print("\nWe can confirm that we have ALL the data for each patient:\n")
# print(df_adults_with_resp_data.loc["104"])
# print(df_children_with_resp_data.head())
# endregion

# =============================================
# === TRANSFORM CYCLE DATA
# =============================================
# region
print("\n=== TRANSFORM CYCLE DATA")
print("We don't care about exact CycleStart and CycleEnd times ,so we'll transform them into" \
      "CycleLength and CycleNumber (which breath within the recording)\n")

# Add a CycleNumber column based on the order of CycleStart
df_adults_with_resp_data['CycleNumber'] = df_adults_with_resp_data.groupby(['PatientNumber', 'RecIndex', 'ChestLocation'])['CycleStart'].rank(ascending=True, method='dense').astype(int)
df_children_with_resp_data['CycleNumber'] = df_children_with_resp_data.groupby(['PatientNumber', 'RecIndex', 'ChestLocation'])['CycleStart'].rank(ascending=True, method='dense').astype(int)

# Add CycleLength in seconds
df_adults_with_resp_data['CycleLength (s)'] = df_adults_with_resp_data['CycleEnd'] - df_adults_with_resp_data['CycleStart']
df_children_with_resp_data['CycleLength (s)'] = df_children_with_resp_data['CycleEnd'] - df_children_with_resp_data['CycleStart']

# Drop columns we don't need
df_adults_with_resp_data = df_adults_with_resp_data.drop(columns=['CycleStart', 'CycleEnd'])
df_children_with_resp_data = df_children_with_resp_data.drop(columns=['CycleStart', 'CycleEnd'])

# print(df_adults_with_resp_data.loc["104"])

# Use One-Hot Encoding to encode all the non-numeric values into numerical columns.
# This needs to be done for PCA and clustering because these techniques require
# treating every category independently
# This converts each category into a binary column (1 for presence, 0 for absence).
# It's fine as long as the categorical data has no inherent order.
# See: https://www.educative.io/blog/one-hot-encoding
print("\nWe also need to encode non-numerical data into numerical labels so we can put them into "\
      "correlation matrices. We can use One-Hot encoding for PCA/Dimension Reduction and Clustering later.\n")

df_adults_with_resp_data = pd.get_dummies(
    df_adults_with_resp_data,
    columns=['Sex', 'ChestLocation', 'Diagnosis'],
    drop_first=True  # Drop the first category to avoid multicollinearity
)

df_children_with_resp_data = pd.get_dummies(
    df_children_with_resp_data,
    columns=['Sex', 'ChestLocation', 'Diagnosis'],
    drop_first=True  # Drop the first category to avoid multicollinearity
)

# print(df_adults_with_resp_data.head())

# endregion

# =============================================
# === PRE-DIMENSIONALITY REDUCTION DATA VIZ
# =============================================
# region
# print("\n=== PRE-DIMENSIONALITY REDUCTION DATA VIZ")
# print("Some explorations with the data as-is before we reduce it\n")

# # Correlation heatmap for adults
# diagnosis_columns = [col for col in df_adults_with_resp_data.columns if col.startswith("Diagnosis_")]
# chest_location_columns = [col for col in df_adults_with_resp_data.columns if col.startswith("ChestLocation_")]
# specific_columns = ['Adult BMI (kg/m2)']  # Manually add this column

# # Combine the columns into one list
# columns_to_correlate = df_adults_with_resp_data[diagnosis_columns + chest_location_columns + specific_columns]
# corr_matrix = columns_to_correlate.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
# plt.title('Correlation Matrix for Adult Respiratory Data')
# plt.show()

# # endregion