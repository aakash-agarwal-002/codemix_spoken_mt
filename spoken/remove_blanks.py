import pandas as pd
import glob
import os

# Folder containing CSV files
input_folder = "/home/cfiltlab/aakash.agarwal/ai4bharat/spoken/outputs"
output_folder = "/home/cfiltlab/aakash.agarwal/ai4bharat/spoken/outputs_clean"

os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    
    df = pd.read_csv(file)
    
    # Remove rows with NaN in any column
    df = df.dropna()
    
    # Remove rows where any column contains empty string or only spaces
    df = df[df.apply(lambda row: all(str(cell).strip() != "" for cell in row), axis=1)]
    
    # Save cleaned file
    filename = os.path.basename(file)
    df.to_csv(os.path.join(output_folder, filename), index=False)

print("All files cleaned successfully.")
