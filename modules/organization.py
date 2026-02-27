# -------------------------------------------------
# Organization module of Project to download and organize the data
# -------------------------------------------------

# ----------------------------
# Imports
# ----------------------------
from pathlib import Path
import subprocess
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

def download_ptbxl(data_dir: Path):
    """
    Downloads the PTB-XL dataset from PhysioNet S3 if it does not already exist.

    Parameters:
        data_dir (Path): Directory to store the dataset.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if folder has files already
    if any(data_dir.iterdir()):
        print(f"Data directory '{data_dir}' already exists and is not empty. Skipping download.")
        return

    s3_bucket = "s3://physionet-open/ptb-xl/1.0.3/"
    print(f"Downloading PTB-XL data from {s3_bucket} into {data_dir} ...")

    try:
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", s3_bucket, str(data_dir)],
            check=True
        )
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print("Download failed. Check AWS CLI installation and network connection.")
        print(e)

#----------------------------
# ----------------------------
# Split Function
# ----------------------------
def create_train_val_test_split(df, Y, patient_id_col="patient_id", random_state=42, save=False, save_dir=Path.cwd() / "splits"):
    """
    Assigns a 'split' column to the dataframe using Multilabel Stratified Shuffle Split.
    Saves train/val/test patient IDs as CSVs in a folder called 'splits'.

    Parameters:
        df (pd.DataFrame): The main dataframe containing patient IDs.
        Y (np.ndarray): Multi-label matrix corresponding to each patient.
        patient_id_col (str): Column name containing patient IDs.
        random_state (int): Random seed for reproducibility.
        save_dir (Path): Directory to save the split CSV files.

    Returns:
        pd.DataFrame: Original df with a new 'split' column ('train', 'val', 'test').
    """
    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure a split column exists
    df['split'] = "train"
    
    # Get unique patient IDs
    patients = df[patient_id_col].unique()

    # First split: 70% train, 30% temp
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(msss.split(patients, Y))
    train_patients = patients[train_idx]
    temp_patients = patients[temp_idx]

    # Second split: split temp into val/test 50-50
    Y_temp = Y[temp_idx]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_idx, test_idx = next(msss2.split(temp_patients, Y_temp))
    val_patients = temp_patients[val_idx]
    test_patients = temp_patients[test_idx]

    # Assign splits
    df.loc[df[patient_id_col].isin(val_patients), 'split'] = 'val'
    df.loc[df[patient_id_col].isin(test_patients), 'split'] = 'test'

    # ----------------------------
    # Save the splits
    # ----------------------------
    # Optionally save CSVs
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_patients, columns=[patient_id_col]).to_csv(save_dir / "train_patients.csv", index=False)
        pd.DataFrame(val_patients, columns=[patient_id_col]).to_csv(save_dir / "val_patients.csv", index=False)
        pd.DataFrame(test_patients, columns=[patient_id_col]).to_csv(save_dir / "test_patients.csv", index=False)
        print(f"Splits saved successfully to {save_dir}")
    
    return df, train_patients, val_patients, test_patients


#----------------------------
# Helper Functions 
# ----------------------------
# Helper function to extract superclasses from SCP codes
def extract_superclasses(scp_dict, code_to_superclass):
    """
    Extracts superclass labels from a dictionary of SCP codes.
    
    Parameters:
        scp_dict (dict): Dictionary of SCP codes for a single ECG.
        code_to_superclass (dict): Mapping from SCP codes to superclasses.
        
    Returns:
        list: List of unique superclasses for this ECG.
    """
    classes = set()
    for code in scp_dict.keys():
        if code in code_to_superclass:
            classes.add(code_to_superclass[code])
    return list(classes)