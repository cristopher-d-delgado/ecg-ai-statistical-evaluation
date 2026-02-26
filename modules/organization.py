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

def download_ptbxl_data(s3_bucket="s3://physionet-open/ptb-xl/1.0.3/", data_dir=Path.cwd() / "data"):
    """"
    Download the PTB-XL dataset from PhysioNet. 
    Retrives the dataset from the Amazon S3 bucket where it is hosted and saves it to a local directory.

    Parameters:
        s3_bucket (str): The S3 bucket URL where the PTB-XL dataset is hosted. Default is "s3://physionet-open/ptb-xl/1.0.3/".
        data_dir (str): The local directory where the dataset should be saved. Default is the current working directory under a folder named "data".
    """
    # Create data directory if needed
    if not data_dir.exists():
        print(f"Creating data directory at {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Data directory already exists at {data_dir}")

    # Download dataset using AWS CLI
    print("Starting dataset download from S3...")
    try:
        result = subprocess.run(
            [
                "aws", "s3", "sync",
                "--no-sign-request",
                s3_bucket,
                str(data_dir)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print("Download failed.")
        print("Error message:")
        print(e.stderr)

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