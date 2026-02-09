
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def identify_key_columns(data):
    """
    Identifies key columns in the dataset based on their type.
    
    Args:
        data (pd.DataFrame): The input dataset.
        
    Returns:
        dict: A dictionary with lists of categorical and numerical column names.
    """
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return {
        'categorical': categorical_columns,
        'numerical': numerical_columns
    }

def drop_duplicates(data):
    """
    Removes duplicate rows from the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
        
    Returns:
        pd.DataFrame: Dataset with duplicate rows removed.
    """
    before = len(data)
    data = data.drop_duplicates()
    after = len(data)
    print(f"Removed {before - after} duplicate rows.")
    return data

def remove_outliers(data, numerical_columns):
    """
    Removes outliers from the dataset using the IQR method.
    
    Args:
        data (pd.DataFrame): The input dataset.
        numerical_columns (list): List of numerical column names.
        
    Returns:
        pd.DataFrame: Dataset with outliers removed.
    """
    before = len(data)
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    after = len(data)
    print(f"Removed {before - after} rows as outliers.")
    return data

def preprocess_data(filepath, target_column='Label'):
    """
    Preprocess the dataset and return processed features and target labels.
    """
    # Load dataset
    data = pd.read_csv(filepath)
    print("Initial dataset shape:", data.shape)

    # Encode the target column ('Label')
    data[target_column] = data[target_column].apply(lambda x: 0 if x == 'BENIGN' else 1)
    print("After encoding 'Label':", data.shape)

    # Drop duplicate rows
    data = data.drop_duplicates()
    print("After dropping duplicates:", data.shape)

    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    print("After handling missing/infinite values:", data.shape)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Normalize numerical data
    if len(X) > 0:  # Check if there are any rows left
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        raise ValueError("No data available for scaling. Check preprocessing steps.")

    return X_scaled, y
