from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

"""_summary_ 
    this module cleans and preprocess the data according to the following worf-flow:
    1. numerate all data (in order to perform feature extraction).
    2. scaling - using standardization.
    3. feature selection - using PCA.
"""


def numerate_df(df):
    """
    Convert non-numerical columns in the DataFrame to numerical values.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    
    Returns:
    - A preprocessed DataFrame with numeric columns ready for further analysis.
    """
    # Convert 'weekday' to numeric
    weekday_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                   'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['weekday'] = df['weekday'].map(weekday_map)

    # Convert 'shift' to numeric (fruhschicht = 1, spaetschicht = 2, Nachtschicht = 3)
    shift_map = {'Fruehschicht': 1, 'Spaetschicht': 2, 'Nachtschicht': 3}
    df['shift'] = df['shift'].map(shift_map)
    
    # Create "time in shift" from timestamp and shift
    df['message_timestamp'] = pd.to_datetime(df['message_timestamp'])
    shift_start_map = {
        1: 5,  # Morning shift starts at 5 AM
        2: 13.5,  # Afternoon shift starts at 1:30 PM
        3: 22  # Night shift starts at 10 PM
    }
    df['shift_start'] = df['shift'].map(shift_start_map)
    df['time_in_shift'] = (
        df['message_timestamp'].dt.hour + df['message_timestamp'].dt.minute / 60
    ) - df['shift_start']
    df['time_in_shift'] = df['time_in_shift'].apply(lambda x: x if x >= 0 else x + 24)  # Adjust for midnight
    columns = list(df.columns)
    columns.insert(4, columns.pop(columns.index('time_in_shift')))
    df = df[columns]
    
    # Convert 'physical_part_type' to numeric
    part_type_map = {'type1': 1, 'type2': 2}
    df['physical_part_type'] = df['physical_part_type'].map(part_type_map)

    # Drop columns that should be excluded
    exclude_columns = ['message_timestamp', 'shift_start']
    df = df.drop(columns=exclude_columns)
    return df



def scale_data(df, method="standard"):
    """_summary_
    scale the data according to the method given as input (min-max or standarization).
    Args:
        df: the dataset, pandas dataframe.
        method: string represents the chosen scaling method. Defaults to "standard".
    Returns:
        df: the scaled pandas dataframe.
    """
    # Drop columns with no values
    df = df.dropna(axis=1, how='all')
    # Fill missing values with the mean for sensor columns
    exclude_columns = ['physical_part_id', 'status']
    required_columns = [col for col in df.columns if col not in exclude_columns]
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
    if method == "min-max":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'min-max' or 'standard'.")
    
    # Scale columns
    df[required_columns] = scaler.fit_transform(df[required_columns])
    return df


def filter_features(df, threshold=0.5):
    """_summary_
    Filter out columns that have many missing values.
    
    Parameters:
    - df: pandas DataFrame containing scaled sensor data.
    - variance_threshold: float, precentage of missing values the df can handle.
    
    Returns:
    - df - the df where all columns have at least 50% non-missing values.
    """
    columns_to_drop = []
    for col in df.columns:
        if df[col].isnull().sum() / len(df) >= threshold:
            columns_to_drop.append(col)
    filtered_df = df.drop(columns=columns_to_drop)
    return filtered_df

def pca_feature_extraction(df, n_components=None, variance_threshold=0.95):
    """_summary_
    Apply PCA to reduce dimensionality of the dataset.
    
    Parameters:
    - df: pandas DataFrame containing scaled sensor data.
    - n_components: int, number of principal components to keep (optional).
    - variance_threshold: float, proportion of variance to retain if n_components is not specified.
    
    Returns:
    - A tuple (pca_df, pca), where:
      - pca_df: DataFrame of principal components.
      - pca: Trained PCA model.
    """
    exclude_columns = ['physical_part_id', 'status']
    columns_to_filter = [col for col in df.columns if col not in exclude_columns]
    
    # Verify there are no missing values
    assert df.isnull().sum().sum() == 0, "There are still NaN values in the DataFrame after preprocessing!"

    # apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[columns_to_filter])
    if n_components is None:
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        n_components = (cumulative_variance < variance_threshold).sum() + 1
        print(f"Selected {n_components} components explaining {variance_threshold*100}% variance")
        
        # Re-run PCA with selected components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df[columns_to_filter])
    
    # Create a DataFrame for the principal components using original feature names
    pc_columns = [f"PC_{columns_to_filter[abs(pca.components_[i]).argmax()]}"
    for i in range(len(pca.components_))]
    pca_df = pd.DataFrame(principal_components, columns=pc_columns, index=df.index)
    
    # Add excluded columns back to the DataFrame
    for col in exclude_columns:
        pca_df.insert(0, col, df[col])
    
    return pca_df, pca