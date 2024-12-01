from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from utils.add_weather_to_train import add_weather_to_train

"""_summary_ 
    this module cleans and preprocess the data according to the following worf-flow:
    1. numerate all data.
    2. scaling - using standardization or min-max.
    3. feature selection - using feature selection, feature engineering and PCA.
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
    # Convert 'physical_part_type' to numeric
    part_type_map = {'type1': 1, 'type2': 2}
    df['physical_part_type'] = df['physical_part_type'].map(part_type_map)
    df["status"] = df["status"].map({"OK": 1, "NOK": 0})
    return df


def scale_data(df, method):
    """_summary_
    scale the data according to the method given as input (min-max or standarization).
    
    Args:
        df: the dataset, pandas dataframe.
        method: string represents the chosen scaling method.
        
    Returns:
        df: the scaled pandas dataframe.
    """
    # Fill missing values with the mean for sensor columns
    categoral_columns = ['physical_part_id', 'message_timestamp', 'status', 'physical_part_type', 'weekday', 'shift']
    required_columns = [col for col in df.columns if col not in categoral_columns]
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
    if method == 2:
        scaler = MinMaxScaler()
    elif method == 1:
        scaler = StandardScaler()
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


def analyze_features(df, low_variance_threshold=0.01, high_correlation_threshold=0.95):
    """
    Filters features based on low variance and high correlation.

    Args:
        df (pd.DataFrame): Input dataset with features.
        low_variance_threshold (float): Threshold below which variance is considered low.
        high_correlation_threshold (float): Threshold above which correlation is considered high.
        
    Returns:
        pd.DataFrame: Cleaned dataset with filtered features.
        dict: Dictionary containing removed features and the reason for removal.
    """
    removed_features = {"low_variance": [], "high_correlation": []}
    # Step 1: Remove low-variance features
    exclude_columns = ["physical_part_type", "status", "shift", "physical_part_id", "weekday"]
    variance = df.drop(columns=exclude_columns).var()
    low_variance_features = variance[variance < low_variance_threshold].index.tolist()
    df = df.drop(columns=low_variance_features)
    removed_features["low_variance"] = low_variance_features
    # Step 2: Remove highly correlated features
    correlation_matrix = df.drop(columns=exclude_columns).corr()
    correlated_pairs = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > high_correlation_threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                # Keep the feature with higher variance
                col_to_remove = col1 if variance[col1] < variance[col2] else col2
                correlated_pairs.add(col_to_remove)
    df = df.drop(columns=list(correlated_pairs))
    removed_features["high_correlation"] = list(correlated_pairs)
    print(f"Removed {len(low_variance_features)} low-variance features.")
    print(f"Removed {len(correlated_pairs)} highly correlated features.") 
    return df, removed_features


def add_features(df):
    """_summary_
        perform feature engineering and add the following features:
        1. time in shift.
        2. temperature.
        3. humidity.

    Parameters:
    - df: pandas DataFrame containing scaled sensor data.
    
    Returns:
    - df: the Dataframe with the added features.
    """
    # Create "time in shift" from timestamp and shift
    df['message_timestamp'] = pd.to_datetime(df['message_timestamp'])
    shift_start_map = {
        1: 4.9167,  # Morning shift starts at 4:55 AM
        2: 13.4167,  # Afternoon shift starts at 1:25 PM
        3: 21.9167  # Night shift starts at 9:55 PM
    }
    df['shift_start'] = df['shift'].map(shift_start_map)
    df['time_in_shift'] = (
        df['message_timestamp'].dt.hour + df['message_timestamp'].dt.minute / 60
    ) - df['shift_start']
    df['time_in_shift'] = df['time_in_shift'].apply(lambda x: x if x >= 0 else x + 24)  # Adjust for midnight
    columns = list(df.columns)
    columns.insert(4, columns.pop(columns.index('time_in_shift')))
    df = df[columns]
    # Create "temp" and "humidity" columns
    df = add_weather_to_train(df, "filtered_weather_data.csv")
    # Drop columns that should be excluded
    exclude_columns = ['message_timestamp', 'shift_start']
    df = df.drop(columns=exclude_columns)
    return df


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
    exclude_columns = ['physical_part_id', 'status', 'physical_part_type']
    columns_to_filter = [col for col in df.columns if col not in exclude_columns]
    # Verify there are no missing values
    assert df[columns_to_filter].isnull().sum().sum() == 0, "There are still NaN values in the DataFrame after preprocessing!"
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
    # Create a DataFrame for the principal components with column names PC_1, PC_2, ..., PC_n
    pc_columns = [f"PC_{i+1}" for i in range(len(pca.components_))]
    pca_df = pd.DataFrame(principal_components, columns=pc_columns, index=df.index)
    # Add excluded columns back to the DataFrame
    for col in exclude_columns:
        pca_df.insert(0, col, df[col])
    return pca_df, pca


def preprocess_data(input_csv, output_csv, filter_method, scaling_method, pca_components=None, pca_variance_threshold=0.95):
    """
    Preprocess the data by numerating, scaling, and applying PCA, then save the preprocessed DataFrame to a CSV.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the preprocessed CSV file.
    - filter_method (int): 1 = baseline, 2 = added features, 3 = PCA
    - scaling_method (int): 1 = standard, 2 = min-max.
    - pca_components (int): Number of PCA components to keep. If None, uses variance threshold.
    - pca_variance_threshold (float): Proportion of variance to retain if pca_components is None. Defaults to 0.95.
    """
    print("Loading dataset.")
    df = pd.read_csv(input_csv)
    print("Numerating dataset.")
    df = numerate_df(df)
    print("Filtering columns withmore than 50% missing values.")
    df = filter_features(df)
    print("Scaling dataset.")
    df = scale_data(df, method=scaling_method)
    pca_model = None
    original_features = df.columns     
    if filter_method == 1:
        print("baseline - no added features.")
        # Drop columns that should be excluded
        exclude_columns = ['message_timestamp']
        df = df.drop(columns=exclude_columns)
    elif filter_method == 2:
        print("remove features with low beneficial value. \n \
            Add features - temp, humidity, time in shift.")
        df = add_features(df)
        df = scale_data(df, method=scaling_method)
        df, removed_features = analyze_features(df)
    elif filter_method == 3:
        print("Applying PCA.")
        exclude_columns = ['message_timestamp']
        df = df.drop(columns=exclude_columns)
        df, pca_model = pca_feature_extraction(df, n_components=pca_components, variance_threshold=pca_variance_threshold)
    print(f"Saving preprocessed data to {output_csv}.")
    df.to_csv(output_csv, index=False)
    print("Preprocessing complete and file saved!")
    return df, pca_model, original_features