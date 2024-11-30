import pandas as pd
from utils.clean_data import numerate_df, scale_data, pca_feature_extraction
from utils.classifier import evaluate_model_kfold, generate_predictions_and_save, analyze_feature_importance
from utils.clustering import perform_clustering, perform_clustering_NOK, compare_ok_nok

def preprocess_data(input_csv, output_csv, filter_method=1, scaling_method="standard", pca_components=None, pca_variance_threshold=0.95):
    """
    Preprocess the data by numerating, scaling, and applying PCA, then save the preprocessed DataFrame to a CSV.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the preprocessed CSV file.
    - scaling_method (str): Scaling method to use ("standard" or "min-max"). Defaults to "standard".
    - pca_components (int): Number of PCA components to keep. If None, uses variance threshold.
    - pca_variance_threshold (float): Proportion of variance to retain if pca_components is None. Defaults to 0.95.
    """
    print("Loading dataset...")
    df = pd.read_csv(input_csv)

    print("Numerating dataset...")
    df = numerate_df(df)

    print("Scaling dataset...")
    df = scale_data(df, method=scaling_method)

    if filter_method == 1:
        print("Filtering columns withmore than 50% missing values...")
        old_m = len(df.columns)
        print("df has " + str(old_m) + "columns.")
        df = filter_features(df)
        new_m = len(df.columns)
        print("after filtering df has " + str(new_m) + "columns.")
    elif filter_method == 2:
        print("Applying PCA...")
        df, pca_model = pca_feature_extraction(df, n_components=pca_components, variance_threshold=pca_variance_threshold)

    print(f"Saving preprocessed data to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Preprocessing complete and file saved!")
    return df


if __name__ == "__main__":
    # Step 1: Preprocess data
    input_csv = r"train.csv"
    output_csv = r"preprocess_train.csv"

    # Assuming this preprocess function exists in your preprocessing script
    cleaned_df = preprocess_data(input_csv, output_csv, scaling_method="standard", pca_components=None, pca_variance_threshold=0.95)

    # Step 2: Split into X (features) and y (target)
    X_train = cleaned_df.drop(columns=["status", "physical_part_id"])  # Features (all except 'status')
    y_train = cleaned_df["status"].map({"OK": 1, "NOK": 0})

    # Step 3: Call functions from predict.py

    # Call the training and evaluation function
    model = evaluate_model_kfold(X_train, y_train)

    # If you have test data and want to generate predictions:
    test_data = pd.read_csv("test.csv")  # Load your test data
    test_data = preprocess_data(test_data, output_csv="preprocessed_test.csv", scaling_method="standard", pca_components=None, pca_variance_threshold=0.95)

    generate_predictions_and_save(model, test_data, "test.csv")

    # Analyze feature importance
    analyze_feature_importance(model, X_train)
