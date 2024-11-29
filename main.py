import pandas as pd
from utils.clean_data import numerate_df, scale_data, pca_feature_extraction
from utils.classifier import train_and_evaluate_model, generate_predictions_and_save, analyze_feature_importance

def preprocess_data(input_csv, output_csv, scaling_method="standard", pca_components=None, pca_variance_threshold=0.95):
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

    print("Applying PCA...")
    pca_df, pca_model = pca_feature_extraction(df, n_components=pca_components, variance_threshold=pca_variance_threshold)

    print(f"Saving preprocessed data to {output_csv}...")
    pca_df.to_csv(output_csv, index=False)
    print("Preprocessing complete and file saved!")


if __name__ == "__main__":
    # Step 1: Preprocess data
    input_csv = r"C:\Users\libby\study\semesterE\Exchange\BMW hackathon\train.csv"
    output_csv = r"C:\Users\libby\study\semesterE\Exchange\BMW hackathon\preprocess_train.csv"

    # Assuming this preprocess function exists in your preprocessing script
    cleaned_df = preprocess_data(input_csv, output_csv, scaling_method="standard", pca_components=None, pca_variance_threshold=0.95)

    # Step 2: Split into X (features) and y (target)
    X_train = cleaned_df.drop(columns=["status"])  # Features (all except 'status')
    y_train = cleaned_df["status"]  # Target labels (status)

    # Now you have X_train and y_train ready for model training

    # Step 3: Call functions from predict.py

    # Call the training and evaluation function
    model = train_and_evaluate_model(X_train, y_train)

    # If you have test data and want to generate predictions:
    test_data = pd.read_csv("test.csv")  # Load your test data
    generate_predictions_and_save(model, test_data, "test.csv")

    # Analyze feature importance
    analyze_feature_importance(model, X_train)



