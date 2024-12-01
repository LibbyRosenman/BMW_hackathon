import pandas as pd
import numpy as np
import shap
from dowhy import CausalModel
import matplotlib.pyplot as plt
import networkx as nx

"""_summary_
    This module performs a causal inferance on the data. The process includes:
    1. displays the results from the models
    2. Feature importance & PCA analysis
    3. Compare different preprocessing methods
    4. Causal inferance:
    4.a. Model the problem - create a DAG.
    4.b. identify estimand - number of NOK parts.
    4.c. estimate the effect.
    4.d. refute the estimate.
"""
def visualize_results(results):
    """
    Visualizes precision, recall, and F1 scores for different preprocessing methods.

    Args:
        results (list of dict): List of dictionaries containing results with keys:
                                "preprocessing_method", "precision", "recall", "f1_score".

    Returns:
        None: Displays the graph.
    """
    # Convert results to a DataFrame for easier plotting
    results_df = pd.DataFrame(results)
    # Set the index as preprocessing methods for better visualization
    results_df.set_index('preprocessing_method', inplace=True)
    # Plot the metrics
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Model Performance by Preprocessing Method")
    plt.ylabel("Score")
    plt.xlabel("Preprocessing Method")
    plt.legend(title="Metrics")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show(block=True)


def feature_importance_analysis(model, X_train, top_n_features=10):
    """
    Performs feature importance analysis on the model and dataset.
    
    Args:
        model: Trained random forest model.
        X_train: Training dataset (pd.DataFrame).
        top_n_features: Number of top features to extract.
        
    Returns:
        dict: Dictionary of top features and their importance scores.
    """
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(top_n_features)
    print("Top Features Based on Importance:")
    print(feature_importance)
    importance_dict = feature_importance.set_index('Feature').to_dict()['Importance']
    return importance_dict


def shap_analysis(model, X_train, top_n_features=10):
    """
    Performs SHAP analysis on the trained model and dataset.
    
    Args:
        model: Trained random forest model.
        X_train: Training dataset.
        top_n_features: Number of top features to analyze.
        
    Returns:
        dict: Mean SHAP values for the top features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    # Mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values[1]).mean(axis=0)
    shap_summary = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean SHAP Value': mean_shap_values
    }).sort_values(by='Mean SHAP Value', ascending=False).head(top_n_features)
    print("SHAP Analysis Completed. Top Features:")
    print(shap_summary)
    shap_dict = shap_summary.set_index('Feature').to_dict()['Mean SHAP Value']
    return shap_dict


def find_original_features(pca, feature_names, pc_index, top_n=5):
    """
    Find the original features contributing to a given principal component.

    Args:
        pca: Trained PCA object (from sklearn.decomposition.PCA).
        feature_names: List of original feature names.
        pc_index: Index of the principal component to analyze.
        top_n: Number of top contributing features to return (default: 10).

    Returns:
        A sorted DataFrame of features and their contributions to the given PC.
    """
    if not hasattr(pca, "components_"):
        raise ValueError("The provided PCA object does not have components_. Make sure it's trained.")
    if pc_index < 0 or pc_index >= pca.components_.shape[0]:
        raise ValueError(f"Invalid pc_index. Must be between 0 and {pca.components_.shape[0] - 1}.")
    # Get the loading vector for the selected principal component
    loading_vector = pca.components_[pc_index]
    # Pair feature names with their corresponding loadings
    exclude_columns = ['physical_part_id', 'status', 'physical_part_type', 'message_timestamp']
    feature_names = [feature for feature in feature_names if feature not in exclude_columns]
    feature_contributions = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": loading_vector
    })
    # Sort by absolute contribution
    feature_contributions["Absolute_Contribution"] = feature_contributions["Contribution"].abs()
    feature_contributions = feature_contributions.sort_values(by="Absolute_Contribution", ascending=False)
    # Return the top_n features
    return feature_contributions.head(top_n)


def run_analysis(random_forest_model, X_train, is_pca, pca_model=None, original_features=None):
    """
    Performs feature importance, SHAP analysis and PCA feature retreival on the trained model and dataset.
    
    Args:
        random_forest_model: Trained random forest model.
        X_train: Training dataset (pd.DataFrame).
        top_n_features: Number of top features to consider for insights.
        is_pca: True if the filtering method is pca.
        pca_model: relevant if is_pca = True.
        original_features: relevant if is_pca = True.
        
    Returns:
        dict: Consolidated insights with top features from importance and PCA analysis.
    """
    print("Starting feature Analysis...")
    # Feature Importance
    feature_importance = feature_importance_analysis(random_forest_model, X_train)
    # SHAP Analysis
    # shap_values = shap_analysis(random_forest_model, X_train)
    # If PCA - retrieve the original features
    if is_pca:
        original_important_features = {}
        for pc in feature_importance:
            original_important_features += find_original_features(pca_model, original_features, int(pc.split('_')[1]))
        feature_importance = original_important_features
    print("All Feature Analysis Metrics Completed.")
    return feature_importance
    

def find_best_method(results, weights=(0.5, 0.3, 0.2)):
    """
    Selects the best model based on a weighted combined score.
    
    Args:
        results (list of dict): List of dictionaries, each containing 'precision', 'recall', and 'F1' scores.
        weights (tuple): Weights for F1, recall, and precision in that order (default: 50%, 30%, 20%).
        
    Returns:
        dict: The dictionary with the highest weighted score.
    """
    best_model = None
    best_score = -1
    for result in results:
        # Extract scores
        f1 = result.get('F1', 0)
        recall = result.get('recall', 0)
        precision = result.get('precision', 0)
        # Calculate weighted score
        weighted_score = (weights[0] * f1) + (weights[1] * recall) + (weights[2] * precision)
        # Compare and update the best model
        if weighted_score > best_score:
            best_score = weighted_score
            best_model = result
    return best_model


def create_dag(data, features, confounders):
    """
    This function creates a DAG for the causal model using DoWhy.
    The effect is 'status'.
    
    Args:
        X: the dataset.
        features: A list of column names representing the most important features.
        confounders: A dictionary:
            - Keys are features that might affect other features or the effect.
            - Values are lists of important features affected by the specific key.
    
    Returns:
        - model: A DoWhy causal model object.
    """
    dag = nx.DiGraph()
    # Add edges from features to outcome
    for feature in features:
        dag.add_edge(feature, "status")
    # Add edges for confounders
    for confounder, affected_features in confounders.items():
        for feature in affected_features:
            dag.add_edge(confounder, feature)
    # Convert NetworkX graph to DOT string
    graph_str = nx.nx_pydot.to_pydot(dag).to_string()
    return graph_str


def single_feature_causal_inferance(data, dag, feature, outcome="status", method="backdoor.linear_regression"):
    """
    Analyze the causal effect of a single feature on the outcome.

    Args:
        data: The dataset (Pandas DataFrame).
        dag: The causal DAG created with DoWhy.
        feature: The feature to analyze.
        outcome: The outcome variable (default is "status").
        method: The estimation method (default is backdoor.linear_regression).

    Returns:
        A dictionary containing the feature, effect size, and textual explanation.
    """
    # Build causal model
    causal_model = CausalModel(
        data=data,
        treatment=feature,
        outcome=outcome,
        graph=dag
    )
    # Identify estimand
    estimand = causal_model.identify_effect()
    # Estimate effect
    estimate = causal_model.estimate_effect(
        identified_estimand=estimand,
        method_name=method
    )
    # Textual explanation
    explanation = (
        f"For feature '{feature}', the causal effect on '{outcome}' is estimated to be {estimate.value:.4f}.\n"
        f"This means that a unit increase in '{feature}' is expected to change '{outcome}' by {estimate.value:.4f} units."
    )
    return {"feature": feature, "effect": estimate.value, "explanation": explanation}


def multiple_feature_causal_inferance(dag, features, outcome="status"):
    """
    Analyze the causal effects of multiple features on the outcome.

    Args:
        dag: The causal DAG created with causalnex.
        features: A list of features to analyze.
        outcome: The outcome variable (default is "status").

    Returns:
        A list of dictionaries containing features, effect sizes, and textual explanations.
    """
    results = []
    for feature in features:
        result = single_feature_causal_inferance(dag, feature, outcome)
        results.append(result)
    return results


def comprehensive_causal_inferance(data, dag, features, k=3, outcome="status"):
    """
    Perform comprehensive causal analysis.

    Args:
        data: The dataset (Pandas DataFrame).
        dag: The causal DAG.
        features: A list of features to analyze.
        k: The number of top features to include in the multiple feature analysis.
        outcome: The outcome variable (default is "status").

    Returns:
        A dictionary containing results of single and multiple feature analyses and textual summaries.
    """
    # Perform single feature analysis
    single_results = [
        single_feature_causal_inferance(data, dag, feature, outcome) for feature in features
    ]
    # Rank features by effect size (absolute value)
    single_results_sorted = sorted(single_results, key=lambda x: abs(x["effect"]), reverse=True)
    # Select top-k features for multiple feature analysis
    top_features = [result["feature"] for result in single_results_sorted[:k]]
    # Perform multiple feature analysis on top-k features
    # multiple_results = multiple_feature_causal_inferance(data, dag, top_features, outcome)
    # Textual explanation
    summary = (
        f"Top {k} features with the largest effects:\n" +
        "\n".join(
            [f"Feature '{res['feature']}' with effect size {res['effect']:.4f}." for res in single_results_sorted[:k]]
        ) 
        # "\n\nCombined Analysis of Top Features:\n" +
        # "\n".join([res["explanation"] for res in multiple_results])
    )
    return {
        "single_results": single_results,
        # "multiple_results": multiple_results,
        "summary": summary,
    }