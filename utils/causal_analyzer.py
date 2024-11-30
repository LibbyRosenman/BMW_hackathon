import pandas as pd
import numpy as np
import shap
from dowhy import CausalModel
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import matplotlib.pyplot as plt
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt

"""_summary_
    This module performs a causal inferance on the data. The process includes:
    1. displays the results from the models
    2. Feature importance & SHAP analysis
    3.Compare different models
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
    plt.show()


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
    import shap
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


def run_analysis(random_forest_model, X_train):
    """
    Performs feature importance and SHAP analysis on the trained model and dataset.
    
    Args:
        random_forest_model: Trained random forest model.
        X_train: Training dataset (pd.DataFrame).
        top_n_features: Number of top features to consider for insights.
        
    Returns:
        dict: Consolidated insights with top features from importance and SHAP analysis.
    """
    print("Starting feature Analysis...")
    # Feature Importance
    feature_importance = feature_importance_analysis(random_forest_model, X_train)
    # SHAP Analysis
    shap_values = shap_analysis(random_forest_model, X_train)
    print("All Feature Analysis Metrics Completed.")
    return [feature_importance, shap_values]
    

def find_best_model(results, weights=(0.5, 0.3, 0.2)):
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


def create_dag(features, confounders):
    """_summary_
    this function creates a DAG for the causal model.
    effect = status
    
    Args:
        features: a list of column names of the most important features.
        confounders: a dictionary: 
                keys are features who might affect other features or the affect.
                values are lists of the important features who are affected by a specific key. 
    
    Returns:
    - DAG: the directed graph.
    """
    dag = StructureModel()
    nodes = ["status"] + features +  confounders
    dag.add_nodes_from(nodes)
    for cause in features:
        dag.add_edge(cause, "status")
    for confounder in confounders.keys():
        for effect in confounders.get(confounder):
            dag.add_edge(confounder, effect)
    # Visualize the DAG
    viz = plot_structure(dag, graph_attributes={"scale": "1.5"})
    viz.view()
    return dag
    

def single_feature_causal_inferance(dag, feature, outcome="status", method="backdoor.linear_regression"):
    """
    Analyze the causal effect of a single feature on the outcome.

    Args:
        dag: The causal DAG created with causalnex.
        feature: The feature to analyze.
        outcome: The outcome variable (default is "status").

    Returns:
        A dictionary containing the feature, effect size, and textual explanation.
    """
    # Build causal model
    causal_model = CausalModel(
        data=None,  # Provide the data used for the DAG
        treatment=feature,
        outcome=outcome,
        graph=dag.to_dot()
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


def comprehensive_causal_inferance(dag, features, k=3, outcome="status"):
    """
    Perform comprehensive causal analysis.

    Args:
        dag: The causal DAG created with causalnex.
        features: A list of features to analyze.
        k: The number of top features to include in the multiple feature analysis.
        outcome: The outcome variable (default is "status").

    Returns:
        A dictionary containing results of single and multiple feature analyses and textual summaries.
    """
    # Perform single feature analysis
    single_results = []
    for feature in features:
        result = single_feature_causal_inferance(dag, feature, outcome)
        single_results.append(result)
    # Rank features by effect size (absolute value)
    single_results_sorted = sorted(single_results, key=lambda x: abs(x["effect"]), reverse=True)
    # Select top-k features for multiple feature analysis
    top_features = [result["feature"] for result in single_results_sorted[:k]]
    # Perform multiple feature analysis on top-k features
    multiple_results = multiple_feature_causal_inferance(dag, top_features, outcome)
    # Textual explanation
    summary = (
        f"Top {k} features with the largest effects:\n" +
        "\n".join(
            [f"Feature '{res['feature']}' with effect size {res['effect']:.4f}." for res in single_results_sorted[:k]]
        ) +
        "\n\nCombined Analysis of Top Features:\n" +
        "\n".join([res["explanation"] for res in multiple_results])
    )
    return {
        "single_results": single_results,
        "multiple_results": multiple_results,
        "summary": summary,
    }


def visualize_dag(dag, features, outcome="status", highlight_color="red", title="Causal Graph"):
    """
    Visualize the causal DAG highlighting the specified features.

    Args:
        dag: The causal DAG created with causalnex.
        features: A list of features whose causal links to the outcome should be highlighted.
        outcome: The outcome variable (default is "status").
        highlight_color: Color for highlighting causal links (default is "red").
        title: Title of the visualization graph.
    """
    # Generate positions for nodes
    pos = nx.spring_layout(dag)
    # Create the plot
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(dag, pos, node_color="lightblue", node_size=5000)
    # Highlight edges for the specified features
    nx.draw_networkx_edges(dag, pos, edgelist=[(feature, outcome) for feature in features], edge_color=highlight_color, width=2)
    # Draw all other edges with low opacity
    nx.draw_networkx_edges(dag, pos, edgelist=[e for e in dag.edges if e[1] != outcome or e[0] not in features], alpha=0.3)
    nx.draw_networkx_labels(dag, pos, font_size=12, font_color="black")
    plt.title(title, fontsize=16)
    plt.show()
