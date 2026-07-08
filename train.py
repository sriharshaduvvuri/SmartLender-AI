"""
Smart Lender - Model Training and Evaluation Pipeline
This script automates the complete ML workflow:
1. Downloads the standard Loan Prediction dataset.
2. Performs data understanding, cleanup, and preprocessing.
3. Conducts Exploratory Data Analysis (EDA) and saves dashboard visualizations.
4. Handles class imbalance using SMOTE.
5. Trains four classification models: Decision Tree, Random Forest, KNN, and XGBoost.
6. Evaluates and compares models, saving them and the best performing model.
"""

import os
import urllib.request
import logging
from typing import Dict, Tuple, Any

# Configure headless Matplotlib before imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import joblib

# Scikit-learn and ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "loan_prediction.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "images")

# Ensure required directories exist
for folder in [DATASET_DIR, MODELS_DIR, STATIC_IMG_DIR]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Verified directory: {folder}")


def download_dataset(url: str, dest_path: str) -> None:
    """
    Downloads the dataset from a remote URL if it does not already exist.

    Args:
        url: Remote URL of the raw CSV file.
        dest_path: Absolute local path to save the CSV.
    """
    if not os.path.exists(dest_path):
        logger.info(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info(f"Dataset successfully downloaded and saved to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise e
    else:
        logger.info("Dataset already exists locally. Skipping download.")


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], StandardScaler]:
    """
    Cleans features, fills missing values, encodes categories, and fits the scaler.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        A tuple containing:
            - Preprocessed feature DataFrame X and label Series y
            - Mappings used for encoding (saved in encoder.pkl)
            - Fitted StandardScaler instance (saved in scaler.pkl)
    """
    logger.info("Starting data preprocessing...")

    # Drop Loan_ID as it is not a feature
    df_clean = df.drop(columns=['Loan_ID'], errors='ignore').copy()

    # Define categorical and numerical columns
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # Impute missing values
    # For categorical features, use the mode (most frequent value)
    for col in cat_cols:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        logger.debug(f"Imputed missing values for {col} using mode: {mode_val}")

    # For numerical features, use median/mode
    # LoanAmount: use median
    loan_amount_median = df_clean['LoanAmount'].median()
    df_clean['LoanAmount'] = df_clean['LoanAmount'].fillna(loan_amount_median)
    
    # Loan_Amount_Term: use mode
    loan_term_mode = df_clean['Loan_Amount_Term'].mode()[0]
    df_clean['Loan_Amount_Term'] = df_clean['Loan_Amount_Term'].fillna(loan_term_mode)

    # Standardize values in categorical columns
    df_clean['Dependents'] = df_clean['Dependents'].astype(str).str.replace('+', '', regex=False)
    
    # Define mapping dictionary for categorical encoding
    encoding_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'Y': 1, 'N': 0}
    }

    # Map categorical columns to numerical keys
    for col, mapping in encoding_mappings.items():
        if col in df_clean.columns:
            # Coerce mapping values and cast to integers
            df_clean[col] = df_clean[col].map(mapping).astype(int)

    # Separate features and target
    X = df_clean.drop(columns=['Loan_Status'])
    y = df_clean['Loan_Status']

    # Fit numerical scaler
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    logger.info("Data preprocessing completed successfully.")
    return X, y, encoding_mappings, scaler


def generate_eda_visualizations(df: pd.DataFrame) -> None:
    """
    Generates exploratory data analysis plots for use in the dashboard.

    Args:
        df: Raw DataFrame containing the loan dataset.
    """
    logger.info("Generating EDA dashboard plots...")
    sns.set_theme(style="darkgrid")

    # Colors suitable for premium look
    primary_color = "#3a86ff"
    secondary_color = "#ff006e"

    # 1. Distribution of Loan Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Loan_Status', data=df, palette=[primary_color, secondary_color])
    plt.title("Loan Approval Status Count", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Approved (Y) / Rejected (N)", fontsize=11)
    plt.ylabel("Count of Applicants", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "loan_status_dist.png"), dpi=150)
    plt.close()

    # 2. Credit History vs. Loan Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, palette=[secondary_color, primary_color])
    plt.title("Credit History influence on Loan Status", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Credit History (1.0 = Good, 0.0 = Bad)", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend(["Rejected", "Approved"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "credit_history_vs_status.png"), dpi=150)
    plt.close()

    # 3. Income vs Loan Amount by Education
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(
        x='ApplicantIncome', 
        y='LoanAmount', 
        hue='Education', 
        style='Loan_Status',
        data=df, 
        palette=[primary_color, secondary_color],
        alpha=0.8
    )
    plt.title("Income vs Loan Amount", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Applicant Income ($)", fontsize=11)
    plt.ylabel("Loan Amount (Thousands)", fontsize=11)
    # Clip extreme income outliers for visualization clarity
    plt.xlim(0, 30000)
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "income_vs_loanamount.png"), dpi=150)
    plt.close()

    logger.info("EDA visualizations saved in static/images.")


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Splits data, handles class balance using SMOTE, trains 4 classifiers, 
    and returns evaluation results.

    Args:
        X: Preprocessed feature DataFrame.
        y: Target label Series.

    Returns:
        Dictionary of trained model instances and their performance metrics.
    """
    logger.info("Splitting dataset and balancing classes with SMOTE...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Balance target classes in training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Original shape: {X_train.shape}, Balanced training shape: {X_train_res.shape}")

    # Define model classifiers
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        model.fit(X_train_res, y_train_res)
        preds = model.predict(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        results[name] = {
            'model_instance': model,
            'metrics': {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1-Score': float(f1)
            }
        }
        logger.info(f"{name} Results - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    return results


def save_models_and_plots(results: Dict[str, Any], scaler: StandardScaler, encoder_mapping: Dict[str, Dict[str, int]]) -> None:
    """
    Saves trained models, fitted preprocessors, and comparison performance chart.

    Args:
        results: Dictionary containing the trained models and metrics.
        scaler: The fitted numerical StandardScaler instance.
        encoder_mapping: The categorical feature mapping dictionary.
    """
    logger.info("Saving preprocessing transformers...")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoder_mapping, os.path.join(MODELS_DIR, "encoder.pkl"))

    # Map model names to expected output filenames
    filename_map = {
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'XGBoost': 'xgboost.pkl'
    }

    best_score = -1.0
    best_model_name = ""
    best_model_instance = None

    # Save individual models and find the best one based on Accuracy
    for name, data in results.items():
        model_inst = data['model_instance']
        metrics = data['metrics']
        filename = filename_map[name]
        
        joblib.dump(model_inst, os.path.join(MODELS_DIR, filename))
        logger.info(f"Saved {name} model to {filename}")

        # Track the model with the highest validation accuracy
        if metrics['Accuracy'] > best_score:
            best_score = metrics['Accuracy']
            best_model_name = name
            best_model_instance = model_inst

    # Save the absolute best model
    if best_model_instance is not None:
        joblib.dump(best_model_instance, os.path.join(MODELS_DIR, "best_model.pkl"))
        logger.info(f"Saved best model ({best_model_name} with Accuracy {best_score:.4f}) to best_model.pkl")

    # Generate and save model comparison chart
    model_names = list(results.keys())
    accuracies = [results[m]['metrics']['Accuracy'] for m in model_names]
    f1_scores = [results[m]['metrics']['F1-Score'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#3a86ff')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#ff006e')

    plt.title('Classifier Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Algorithm', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.xticks(x, model_names)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved performance comparison chart to static/images/model_comparison.png")


def main() -> None:
    """
    Main orchestrator function for dataset retrieval and training.
    """
    logger.info("Starting training orchestration pipeline...")
    
    # Module 2: Dataset Collection
    download_dataset(DATASET_URL, DATASET_PATH)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded. Dimensions: {df.shape}")

    # Module 3-4: Data Understanding & EDA Visualizations
    generate_eda_visualizations(df)

    # Module 5: Data Preprocessing
    X, y, mappings, scaler = preprocess_data(df)

    # Module 6-7: Model Building & Evaluation
    training_results = train_and_evaluate_models(X, y)

    # Module 8: Save Preprocessors and Trained Models
    save_models_and_plots(training_results, scaler, mappings)
    
    logger.info("Training pipeline completed successfully! All artifacts generated.")


if __name__ == "__main__":
    main()
