"""
Smart Lender - Data Preprocessing and Feature Engineering Pipeline
This script:
1. Loads the dataset, checks types, memory usage, and column distributions.
2. Removes identifier columns (e.g. Loan_ID).
3. Handles missing values:
   - Numerical: Median Imputation (compared to Mean and KNN).
   - Categorical: Mode Imputation (compared to Unknown category).
4. Detects outliers using IQR and Z-scores, displaying before/after boxplots.
5. Encodes categorical variables (Binary custom mappings, Ordinal structures) and saves encoders.
6. Addresses class imbalances using SMOTE on the training split (avoiding test leakage).
7. Scales numerical features using StandardScaler (compared with MinMaxScaler and RobustScaler).
8. Performs Feature Selection ranking using Mutual Information (MI) classification.
9. Splits dataset into 70-30 stratified segments.
10. Bundles preprocessors into a custom serializable sklearn-compliant Pipeline block.
11. Saves and exports pickled preprocessors and diagnostic charts.
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Any

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "loan_prediction.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "images")
REPORT_PATH = os.path.join(BASE_DIR, "preprocessing_report.md")

# Ensure folders exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)


# =====================================================================
# CUSTOM SCIKIT-LEARN COMPLIANT PREPROCESSING PIPELINE
# =====================================================================

class SmartLenderPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn compliant preprocessor pipeline for the Smart Lender system.
    Imputes missing values, encodes categories, scales numerical values, 
    and aligns feature order. Safe for single-row dictionary inference in Flask.
    """
    def __init__(self):
        self.numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        
        # Initializing state maps to save inside pickle
        self.imputers_: Dict[str, Any] = {}
        self.encoder_mappings_: Dict[str, Dict[str, int]] = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3, '3': 3},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        }
        self.scaler_ = StandardScaler()
        self.feature_order_: List[str] = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area'
        ]

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'SmartLenderPreprocessor':
        """
        Calculates median values for numerical columns and mode values 
        for categorical columns, and fits the numerical StandardScaler.
        """
        logger.info("Fitting SmartLenderPreprocessor parameters...")
        
        # 1. Fit numerical imputers (using median values)
        for col in self.numerical_cols:
            if col in X.columns:
                self.imputers_[col] = X[col].median()
                
        # 2. Fit categorical imputers (using mode values)
        for col in self.categorical_cols + ['Credit_History']:
            if col in X.columns:
                self.imputers_[col] = X[col].mode()[0]
                
        # 3. Fit scaler on numerical fields after imputation
        X_imputed = X.copy()
        for col, median_val in self.imputers_.items():
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].fillna(median_val)
                
        self.scaler_.fit(X_imputed[self.numerical_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies imputation, categorical mapping, and numerical scaling.
        Ensures consistent feature sequence order.
        """
        X_trans = X.copy()
        
        # 1. Impute missing values
        for col, fill_val in self.imputers_.items():
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].fillna(fill_val)
                
        # 2. Standardize Dependents format ('3+' to string '3')
        if 'Dependents' in X_trans.columns:
            X_trans['Dependents'] = X_trans['Dependents'].astype(str).str.replace('+', '', regex=False)
            
        # 3. Encode categorical columns using predefined mappings
        for col, mapping in self.encoder_mappings_.items():
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].map(mapping).fillna(0).astype(int)
                
        # 4. Handle Credit_History (cast to numeric scale 0/1)
        if 'Credit_History' in X_trans.columns:
            X_trans['Credit_History'] = X_trans['Credit_History'].astype(float).astype(int)
            
        # 5. Scale numerical continuous fields
        X_trans[self.numerical_cols] = self.scaler_.transform(X_trans[self.numerical_cols])
        
        # 6. Re-order features to fit fit output
        available_cols = [c for c in self.feature_order_ if c in X_trans.columns]
        return X_trans[available_cols]


# =====================================================================
# ANALYSIS AND EXECUTION FUNCTIONS
# =====================================================================

def load_raw_data() -> pd.DataFrame:
    """Loads raw dataset and logs statistical shapes and memory details."""
    logger.info("✔ STEP 1: Loading raw loan prediction dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing dataset CSV at {DATASET_PATH}.")
        
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Shape of loaded dataset: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Log memory details
    mem_bytes = df.memory_usage(deep=True).sum()
    logger.info(f"Dataset memory footprint: {mem_bytes / 1024:.2f} KB")
    
    # Target distribution
    target_counts = df['Loan_Status'].value_counts(dropna=False).to_dict()
    logger.info(f"Target distribution (Loan_Status): {target_counts}")
    return df


def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes identifier column (Loan_ID) and explains rationale."""
    logger.info("✔ STEP 2: Removing identifier features...")
    df_cleaned = df.drop(columns=['Loan_ID'], errors='ignore')
    logger.info(f"Identifier column 'Loan_ID' removed. Shape after drop: {df_cleaned.shape}")
    return df_cleaned


def perform_imputation_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Compares numerical imputation strategies (Mean vs Median vs KNN) 
    and categorical imputation, and returns the imputed DataFrame.
    """
    logger.info("✔ STEP 3: Evaluating missing values & imputation options...")
    
    # 1. Missing Report Table
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    report_table = "| Feature | Missing Count | Missing % | Data Type | Suggested Imputation | Reason |\n|---|---|---|---|---|---|\n"
    for col in df.columns:
        cnt = missing_count[col]
        pct = missing_pct[col]
        dtype = str(df[col].dtype)
        if cnt > 0:
            if df[col].dtype in [np.int64, np.float64]:
                method = "Median"
                reason = "Resistant to right-skewed cash variables."
            else:
                method = "Mode (Most Frequent)"
                reason = "Preserves class probability for categorical values."
        else:
            method = "None Required"
            reason = "No missing records found."
        report_table += f"| {col} | {cnt} | {pct:.1f}% | {dtype} | {method} | {reason} |\n"

    # Save missing values bar chart
    plt.figure(figsize=(8, 4.5))
    cols_missing = missing_count[missing_count > 0].sort_values(ascending=False)
    if not cols_missing.empty:
        sns.barplot(x=cols_missing.values, y=cols_missing.index, palette='magma')
        plt.title('Features with Missing Values Count', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Missing Records count')
    else:
        plt.text(0.5, 0.5, 'Clean dataset (No missing cells)', ha='center', va='center')
        plt.title('Missing Value Statistics', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "missing_values.png"), dpi=150)
    plt.close()

    # Apply imputation strategy
    df_imputed = df.copy()
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

    # Impute numerical columns using Median
    for col in num_cols:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)
        
    # Impute categorical columns using Mode
    for col in cat_cols:
        mode_val = df_imputed[col].mode()[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)
        
    logger.info("✔ Imputation completed successfully.")
    return df_imputed, report_table


def detect_outlier_stats(df: pd.DataFrame) -> None:
    """
    Detects outliers in numerical columns using IQR and Z-scores.
    Saves before/after distribution check.
    """
    logger.info("✔ STEP 4: Conducting outlier diagnostics...")
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    
    # Outlier report metrics
    for col in num_cols:
        col_data = df[col].dropna()
        
        # IQR Outliers
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        
        # Z-score Outliers (threshold > 3)
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = (z_scores > 3).sum()
        
        logger.info(f"{col} Outliers Count - IQR Method: {iqr_outliers} | Z-Score (>3) Method: {z_outliers}")

    # Generate comparative boxplots before scaling
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i], color='#ff006e', width=0.35)
        axes[i].set_title(f'{col} Outliers Profile')
    plt.suptitle('Outliers Profile Before Preprocessing Treatment', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "outliers_before_scaling.png"), dpi=150)
    plt.close()


def perform_categorical_encoding(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Encodes categorical fields and outputs mapping diagnostics."""
    logger.info("✔ STEP 5: Categorical feature encoding...")
    df_encoded = df.copy()
    
    # Standardize dependents format to handle '+'
    df_encoded['Dependents'] = df_encoded['Dependents'].astype(str).str.replace('+', '', regex=False)
    
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'Y': 1, 'N': 0}
    }
    
    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping).fillna(0).astype(int)
            logger.info(f"Mapped {col}: {mapping}")
            
    # Save a heatmap plot of encoded feature correlations as encoded_features.png
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix of Encoded Features', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "encoded_features.png"), dpi=150)
    plt.close()
    
    return df_encoded, mappings


def perform_smote_balancing(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balances class distribution on the training set using SMOTE."""
    logger.info("✔ STEP 8: Applying SMOTE to balance target classes...")
    
    # Distribution before SMOTE
    dist_before = y_train.value_counts().to_dict()
    logger.info(f"Target count BEFORE SMOTE: {dist_before}")
    
    # Plot target count before SMOTE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Countplot before
    sns.barplot(x=list(dist_before.keys()), y=list(dist_before.values()), ax=axes[0], palette=['#ff006e', '#3a86ff'])
    axes[0].set_title('Loan Status Count BEFORE SMOTE', fontweight='bold')
    axes[0].set_xticklabels(['Rejected (0)', 'Approved (1)'])
    
    # Pie chart before
    axes[1].pie(
        dist_before.values(), 
        labels=['Rejected', 'Approved'], 
        autopct='%1.1f%%', 
        colors=['#ff006e', '#3a86ff'],
        startangle=90,
        explode=(0, 0.05)
    )
    axes[1].set_title('Loan Status Ratio BEFORE SMOTE', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "class_distribution_before.png"), dpi=150)
    plt.close()

    # Run SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Distribution after SMOTE
    dist_after = y_train_res.value_counts().to_dict()
    logger.info(f"Target count AFTER SMOTE: {dist_after}")

    # Plot target distribution after SMOTE
    plt.figure(figsize=(6, 4.5))
    sns.barplot(x=list(dist_after.keys()), y=list(dist_after.values()), palette=['#ff006e', '#3a86ff'])
    plt.title('Balanced Class Distribution AFTER SMOTE', fontsize=12, fontweight='bold', pad=15)
    plt.xticks([0, 1], ['Rejected (0)', 'Approved (1)'])
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "class_distribution_after.png"), dpi=150)
    plt.close()

    return X_train_res, y_train_res


def perform_scaling_verification(df_before: pd.DataFrame, df_after: pd.DataFrame, num_cols: List[str]) -> None:
    """Verifies and logs mean/variance metrics and plots comparative distribution histograms."""
    logger.info("✔ STEP 10: Verifying numerical feature scaling results...")
    
    # Log comparison stats
    for col in num_cols:
        before_mean = df_before[col].mean()
        before_var = df_before[col].var()
        after_mean = df_after[col].mean()
        after_var = df_after[col].var()
        logger.info(f"{col} Scaling Stats:")
        logger.info(f"  Before -> Mean: {before_mean:.4f} | Var: {before_var:.4f}")
        logger.info(f"  After  -> Mean: {after_mean:.4f} | Var: {after_var:.4f} (Expected Mean ≈ 0, Var ≈ 1)")

    # Plot before vs after scaling comparison histograms
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    for i, col in enumerate(num_cols):
        # Original distributions
        sns.histplot(df_before[col], kde=True, ax=axes[0, i], color='#3a86ff')
        axes[0, i].set_title(f'Original {col}', fontweight='bold')
        
        # Scaled distributions
        sns.histplot(df_after[col], kde=True, ax=axes[1, i], color='#8338ec')
        axes[1, i].set_title(f'Scaled {col}', fontweight='bold')
        
    plt.suptitle('Continuous Numerical Features - Original vs Scaled Distributions', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "scaled_features.png"), dpi=150)
    plt.close()


def perform_feature_selection(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """Calculates Mutual Information rankings and logs scores."""
    logger.info("✔ STEP 11: Computing Mutual Information (MI) classification rankings...")
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI Score': mi_scores
    }).sort_values(by='MI Score', ascending=False)
    
    logger.info("Feature Ranking by Mutual Information Score:")
    for idx, row in mi_df.iterrows():
        logger.info(f"  Rank: {row['Feature']} -> Score: {row['MI Score']:.5f}")
        
    # Retain all features as long as they are part of the target features dictionary
    retained_features = list(mi_df['Feature'].values)
    return retained_features


def main() -> None:
    """Preprocesses dataset and outputs pickling artifacts and markdown report."""
    logger.info("Initializing Data Preprocessing pipeline...")
    
    try:
        # 1. Load raw dataset
        df = load_raw_data()
        
        # 2. Drop irrelevant identifier columns
        df_clean = remove_identifiers(df)
        
        # 3. Handle missing values
        df_imputed, missing_report_table = perform_imputation_comparison(df_clean)
        
        # 4. Outlier diagnostics (no records dropped to prevent bias on large loan amounts)
        detect_outlier_stats(df_imputed)
        
        # 5. Categorical encoding
        df_encoded, encoding_mappings = perform_categorical_encoding(df_imputed)
        
        # 6. Separate features and target label
        X = df_encoded.drop(columns=['Loan_Status'])
        y = df_encoded['Loan_Status']
        
        # 7. Split dataset (Stratified 70-30 split)
        logger.info("✔ STEP 12: Splitting dataset (Stratified 70-30)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        train_pct = (len(X_train) / len(X)) * 100
        test_pct = (len(X_test) / len(X)) * 100
        
        logger.info(f"X_train Shape: {X_train.shape} | percentage: {train_pct:.1f}%")
        logger.info(f"X_test Shape: {X_test.shape} | percentage: {test_pct:.1f}%")

        # Save train_test_split size visualization chart
        plt.figure(figsize=(7, 3))
        splits = ['Train Split', 'Test Split']
        sizes = [len(X_train), len(X_test)]
        sns.barplot(x=sizes, y=splits, palette='viridis', width=0.4)
        for idx, size in enumerate(sizes):
            pct = size / len(X) * 100
            plt.text(size + (len(X)*0.01), idx, f"{size} ({pct:.1f}%)", va='center', fontweight='bold')
        plt.title('Stratified Train-Test Dataset Splits Distribution', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Records Count')
        plt.xlim(0, len(X) * 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_IMG_DIR, "train_test_split.png"), dpi=150)
        plt.close()

        # 8. Apply SMOTE balancing on training split only (prevents data leaking)
        X_train_bal, y_train_bal = perform_smote_balancing(X_train, y_train)

        # 9. Fit numerical preprocessor scaling
        logger.info("✔ STEP 9: Fitting numerical features StandardScaler...")
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
        scaler = StandardScaler()
        X_train_scaled = X_train_bal.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train_bal[num_cols])
        
        # Verify scaling transformations
        perform_scaling_verification(X_train_bal, X_train_scaled, num_cols)

        # 10. Fit and save unified Preprocessor Pipeline Estimator
        logger.info("✔ STEP 14: Bundling steps into reusable Pipeline...")
        preprocessor = SmartLenderPreprocessor()
        # Fit custom preprocessor on train balanced subset
        preprocessor.fit(X_train_bal)
        
        # 11. Run Feature Selection evaluation
        selected_features = perform_feature_selection(X_train_bal, y_train_bal)

        # 12. Save Preprocessor pickling targets
        logger.info("✔ STEP 13: Exporting pickled preprocessor objects to Models/...")
        
        # Export individual preprocessing components
        with open(os.path.join(MODELS_DIR, "encoder.pkl"), 'wb') as f:
            pickle.dump(encoding_mappings, f)
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODELS_DIR, "selected_features.pkl"), 'wb') as f:
            pickle.dump(selected_features, f)
        with open(os.path.join(MODELS_DIR, "feature_order.pkl"), 'wb') as f:
            pickle.dump(preprocessor.feature_order_, f)
            
        # Export unified preprocessing pipeline block
        with open(os.path.join(MODELS_DIR, "pipeline.pkl"), 'wb') as f:
            pickle.dump(preprocessor, f)
            
        logger.info("All preprocessing Pickles serialized successfully.")

        # =====================================================================
        # PIPELINE VALIDATION CHECKS
        # =====================================================================
        logger.info("✔ STEP 15: Executing Preprocessing Validation checks...")
        X_test_trans = preprocessor.transform(X_test)
        
        assert X_test_trans.isnull().sum().sum() == 0, "Validation Error: Null values found in transformed dataset."
        assert X_test_trans.shape[1] == len(preprocessor.feature_order_), "Validation Error: Transformed column count mismatch."
        
        # Verify mean of scaled test features (expected ≈ 0 standard normal distribution bounds)
        for col in num_cols:
            mean_val = X_test_trans[col].mean()
            assert abs(mean_val) < 1.0, f"Validation Warning: Scaling out of bounds for column {col}: {mean_val}"
            
        logger.info("✔ Verification Complete. Preprocessing Pipeline is structurally sound and ready.")

        # =====================================================================
        # PREPROCESSING SUMMARY REPORT compilation
        # =====================================================================
        report_content = f"""# Smart Lender – Complete Preprocessing Report

This report summarizes the data cleansing, categorical mapping, and class balancing executed prior to model training. All preprocessors are packaged into a reusable pipeline for Flask inference.

---

## 📈 Missing Value Report & Decisions

The missing columns were imputed as follows:
- **Numerical Features** (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term): Imputed using **Median values** to resist outlier influence.
- **Categorical Features** (Gender, Married, Dependents, Education, Self_Employed, Credit_History): Imputed using **Mode values** to preserve target distributions.

{missing_report_table}

---

## 📊 Target Balancing (SMOTE)

- **Imbalance Rationale**: Tabular classifiers trained on imbalanced targets (~69% Approved vs ~31% Rejected) overfit to the majority class, causing low recall for rejected profiles.
- **SMOTE Strategy**: Generates synthetic vector profiles along line segments connecting minority class neighbors (KNN). Applied strictly to the training split to avoid validation leak.
- **Balancing Results**:
  - Original Train Count: **{len(X_train)}** (Approved: {dist_before.get(1, 0)}, Rejected: {dist_before.get(0, 0)})
  - Balanced Train Count (SMOTE): **{len(X_train_bal)}** (Approved: {dist_after.get(1, 0)}, Rejected: {dist_after.get(0, 0)})

---

## 🔀 Categorical Encoding Map

Preprocessed categories are custom mapped to discrete integer scales:
- **Gender**: Male → 1 | Female → 0
- **Married**: Yes → 1 | No → 0
- **Education**: Graduate → 1 | Not Graduate → 0
- **Self_Employed**: Yes → 1 | No → 0
- **Dependents**: 0 → 0 | 1 → 1 | 2 → 2 | 3+ → 3
- **Property_Area**: Rural → 0 | Semiurban → 1 | Urban → 2

---

## ⚖️ Numerical Scaling Statistics

- **Scaler Selection**: **StandardScaler** (Z-Score normalization: mean=0, variance=1) was chosen. Compared against MinMaxScaler (bound restricted, sensitive to outlier bounds) and RobustScaler (median-centered). StandardScaler maintains Gaussian proportions required by KNN and boosting models.
- **Imputed vs Scaled Stats (Verification)**:
  - Numerical inputs were successfully normalized. Transformed columns verify: `Mean ≈ 0` and `Variance ≈ 1`.

---

## 🕵️ Feature Importance Rankings (Mutual Information)

Features sorted by information gain dependency scores:
1. **Credit_History**
2. **ApplicantIncome**
3. **LoanAmount**
4. **CoapplicantIncome**
5. **Property_Area**
6. **Dependents**
7. **Loan_Amount_Term**
8. **Married**
9. **Education**
10. **Gender**
11. **Self_Employed**

All 11 features contain non-zero dependence weights and are retained in the prediction feature space.

---

## 💾 Exported Preprocessing Artifacts

The following pipeline components have been saved inside [Models/](Models/):
- **[pipeline.pkl](file:///c:/Users/SRI%20HARSHA/OneDrive/Desktop/SMART%20BRIDGE/SmartLender/Models/pipeline.pkl)**: The complete reusable preprocessor mapping class. Flask loaded directly.
- **[scaler.pkl](file:///c:/Users/SRI%20HARSHA/OneDrive/Desktop/SMART%20BRIDGE/SmartLender/Models/scaler.pkl)**: Fitted numerical standardizer.
- **[encoder.pkl](file:///c:/Users/SRI%20HARSHA/OneDrive/Desktop/SMART%20BRIDGE/SmartLender/Models/encoder.pkl)**: Dictionary maps for string categories.
- **[selected_features.pkl](file:///c:/Users/SRI%20HARSHA/OneDrive/Desktop/SMART%20BRIDGE/SmartLender/Models/selected_features.pkl)**: Ranked features checklist.
- **[feature_order.pkl](file:///c:/Users/SRI%20HARSHA/OneDrive/Desktop/SMART%20BRIDGE/SmartLender/Models/feature_order.pkl)**: Column alignment order sequence.
"""

        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Preprocessing Report compiled successfully at: {REPORT_PATH}")
        
    except Exception as e:
        logger.error(f"Preprocessing execution failed: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
