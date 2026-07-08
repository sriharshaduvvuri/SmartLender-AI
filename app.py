"""
Smart Lender - Loan Approval Prediction System
Flask Backend Application following the Three-Layer Architecture:
1. Presentation Layer (Flask routes rendering templates)
2. Business Logic Layer (validation, performance tracking, logging, DB operations)
3. Machine Learning Inference Layer (model/pipeline loader, feature preprocessing, prediction)
"""

import os
import time
import logging
import sqlite3
import functools
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash

import sys
import preprocess
sys.modules['__main__'].SmartLenderPreprocessor = preprocess.SmartLenderPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    import secrets
    logger.warning("SECRET_KEY environment variable is not set. Generating a temporary random key for local development.")
    secret_key = secrets.token_hex(32)
app.secret_key = secret_key

# Config Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

# Cache for ML assets
_ML_CACHE: Dict[str, Any] = {}


# =====================================================================
# DATA LAYER & INFRASTRUCTURE
# =====================================================================

def init_db() -> None:
    """
    Initializes the SQLite database to store predictions and users.
    Creates tables and seeds default user if empty.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # 1. Create Predictions Audit Log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    gender TEXT,
                    married TEXT,
                    dependents TEXT,
                    education TEXT,
                    self_employed TEXT,
                    applicant_income REAL,
                    coapplicant_income REAL,
                    loan_amount REAL,
                    loan_amount_term REAL,
                    credit_history REAL,
                    property_area TEXT,
                    prediction_status TEXT,
                    confidence_score REAL,
                    model_used TEXT,
                    latency_ms REAL
                )
            """)
            
            # 2. Create Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                )
            """)
            conn.commit()
            
            # 3. Seed Admin only if custom admin environment variables are explicitly configured
            admin_username = os.environ.get("ADMIN_USERNAME")
            admin_password = os.environ.get("ADMIN_PASSWORD")
            
            if admin_username and admin_password:
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (admin_username,))
                if cursor.fetchone()[0] == 0:
                    pass_hash = generate_password_hash(admin_password)
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                        (admin_username, pass_hash, "admin")
                    )
                    conn.commit()
                    logger.info("Administrator account seeded successfully from environment variables.")
            else:
                logger.warning("Administrator account seeding skipped: ADMIN_USERNAME or ADMIN_PASSWORD environment variables are missing.")
            
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")


# =====================================================================
# AUTHENTICATION & SESSION MANAGEMENT
# =====================================================================

def register_user(username: str, password_clear: str, role: str = "officer") -> Tuple[bool, str]:
    """Registers a new user in the SQLite database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, "Username already exists."
            
            pass_hash = generate_password_hash(password_clear)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, pass_hash, role)
            )
            conn.commit()
            return True, "User registered successfully."
    except Exception as e:
        logger.error(f"Registration failed for {username}: {e}")
        return False, f"Database error: {str(e)}"


def authenticate_user(username: str, password_clear: str) -> Optional[Dict[str, Any]]:
    """Checks user credentials and returns user metadata if authenticated."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT username, password_hash, role FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if row and check_password_hash(row["password_hash"], password_clear):
                return {"username": row["username"], "role": row["role"]}
    except Exception as e:
        logger.error(f"Authentication failed for {username}: {e}")
    return None


def login_required(view):
    """Decorator to enforce authenticated session access."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if session.get("user") is None:
            flash("Please sign in to access this feature.", "warning")
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


@app.before_request
def load_logged_in_user():
    """Sets current user global variable on each web request."""
    g.user = session.get("user")



# =====================================================================
# MACHINE LEARNING INFERENCE LAYER
# =====================================================================

def load_ml_assets() -> Dict[str, Any]:
    """
    Loads all pickled models, encoders, and scalers into memory.
    Caches loaded artifacts to avoid slow disk I/O on every request.

    Returns:
        A dictionary containing all loaded ML components.
    """
    if _ML_CACHE:
        return _ML_CACHE

    assets = {}
    required_files = {
        "scaler": "scaler.pkl",
        "encoder": "encoder.pkl",
        "best_model": "best_model.pkl",
        "decision_tree": "decision_tree.pkl",
        "random_forest": "random_forest.pkl",
        "knn": "knn.pkl",
        "xgboost": "xgboost.pkl",
        "pipeline": "pipeline.pkl"
    }

    logger.info("Loading Machine Learning artifacts from disk...")

    for key, filename in required_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"ML Asset missing: {filepath}. Running train.py may be required.")
            continue
        try:
            assets[key] = joblib.load(filepath)
            logger.info(f"Successfully loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    _ML_CACHE.update(assets)
    return _ML_CACHE


def get_prediction_inference(form_data: Dict[str, Any], model_key: str) -> Tuple[str, float, str, float]:
    """
    Core ML prediction function. Preprocesses the input fields,
    scales numerical values, runs prediction inference, and computes confidence.

    Args:
        form_data: Validated dictionary of application parameters.
        model_key: Key of the selected classifier (e.g., 'xgboost', 'random_forest').

    Returns:
        A tuple of (prediction_label, probability_score, actual_model_name, execution_time_ms).
    """
    start_time = time.perf_counter()
    assets = load_ml_assets()

    # Determine which model to run
    model_name_map = {
        "best_model": "Best Model",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "knn": "K-Nearest Neighbors",
        "xgboost": "XGBoost"
    }
    
    # Fallback to best_model if requested key is missing
    model = assets.get(model_key) or assets.get("best_model")
    actual_model_name = model_name_map.get(model_key, "Best Model")

    if not model:
        raise ValueError("Selected machine learning model is not available. Please train models first.")

    scaler = assets.get("scaler")
    encoder = assets.get("encoder")

    if not scaler or not encoder:
        raise ValueError("Data preprocessors (scaler/encoder) are missing.")

    # 1. Map Categorical inputs using encoder dictionary
    try:
        gender_enc = encoder['Gender'][form_data['gender']]
        married_enc = encoder['Married'][form_data['married']]
        
        dep_val = form_data['dependents']
        if dep_val == '3+' and '3+' not in encoder['Dependents'] and '3' in encoder['Dependents']:
            dep_val = '3'
        elif dep_val == '3' and '3' not in encoder['Dependents'] and '3+' in encoder['Dependents']:
            dep_val = '3+'
            
        dependents_enc = encoder['Dependents'][dep_val]
        education_enc = encoder['Education'][form_data['education']]
        self_employed_enc = encoder['Self_Employed'][form_data['self_employed']]
        property_area_enc = encoder['Property_Area'][form_data['property_area']]
        credit_history_val = float(form_data['credit_history'])
    except KeyError as ke:
        logger.error(f"Encoding mapping failed. Key not found: {ke}")
        raise ValueError(f"Invalid categorical value supplied: {ke}")

    # 2. Extract numerical inputs
    app_income = float(form_data['applicant_income'])
    coapp_income = float(form_data['coapplicant_income'])
    loan_amount = float(form_data['loan_amount'])
    loan_term = float(form_data['loan_amount_term'])

    # 3. Create DataFrame matching original features ordering
    # Columns order: Gender, Married, Dependents, Education, Self_Employed, 
    #                ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
    #                Credit_History, Property_Area
    input_df = pd.DataFrame([{
        'Gender': gender_enc,
        'Married': married_enc,
        'Dependents': dependents_enc,
        'Education': education_enc,
        'Self_Employed': self_employed_enc,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history_val,
        'Property_Area': property_area_enc
    }])

    # 4. Scale numerical values using trained scaler
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # 5. Execute Prediction and get probabilities
    # Model returns 0 for rejected, 1 for approved
    prediction = int(model.predict(input_df)[0])
    
    # Handle confidence score/probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        # Probability of the predicted class
        probability = float(probs[prediction])
    else:
        # Fallback for models without probability output (rare)
        probability = None

    prediction_label = "Approved" if prediction == 1 else "Rejected"
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    logger.info(f"Prediction generated: {prediction_label} with prob {probability:.2f} using {actual_model_name} in {latency_ms:.2f}ms")
    return prediction_label, probability, actual_model_name, latency_ms


# =====================================================================
# BUSINESS LOGIC LAYER
# =====================================================================

def validate_loan_inputs(form: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates form data sent from predict form. Ensures standard limits
    and data types are met.

    Args:
        form: Dict containing raw inputs from request.form.

    Returns:
        A tuple: (validated_data_dict, error_message_str)
        One of the items will be None.
    """
    try:
        # Validate categorical values
        gender = form.get("gender")
        if gender not in ["Male", "Female"]:
            return None, "Gender must be 'Male' or 'Female'."

        married = form.get("married")
        if married not in ["Yes", "No"]:
            return None, "Married status must be 'Yes' or 'No'."

        dependents = form.get("dependents")
        if dependents not in ["0", "1", "2", "3+"]:
            return None, "Dependents must be '0', '1', '2', or '3+'."

        education = form.get("education")
        if education not in ["Graduate", "Not Graduate"]:
            return None, "Education must be 'Graduate' or 'Not Graduate'."

        self_employed = form.get("self_employed")
        if self_employed not in ["Yes", "No"]:
            return None, "Self Employed status must be 'Yes' or 'No'."

        property_area = form.get("property_area")
        if property_area not in ["Rural", "Semiurban", "Urban"]:
            return None, "Property Area must be 'Rural', 'Semiurban', or 'Urban'."

        credit_history = form.get("credit_history")
        if credit_history not in ["1.0", "0.0"]:
            return None, "Credit History must be 'Good (1.0)' or 'Bad (0.0)'."

        # Validate numerical bounds
        try:
            applicant_income = float(form.get("applicant_income", 0))
            if applicant_income < 0:
                return None, "Applicant Income cannot be negative."
        except ValueError:
            return None, "Applicant Income must be a number."

        try:
            coapplicant_income = float(form.get("coapplicant_income", 0))
            if coapplicant_income < 0:
                return None, "Co-Applicant Income cannot be negative."
        except ValueError:
            return None, "Co-Applicant Income must be a number."

        try:
            loan_amount = float(form.get("loan_amount", 0))
            if loan_amount <= 0:
                return None, "Loan Amount must be greater than zero."
        except ValueError:
            return None, "Loan Amount must be a number."

        try:
            loan_amount_term = float(form.get("loan_amount_term", 0))
            if loan_amount_term <= 0:
                return None, "Loan Amount Term must be greater than zero."
        except ValueError:
            return None, "Loan Amount Term must be a number."

        validated = {
            "gender": gender,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "property_area": property_area
        }
        return validated, None

    except Exception as e:
        logger.error(f"Error during input validation: {e}")
        return None, f"An unexpected validation error occurred: {str(e)}"


def save_prediction_record(data: Dict[str, Any], status: str, confidence: float, model: str, latency: float) -> None:
    """
    Saves predictions to SQLite logs for analytics and auditing.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO loan_predictions (
                    timestamp, gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                    credit_history, property_area, prediction_status, confidence_score,
                    model_used, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, data['gender'], data['married'], data['dependents'], data['education'], data['self_employed'],
                data['applicant_income'], data['coapplicant_income'], data['loan_amount'], data['loan_amount_term'],
                float(data['credit_history']), data['property_area'], status, confidence, model, latency
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to record prediction in DB: {e}")


# =====================================================================
# PRESENTATION LAYER (ROUTING)
# =====================================================================

@app.route("/")
@login_required
def home():
    """
    Serves the Dashboard/Home landing page.
    Includes about text, dataset fields description, and live prediction logs summary.
    """
    # Fetch summary stats from database to render on home dashboard
    stats = {"total": 0, "approved": 0, "rejected": 0}
    recent_predictions = []
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Read counts
            cursor.execute("SELECT COUNT(*) FROM loan_predictions")
            stats["total"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM loan_predictions WHERE prediction_status = 'Approved'")
            stats["approved"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM loan_predictions WHERE prediction_status = 'Rejected'")
            stats["rejected"] = cursor.fetchone()[0]
            
            # Read recent 5 predictions
            cursor.execute("SELECT timestamp, prediction_status, confidence_score, model_used, latency_ms FROM loan_predictions ORDER BY id DESC LIMIT 5")
            recent_predictions = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error querying home statistics: {e}")

    # Check if visualizations exist in static images
    has_eda_images = os.path.exists(os.path.join(BASE_DIR, "static", "images", "loan_status_dist.png"))

    return render_template(
        "home.html", 
        stats=stats, 
        recent_predictions=recent_predictions,
        has_eda_images=has_eda_images
    )


def generate_rule_based_explanations(form_data: Dict[str, Any], target_class: int) -> List[Dict[str, Any]]:
    """Safe fallback explanation strategy using standard rule-based heuristics."""
    contributions = []
    
    # 1. Credit History is the strongest factor
    if str(form_data.get("credit_history")) in ["1.0", "1"]:
        contributions.append({
            "feature_name": "Credit History Score",
            "impact": 50.0,
            "direction": "positive" if target_class == 1 else "negative"
        })
    else:
        contributions.append({
            "feature_name": "Credit History Score",
            "impact": 70.0,
            "direction": "negative" if target_class == 0 else "positive"
        })
        
    # 2. Debt-to-income heuristic
    try:
        app_income = float(form_data.get("applicant_income", 3800))
        loan_amt = float(form_data.get("loan_amount", 128))
        if app_income > 0:
            ratio = (loan_amt * 1000) / app_income
            if ratio > 40: # High loan compared to monthly income
                contributions.append({
                    "feature_name": "Requested Loan Amount",
                    "impact": 35.0,
                    "direction": "negative" if target_class == 0 else "positive"
                })
            else:
                contributions.append({
                    "feature_name": "Requested Loan Amount",
                    "impact": 20.0,
                    "direction": "positive" if target_class == 1 else "negative"
                })
    except:
        pass
        
    # 3. Employment heuristic
    if form_data.get("self_employed") == "Yes":
        contributions.append({
            "feature_name": "Self Employment Status",
            "impact": 15.0,
            "direction": "negative" if target_class == 0 else "positive"
        })
    else:
        contributions.append({
            "feature_name": "Self Employment Status",
            "impact": 10.0,
            "direction": "positive" if target_class == 1 else "negative"
        })
        
    return contributions[:3]


def explain_prediction(form_data: Dict[str, Any], model_key: str) -> List[Dict[str, Any]]:
    """
    Computes local feature impact (sensitivity) by comparing the prediction probability 
    against a perturbed scenario where the target feature is set to its median/mode baseline.
    """
    try:
        assets = load_ml_assets()
        model = assets.get(model_key) or assets.get("best_model")
        
        # Check prediction status even if predict_proba is not available
        try:
            pred_label, prob, _, _ = get_prediction_inference(form_data, model_key)
        except Exception as e:
            logger.error(f"Failed to run reference inference during explanation: {e}")
            return []
            
        target_class = 1 if pred_label == "Approved" else 0
        
        if not model or not hasattr(model, "predict_proba"):
            # Fallback strategy
            return generate_rule_based_explanations(form_data, target_class)
            
        original_prob = prob if target_class == 1 else (1.0 - prob)
        
        # Medians / standard values from preprocessor and dataset
        baselines = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "3812.5",
            "coapplicant_income": "0.0",
            "loan_amount": "128.0",
            "loan_amount_term": "360.0",
            "credit_history": "1.0",
            "property_area": "Semiurban"
        }
        
        contributions = []
        feature_labels = {
            "gender": "Gender",
            "married": "Marital Status",
            "dependents": "Dependents Count",
            "education": "Education Level",
            "self_employed": "Self Employment Status",
            "applicant_income": "Applicant Income",
            "coapplicant_income": "Co-Applicant Income",
            "loan_amount": "Requested Loan Amount",
            "loan_amount_term": "Repayment Term",
            "credit_history": "Credit History Score",
            "property_area": "Property Location"
        }
        
        for key in form_data.keys():
            if key not in baselines or key == "model_used":
                continue
                
            # If the current value is already equal to the baseline, its impact is 0
            if str(form_data[key]) == str(baselines[key]):
                continue
                
            # Perturb single key to baseline
            perturbed_data = form_data.copy()
            perturbed_data[key] = str(baselines[key])
            
            try:
                p_label, p_prob, _, _ = get_prediction_inference(perturbed_data, model_key)
                p_target_prob = p_prob if target_class == 1 else (1.0 - p_prob)
                
                # Impact: how much original value changed the probability of this decision vs baseline
                impact = original_prob - p_target_prob
                
                import math
                if not math.isfinite(impact):
                    impact = 0.0
                
                # Cap progress bar impacts at 100% and map to magnitude
                impact_pct = max(min(impact * 100, 100.0), -100.0)
                
                if abs(impact_pct) > 0.1:
                    contributions.append({
                        "feature_key": key,
                        "feature_name": feature_labels.get(key, key),
                        "impact": round(abs(impact_pct), 2),
                        "direction": "positive" if impact > 0 else "negative"
                    })
            except Exception as ex:
                logger.error(f"Perturbation failed for feature {key}: {ex}")
                
        # Sort contributions by absolute impact descending
        contributions = sorted(contributions, key=lambda x: x["impact"], reverse=True)
        
        # Padded fallback to guarantee exactly 3 drivers
        if len(contributions) < 3:
            fallback = generate_rule_based_explanations(form_data, target_class)
            for item in fallback:
                if len(contributions) >= 3:
                    break
                if not any(x["feature_name"] == item["feature_name"] for x in contributions):
                    contributions.append({
                        "feature_key": item["feature_name"].lower().replace(" ", "_"),
                        "feature_name": item["feature_name"],
                        "impact": round(abs(item["impact"]), 2),
                        "direction": item["direction"]
                    })
                    
        return contributions[:3]
    except Exception as e:
        logger.error(f"Error during feature importance generation: {e}")
        return []


@app.route("/login", methods=["GET", "POST"])
def login():
    """Renders user login template and handles validation logic."""
    if g.user:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")
            
        user = authenticate_user(username, password)
        if user:
            session.clear()
            session["user"] = user
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
            
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Renders officer registration page."""
    if g.user:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html")
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
            
        success, message = register_user(username, password)
        if success:
            flash(message + " Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash(message, "danger")
            
    return render_template("register.html")


@app.route("/logout")
def logout():
    """Clears user session and logs out."""
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


@app.route("/predict/bulk", methods=["GET", "POST"])
@login_required
def predict_bulk():
    """Handles bulk CSV uploads, runs batch predictions with row-level validation, and displays results."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)
            
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for upload.", "danger")
            return redirect(request.url)
            
        if not file.filename.endswith(".csv"):
            flash("Only CSV files are supported.", "danger")
            return redirect(request.url)
            
        model_key = request.form.get("model_used", "best_model")
        
        try:
            # Check payload size (limit to 5MB)
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset pointer
            if file_size > 5 * 1024 * 1024:
                flash("Payload Too Large. CSV file size must be less than 5MB.", "danger")
                return redirect(request.url)
                
            # Read CSV file using pandas
            df = pd.read_csv(file)
            
            if df.empty:
                flash("The uploaded CSV file is empty.", "danger")
                return redirect(request.url)
                
            # Check required columns
            required_cols = [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                "Credit_History", "Property_Area"
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                flash(f"CSV is missing required column(s): {', '.join(missing_cols)}", "danger")
                return redirect(request.url)
                
            assets = load_ml_assets()
            preprocessor = assets.get("pipeline")
            if not preprocessor:
                preprocessor_path = os.path.join(MODELS_DIR, "pipeline.pkl")
                if os.path.exists(preprocessor_path):
                    preprocessor = joblib.load(preprocessor_path)
                else:
                    raise ValueError("Preprocessing pipeline is missing. Please train models first.")
            
            model = assets.get(model_key) or assets.get("best_model")
            if not model:
                raise ValueError("Selected model is not available.")
                
            # Prepare result lists
            decisions = []
            confidences = []
            models_used_col = []
            statuses = []
            errors = []
            
            processed_count = 0
            failed_count = 0
            approved_count = 0
            rejected_count = 0
            valid_confidences = []
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Row-by-row validation & inference
            default_imputers = {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 3812.5,
                "CoapplicantIncome": 0.0,
                "LoanAmount": 128.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Semiurban"
            }

            for idx, row in df.iterrows():
                row_errors = []
                
                # Fetch field value using preprocessor modes/medians as fallbacks for missing data
                def get_field_val(field_name):
                    val = row.get(field_name)
                    if pd.isna(val):
                        if preprocessor and hasattr(preprocessor, "imputers_"):
                            return preprocessor.imputers_.get(field_name, default_imputers.get(field_name))
                        return default_imputers.get(field_name)
                    return val

                # Helper for categorical validation
                def validate_cat(val, choices, field_name):
                    v = str(val).strip() if not pd.isna(val) else ""
                    if v not in choices:
                        row_errors.append(f"{field_name} must be one of {choices}")
                    return v
                
                gender_v = validate_cat(get_field_val("Gender"), ["Male", "Female"], "Gender")
                married_v = validate_cat(get_field_val("Married"), ["Yes", "No"], "Married")
                dependents_raw = str(get_field_val("Dependents")).strip().replace('+', '')
                dependents_v = validate_cat(dependents_raw, ["0", "1", "2", "3+", "3"], "Dependents")
                education_v = validate_cat(get_field_val("Education"), ["Graduate", "Not Graduate"], "Education")
                self_emp_v = validate_cat(get_field_val("Self_Employed"), ["Yes", "No"], "Self_Employed")
                prop_v = validate_cat(get_field_val("Property_Area"), ["Rural", "Semiurban", "Urban"], "Property_Area")
                
                credit_hist_raw = str(get_field_val("Credit_History")).strip()
                if credit_hist_raw in ["1.0", "1", "1.0", "1"]:
                    credit_hist_v = "1.0"
                elif credit_hist_raw in ["0.0", "0", "0.0", "0"]:
                    credit_hist_v = "0.0"
                else:
                    row_errors.append("Credit_History must be 0 or 1")
                    credit_hist_v = "1.0"
                
                # Helper for numerical validation
                def validate_num(val, field_name, allow_zero=True):
                    try:
                        v = float(val)
                        import math
                        if not math.isfinite(v):
                            row_errors.append(f"{field_name} must be a finite number")
                            return 0.0
                        if v < 0:
                            row_errors.append(f"{field_name} cannot be negative")
                        if not allow_zero and v == 0:
                            row_errors.append(f"{field_name} must be greater than zero")
                        return v
                    except:
                        row_errors.append(f"{field_name} must be a valid number")
                        return 0.0
                
                app_income_v = validate_num(get_field_val("ApplicantIncome"), "ApplicantIncome")
                coapp_income_v = validate_num(get_field_val("CoapplicantIncome"), "CoapplicantIncome")
                loan_amt_v = validate_num(get_field_val("LoanAmount"), "LoanAmount", allow_zero=False)
                loan_term_v = validate_num(get_field_val("Loan_Amount_Term"), "Loan_Amount_Term", allow_zero=False)
                
                if row_errors:
                    decisions.append("N/A")
                    confidences.append("0.0%")
                    models_used_col.append("None")
                    statuses.append("Failed")
                    errors.append("; ".join(row_errors))
                    failed_count += 1
                else:
                    try:
                        # Assemble single-row validated inputs
                        validated_row = {
                            "gender": gender_v,
                            "married": married_v,
                            "dependents": "3+" if dependents_v in ["3+", "3"] else dependents_v,
                            "education": education_v,
                            "self_employed": self_emp_v,
                            "applicant_income": app_income_v,
                            "coapplicant_income": coapp_income_v,
                            "loan_amount": loan_amt_v,
                            "loan_amount_term": loan_term_v,
                            "credit_history": credit_hist_v,
                            "property_area": prop_v
                        }
                        
                        prediction, confidence, actual_model_name, latency = get_prediction_inference(validated_row, model_key)
                        
                        # Save single prediction record to SQLite
                        save_prediction_record(validated_row, prediction, confidence, f"{actual_model_name} (Bulk)", latency)
                        
                        if prediction == "Approved":
                            approved_count += 1
                        else:
                            rejected_count += 1
                            
                        decisions.append(prediction)
                        confidences.append(f"{confidence * 100:.1f}%" if confidence is not None else "N/A")
                        if confidence is not None:
                            valid_confidences.append(confidence)
                        models_used_col.append(actual_model_name)
                        statuses.append("Success")
                        errors.append("")
                        processed_count += 1
                    except Exception as ex:
                        decisions.append("N/A")
                        confidences.append("0.0%")
                        models_used_col.append("None")
                        statuses.append("Failed")
                        errors.append(f"Inference error: {str(ex)}")
                        failed_count += 1
            
            # Build output DataFrame
            report_df = df.copy()
            report_df["Prediction"] = decisions
            report_df["Confidence"] = confidences
            report_df["Model Used"] = models_used_col
            report_df["Status"] = statuses
            report_df["Error"] = errors
            
            # Mitigate CSV Formula Injections (prefix critical values with a single quote)
            def escape_csv_injection(val):
                s = str(val)
                if s.startswith(('=', '+', '-', '@')):
                    return "'" + s
                return val
                
            for col in report_df.columns:
                report_df[col] = report_df[col].apply(escape_csv_injection)
            
            # Export results to CSV inside static directory for download
            reports_dir = os.path.join(BASE_DIR, "static", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_filename = f"bulk_report_{int(time.time())}.csv"
            report_filepath = os.path.join(reports_dir, report_filename)
            report_df.to_csv(report_filepath, index=False)
            
            avg_conf_val = np.mean(valid_confidences) if valid_confidences else 0.0
            
            # Prepare result stats
            results = {
                "total": len(df),
                "processed": processed_count,
                "failed": failed_count,
                "approved": approved_count,
                "rejected": rejected_count,
                "approved_pct": f"{(approved_count / max(processed_count, 1)) * 100:.1f}%" if processed_count > 0 else "0.0%",
                "rejected_pct": f"{(rejected_count / max(processed_count, 1)) * 100:.1f}%" if processed_count > 0 else "0.0%",
                "avg_confidence": f"{avg_conf_val * 100:.1f}%",
                "report_file": report_filename,
                "preview_rows": report_df.head(10).to_dict(orient="records"),
                "headers": report_df.columns.tolist()
            }
            
            return render_template("bulk_output.html", results=results)
            
        except Exception as e:
            logger.error(f"Bulk ingestion failed: {e}", exc_info=True)
            flash(f"Failed to process CSV: {str(e)}", "danger")
            return render_template("bulk_upload.html")
            
    return render_template("bulk_upload.html")


@app.route("/predict/bulk/download/<filename>")
@login_required
def download_bulk_report(filename):
    """Serves the generated bulk prediction CSV file."""
    reports_dir = os.path.join(BASE_DIR, "static", "reports")
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        from flask import send_from_directory
        return send_from_directory(reports_dir, filename, as_attachment=True)
    else:
        flash("Requested report file not found.", "danger")
        return redirect(url_for("predict_bulk"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """
    Renders prediction input form and handles submission logic.
    """
    if request.method == "POST":
        # Get raw form parameters
        form_data = request.form.to_dict()
        model_key = form_data.get("model_used", "best_model")

        # 1. Validate Form Inputs
        validated_data, err_msg = validate_loan_inputs(form_data)
        if err_msg:
            flash(err_msg, "danger")
            # Preserve form values for user correction
            return render_template("predict.html", form_values=form_data, model_used=model_key)

        # 2. Check if models are trained and ready
        assets = load_ml_assets()
        if not assets:
            flash("Machine Learning models are not trained yet. Please run training script first.", "warning")
            return redirect(url_for("predict"))

        # 3. Process ML Inference
        try:
            prediction, confidence, model_used, latency = get_prediction_inference(validated_data, model_key)
            
            # Generate feature explanations
            explanations = explain_prediction(validated_data, model_key)
            
            # 4. Store prediction log
            save_prediction_record(validated_data, prediction, confidence, model_used, latency)

            confidence_str = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
            # Redirect to output page with prediction parameters
            return render_template(
                "output.html",
                prediction=prediction,
                confidence=confidence_str,
                model_used=model_used,
                latency=f"{latency:.2f} ms",
                inputs=validated_data,
                explanations=explanations
            )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            flash(f"Inference failure: {str(e)}", "danger")
            return render_template("predict.html", form_values=form_data, model_used=model_key)

    # GET Request: render blank form
    return render_template("predict.html", form_values={}, model_used="best_model")


@app.errorhandler(400)
def bad_request(error):
    return render_template("errors.html", title="400 Bad Request", message="The request could not be understood by the server due to malformed syntax."), 400

@app.errorhandler(404)
def not_found(error):
    return render_template("errors.html", title="404 Not Found", message="The requested URL was not found on the server."), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("errors.html", title="413 Payload Too Large", message="The uploaded file size exceeds the server limit of 5MB."), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal Server Error: {error}", exc_info=True)
    return render_template("errors.html", title="500 Internal Server Error", message="An unexpected error occurred on the server. Our technical team has been notified."), 500


# Initialize DB on application startup
init_db()

if __name__ == "__main__":
    # Ensure port is bindable on development environments
    app.run(host="0.0.0.0", port=5000, debug=True)
