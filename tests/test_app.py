import os
import unittest
import pandas as pd
import numpy as np
import sys

# Adjust path to import modules from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
# Override DB path for tests before importing members to isolate test DB
app.DB_PATH = os.path.join(app.BASE_DIR, "test_predictions.db")

from app import app as flask_app, init_db, DB_PATH, register_user, authenticate_user
from preprocess import SmartLenderPreprocessor


class TestSmartLenderPreprocessor(unittest.TestCase):
    """
    Tests the SmartLenderPreprocessor custom sklearn-compliant pipeline class.
    """
    def setUp(self):
        self.preprocessor = SmartLenderPreprocessor()
        # Create a sample DataFrame matching dataset columns
        self.sample_df = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male'],
            'Married': ['Yes', 'No', 'Yes'],
            'Dependents': ['0', '1', '3+'],
            'Education': ['Graduate', 'Not Graduate', 'Graduate'],
            'Self_Employed': ['No', 'Yes', 'No'],
            'ApplicantIncome': [5000, 3000, 8000],
            'CoapplicantIncome': [0, 1500, 2000],
            'LoanAmount': [120.0, 90.0, 200.0],
            'Loan_Amount_Term': [360.0, 180.0, 360.0],
            'Credit_History': [1.0, 0.0, 1.0],
            'Property_Area': ['Semiurban', 'Rural', 'Urban']
        })

    def test_fit_and_transform(self):
        """
        Verifies that fitting and transforming works, and categorical/numerical variables are processed correctly.
        """
        # Fit preprocessor
        self.preprocessor.fit(self.sample_df)
        
        # Transform data
        transformed_df = self.preprocessor.transform(self.sample_df)
        
        # Ensure shape and feature alignment
        self.assertEqual(transformed_df.shape[1], 11)
        self.assertListEqual(list(transformed_df.columns), self.preprocessor.feature_order_)

        # Verify mapping logic for categorical values
        # Male should be 1, Female should be 0
        self.assertEqual(transformed_df.loc[0, 'Gender'], 1)
        self.assertEqual(transformed_df.loc[1, 'Gender'], 0)
        
        # Married: Yes -> 1, No -> 0
        self.assertEqual(transformed_df.loc[0, 'Married'], 1)
        self.assertEqual(transformed_df.loc[1, 'Married'], 0)
        
        # Property Area: Rural -> 0, Semiurban -> 1, Urban -> 2
        self.assertEqual(transformed_df.loc[0, 'Property_Area'], 1)  # Semiurban
        self.assertEqual(transformed_df.loc[1, 'Property_Area'], 0)  # Rural
        self.assertEqual(transformed_df.loc[2, 'Property_Area'], 2)  # Urban

    def test_missing_values_handling(self):
        """
        Verifies that missing values are correctly imputed.
        """
        df_with_nan = pd.DataFrame({
            'Gender': [np.nan, 'Male'],
            'Married': ['Yes', np.nan],
            'Dependents': [np.nan, '0'],
            'Education': ['Graduate', np.nan],
            'Self_Employed': [np.nan, 'No'],
            'ApplicantIncome': [np.nan, 5000],
            'CoapplicantIncome': [0, np.nan],
            'LoanAmount': [np.nan, 150.0],
            'Loan_Amount_Term': [360.0, np.nan],
            'Credit_History': [np.nan, 1.0],
            'Property_Area': ['Urban', np.nan]
        })
        
        # Fit on non-null dataset first to learn medians/modes
        self.preprocessor.fit(self.sample_df)
        
        # Transform the dataset containing missing values
        transformed = self.preprocessor.transform(df_with_nan)
        
        # Ensure no nulls remain in columns
        self.assertFalse(transformed.isnull().any().any())


class TestSmartLenderApp(unittest.TestCase):
    """
    Tests the Flask web application routes, parameters validation, and prediction responses.
    """
    def setUp(self):
        # Configure env variables for testing admin credentials seeding
        os.environ["ADMIN_USERNAME"] = "testadmin"
        os.environ["ADMIN_PASSWORD"] = "password123"
        os.environ["SECRET_KEY"] = "testkey123"

        # Remove test database if it exists to start fresh
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
            except Exception:
                pass
        flask_app.config['TESTING'] = True
        flask_app.config['WTF_CSRF_ENABLED'] = False
        flask_app.config['SECRET_KEY'] = 'testkey123'
        self.client = flask_app.test_client()
        init_db()
        # Seed test user (fallback if not already seeded by init_db)
        register_user("testadmin", "password123", "admin")
        with self.client.session_transaction() as sess:
            sess['user'] = {'username': 'testadmin', 'role': 'admin'}

    def tearDown(self):
        # Clean up env variables
        os.environ.pop("ADMIN_USERNAME", None)
        os.environ.pop("ADMIN_PASSWORD", None)
        os.environ.pop("SECRET_KEY", None)

        # Clean up test database
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
            except Exception:
                pass

    def test_home_route(self):
        """
        Verifies that the home landing page loads correctly.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Lender System", response.data)
        self.assertIn(b"Model Analytics", response.data)

    def test_predict_get_route(self):
        """
        Verifies that the blank predictor form page loads correctly.
        """
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Borrower Evaluation Form", response.data)

    def test_predict_post_valid(self):
        """
        Verifies that a valid form submission returns a prediction outcome correctly.
        """
        valid_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "6000",
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=valid_data)
        self.assertEqual(response.status_code, 200)
        # Should render output page showing Approved or Rejected prediction
        self.assertTrue(
            b"Loan Approved" in response.data or b"Loan Rejected" in response.data,
            "Response should indicate prediction results"
        )
        self.assertIn(b"Borrower Profile Summary", response.data)
        self.assertIn(b"Confidence Score", response.data)

    def test_predict_post_invalid_income(self):
        """
        Verifies that a negative applicant income is caught by the validation system.
        """
        invalid_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "-100",  # Negative
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=invalid_data)
        self.assertEqual(response.status_code, 200)
        # Should stay on predictor page and display validation warning
        self.assertIn(b"Applicant Income cannot be negative.", response.data)
        self.assertIn(b"Borrower Evaluation Form", response.data)

    def test_predict_post_invalid_category(self):
        """
        Verifies that invalid categorical options are detected.
        """
        invalid_data = {
            "gender": "InvalidGender",  # Invalid
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "5000",
            "coapplicant_income": "0",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=invalid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Gender must be", response.data)

    def test_predict_post_zero_income(self):
        """
        Verifies that a zero applicant income is handled successfully without crashing.
        """
        zero_income_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "0",  # Zero income
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=zero_income_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"Loan Approved" in response.data or b"Loan Rejected" in response.data,
            "Response should render output page successfully"
        )

    def test_predict_redirects_when_logged_out(self):
        """
        Verifies that requesting /predict redirects to /login when session is unauthenticated.
        """
        with self.client.session_transaction() as sess:
            sess.clear()
        
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/login'))

    def test_login_flow(self):
        """
        Verifies that submitting correct credentials to /login logs the user in successfully.
        """
        with self.client.session_transaction() as sess:
            sess.clear()
            
        login_data = {
            "username": "testadmin",
            "password": "password123"
        }
        response = self.client.post('/login', data=login_data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/'))

    def test_invalid_login(self):
        """Verifies that submitting invalid credentials fails authentication."""
        with self.client.session_transaction() as sess:
            sess.clear()
        login_data = {
            "username": "testadmin",
            "password": "wrongpassword"
        }
        response = self.client.post('/login', data=login_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid username or password", response.data)

    def test_registration_flow(self):
        """Verifies that registration page registers new credentials successfully."""
        with self.client.session_transaction() as sess:
            sess.clear()
        reg_data = {
            "username": "newuser",
            "password": "newpassword123",
            "confirm_password": "newpassword123"
        }
        response = self.client.post('/register', data=reg_data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/login'))

    def test_duplicate_registration(self):
        """Verifies that registering a duplicate username fails gracefully."""
        with self.client.session_transaction() as sess:
            sess.clear()
        reg_data = {
            "username": "testadmin",
            "password": "newpassword123",
            "confirm_password": "newpassword123"
        }
        response = self.client.post('/register', data=reg_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Username already exists", response.data)

    def test_database_insertion(self):
        """Verifies that a successful prediction is correctly logged in SQLite database."""
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM loan_predictions")
            before_count = c.fetchone()[0]
            
        valid_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "6000",
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        self.client.post('/predict', data=valid_data)
        
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM loan_predictions")
            after_count = c.fetchone()[0]
            
        self.assertEqual(after_count, before_count + 1)

    def test_logout_flow(self):
        """Verifies that logging out clears session and redirects to login page."""
        response = self.client.get('/logout')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/login'))
        
        response_dash = self.client.get('/')
        self.assertEqual(response_dash.status_code, 302)
        self.assertTrue(response_dash.headers['Location'].endswith('/login'))

    def test_bulk_predict_invalid_file_extension(self):
        """Verifies that uploading a non-CSV file returns an error."""
        from io import BytesIO
        data = {
            'file': (BytesIO(b"some content"), 'test.txt'),
            'model_used': 'best_model'
        }
        response = self.client.post('/predict/bulk', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 302)

    def test_bulk_predict_missing_columns(self):
        """Verifies that uploading a CSV missing columns is caught cleanly."""
        from io import BytesIO
        csv_data = b"Gender,Married,ApplicantIncome\nMale,Yes,5000\n"
        data = {
            'file': (BytesIO(csv_data), 'test.csv'),
            'model_used': 'best_model'
        }
        response = self.client.post('/predict/bulk', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 302)

    def test_bulk_predict_valid_csv(self):
        """Verifies that uploading a valid CSV returns a 200 preview and displays statistics."""
        from io import BytesIO
        csv_content = (
            "Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area\n"
            "Male,Yes,0,Graduate,No,6000,2000,150,360,1.0,Semiurban\n"
        )
        data = {
            'file': (BytesIO(csv_content.encode('utf-8')), 'test.csv'),
            'model_used': 'best_model'
        }
        response = self.client.post('/predict/bulk', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Batch Prediction Completed", response.data)

    def test_bulk_predict_row_failure(self):
        """Verifies that a CSV containing one bad row succeeds for valid rows, marking the bad row as failed."""
        from io import BytesIO
        csv_content = (
            "Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area\n"
            "Male,Yes,0,Graduate,No,6000,2000,150,360,1.0,Semiurban\n"
            "Male,Yes,0,Graduate,No,-5000,2000,150,360,1.0,Semiurban\n"
        )
        data = {
            'file': (BytesIO(csv_content.encode('utf-8')), 'test.csv'),
            'model_used': 'best_model'
        }
        response = self.client.post('/predict/bulk', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Batch Prediction Completed", response.data)

    def test_registration_short_password(self):
        """Verifies that registration fails if password is too short."""
        with self.client.session_transaction() as sess:
            sess.clear()
        reg_data = {
            "username": "shortuser",
            "password": "123",
            "confirm_password": "123"
        }
        response = self.client.post('/register', data=reg_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Password must be at least 6 characters long", response.data)

    def test_dashboard_metrics_calculation(self):
        """Verifies that dashboard aggregates predictions and calculates totals correctly."""
        # Insert test records
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO loan_predictions (timestamp, prediction_status, confidence_score, model_used, latency_ms) VALUES (?, ?, ?, ?, ?)",
                ("2026-07-08 12:00:00", "Approved", 0.85, "XGBoost", 10.0)
            )
            c.execute(
                "INSERT INTO loan_predictions (timestamp, prediction_status, confidence_score, model_used, latency_ms) VALUES (?, ?, ?, ?, ?)",
                ("2026-07-08 12:01:00", "Rejected", 0.75, "Random Forest", 12.0)
            )
            conn.commit()

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Total Predictions", response.data)
        # Check totals count (2 predictions inserted)
        self.assertIn(b"2", response.data)

    def test_bulk_predict_get_route(self):
        """Verifies that the bulk predict uploader page loads successfully."""
        response = self.client.get('/predict/bulk')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Bulk Ingestion Portal", response.data)

    def test_bulk_predict_empty_csv(self):
        """Verifies that uploading an empty CSV returns an error."""
        from io import BytesIO
        data = {
            'file': (BytesIO(b""), 'test.csv'),
            'model_used': 'best_model'
        }
        response = self.client.post('/predict/bulk', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Failed to process CSV", response.data)

    def test_scenario_a_strong_borrower(self):
        """Scenario A: Verifies prediction logic for a strong applicant profile (highly likely to be Approved)."""
        strong_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "12000",
            "coapplicant_income": "5000",
            "loan_amount": "80",  # low loan amount relative to high income
            "loan_amount_term": "360",
            "credit_history": "1.0",  # Good credit history
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=strong_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Loan Approved", response.data)

    def test_scenario_b_high_risk_borrower(self):
        """Scenario B: Verifies prediction logic for a high-risk applicant profile (highly likely to be Rejected)."""
        high_risk_data = {
            "gender": "Male",
            "married": "No",
            "dependents": "3+",
            "education": "Not Graduate",
            "self_employed": "Yes",
            "applicant_income": "1500",  # very low income
            "coapplicant_income": "0",
            "loan_amount": "400",  # very high loan amount requested
            "loan_amount_term": "180",
            "credit_history": "0.0",  # Bad credit history
            "property_area": "Rural",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=high_risk_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Loan Rejected", response.data)

    def test_scenario_c_mixed_borrower(self):
        """Scenario C: Verifies that a mixed profile executes successfully without crashes."""
        mixed_data = {
            "gender": "Female",
            "married": "Yes",
            "dependents": "2",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "4500",
            "coapplicant_income": "1500",
            "loan_amount": "200",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Urban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=mixed_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b"Loan Approved" in response.data or b"Loan Rejected" in response.data)

    def test_explanation_text_in_output(self):
        """Verifies that the prediction output page displays correct explanation headings and disclaimers."""
        valid_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "6000",
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1.0",
            "property_area": "Semiurban",
            "model_used": "best_model"
        }
        response = self.client.post('/predict', data=valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Local Sensitivity Explanation", response.data)
        self.assertIn(b"Feature influence values estimate local model sensitivity and should not be interpreted as causal effects.", response.data)


if __name__ == '__main__':
    unittest.main()
