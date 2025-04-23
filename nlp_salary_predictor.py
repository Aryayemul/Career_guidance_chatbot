import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Download NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
    
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Function to discretize salary values into salary bands
def discretize_salary(salary_values, n_bins=5):
    """Convert continuous salary values into discrete bins for confusion matrix"""
    min_val = np.min(salary_values)
    max_val = np.max(salary_values)
    
    # Create bins with equal width
    bins = np.linspace(min_val, max_val, n_bins + 1)
    bin_labels = [f"₹{bins[i]:,.0f}-₹{bins[i+1]:,.0f}" for i in range(n_bins)]
    
    # Discretize the values
    discretized = np.digitize(salary_values, bins[1:-1])
    
    return discretized, bin_labels, bins

# Function to find best match using fuzzy matching
def find_best_match(query, choices, threshold=80):
    best_match = None
    best_score = 0
    
    # Clean the query
    query = preprocess_text(query)
    
    for choice in choices:
        # Clean the choice
        clean_choice = preprocess_text(choice)
        
        # Calculate similarity scores
        ratio = fuzz.ratio(query, clean_choice)
        partial_ratio = fuzz.partial_ratio(query, clean_choice)
        token_sort_ratio = fuzz.token_sort_ratio(query, clean_choice)
        token_set_ratio = fuzz.token_set_ratio(query, clean_choice)
        
        # Take the best score
        score = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        if score > best_score:
            best_score = score
            best_match = choice
    
    # Only return a match if it's above the threshold
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

# Function to convert years of experience to experience level
def years_to_experience_level(years_text):
    # Try to extract the number of years from the text
    year_pattern = re.compile(r'(\d+)\s*(?:year|yr)')
    match = year_pattern.search(years_text.lower())
    
    if match:
        years = int(match.group(1))
        if years < 2:
            return "Entry"  # Use standardized format
        elif years < 5:
            return "Mid"    # Use standardized format
        else:
            return "Senior" # Use standardized format
    
    # If we couldn't extract years, return the original text
    return years_text

# Function to extract job role and experience from natural language input
def parse_user_input(user_input):
    # Check for years of experience pattern in the input
    year_pattern = re.compile(r'(\d+)\s*(?:year|yr)')
    year_match = year_pattern.search(user_input.lower())
    
    if year_match:
        years = int(year_match.group(1))
        # Determine experience level based on years
        if years < 2:
            experience = "Entry"  # Use standardized format
        elif years < 5:
            experience = "Mid"    # Use standardized format
        else:
            experience = "Senior" # Use standardized format
            
        # Remove the years part from the job role
        # Look for common phrases that separate job role from experience
        separators = [
            "with", "having", "of", "for", "experience", 
            "experiance", "years", "year", "yr", "yrs"
        ]
        
        job_role = user_input
        for separator in separators:
            parts = user_input.lower().split(separator, 1)
            if len(parts) > 1:
                job_role = parts[0].strip()
                break
                
        return job_role, f"{years} years ({experience})"
    
    # Map experience level terms to standardized values
    exp_terms = {
        "entry": "Entry",
        "junior": "Entry",
        "beginner": "Entry",
        "fresher": "Entry",
        "entry-level": "Entry",
        "mid": "Mid",
        "middle": "Mid",
        "intermediate": "Mid",
        "mid-level": "Mid",
        "senior": "Senior",
        "experienced": "Senior",
        "expert": "Senior",
        "senior-level": "Senior",
        "executive": "Senior"
    }
    
    # If no years pattern is found, use comma to split if present
    if "," in user_input:
        parts = [part.strip() for part in user_input.split(",", 1)]
        job_role = parts[0]
        
        # Check if second part contains any experience level terms
        if len(parts) > 1:
            exp_input = parts[1].lower()
            for term, level in exp_terms.items():
                if term in exp_input:
                    return job_role, level
            
            # If no known term found, return as is
            return job_role, parts[1]
    
    # Try to extract experience level from the whole input
    for term, level in exp_terms.items():
        if term in user_input.lower():
            # Remove the experience term from job role
            job_role = re.sub(r'\b' + term + r'\b', '', user_input, flags=re.IGNORECASE).strip()
            return job_role, level
    
    # If no clear separation, return the whole input as job role
    return user_input, None

class NLPSalaryPredictor:
    def __init__(self, data_path='Expanded_Job_Dataset_Trimmed.csv'):
        # Create results directory for saving evaluation metrics and plots
        self.results_dir = 'model_evaluation_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
        # Create TF-IDF embeddings
        self.create_embeddings()
        
        # Train model
        self.train_model()
    
    def preprocess_data(self):
        # Make sure Skills Required is handled properly
        self.df = self.df[self.df['Skills Required'].notna()]
        
        # Clean salary columns - first replace all non-numeric values
        self.df['Min Salary'] = self.df['Min Salary'].replace('Variable', np.nan)
        self.df['Max Salary'] = self.df['Max Salary'].replace('Variable', np.nan)
        
        # Function to safely convert salaries to numeric values
        def safe_convert_salary(value):
            if not isinstance(value, str):
                return value
            
            # Check if the value looks like a salary (has digits)
            if re.search(r'\d', value):
                # Remove commas
                cleaned = value.replace(',', '')
                try:
                    return float(cleaned)
                except ValueError:
                    return np.nan
            return np.nan
        
        # Apply safe conversion
        self.df['Min Salary'] = self.df['Min Salary'].apply(safe_convert_salary)
        self.df['Max Salary'] = self.df['Max Salary'].apply(safe_convert_salary)
        
        # Drop rows with NaN salary values
        self.df = self.df.dropna(subset=['Min Salary', 'Max Salary'])
        
        # Verify that all salary values are numeric
        print(f"Rows after salary cleaning: {len(self.df)}")
        
        # Create preprocessed text columns
        self.df['Job Role Processed'] = self.df['Job Role'].apply(preprocess_text)
        self.df['Experience Level Processed'] = self.df['Experience Level'].apply(preprocess_text)
        
        # Store unique values
        self.unique_job_roles = self.df['Job Role'].unique()
        self.unique_exp_levels = self.df['Experience Level'].unique()
        
        print(f"Unique experience levels found: {', '.join(sorted(self.unique_exp_levels))}")
    
    def create_embeddings(self):
        # Create TF-IDF vectors for job roles
        self.job_vectorizer = TfidfVectorizer()
        job_tfidf = self.job_vectorizer.fit_transform(self.df['Job Role Processed'])
        self.job_vectors = job_tfidf.toarray()
        
        # Create TF-IDF vectors for experience levels
        self.exp_vectorizer = TfidfVectorizer()
        exp_tfidf = self.exp_vectorizer.fit_transform(self.df['Experience Level Processed'])
        self.exp_vectors = exp_tfidf.toarray()
        
        # Create mapping from vectors to original values
        self.job_to_vector = {job: self.job_vectorizer.transform([preprocess_text(job)]).toarray()[0] 
                             for job in self.unique_job_roles}
        
        self.exp_to_vector = {exp: self.exp_vectorizer.transform([preprocess_text(exp)]).toarray()[0] 
                             for exp in self.unique_exp_levels}
    
    def train_model(self):
        # Create label encoders
        self.job_encoder = LabelEncoder()
        self.exp_encoder = LabelEncoder()
        
        # Fit the encoders
        self.df['Job Role Encoded'] = self.job_encoder.fit_transform(self.df['Job Role'])
        self.df['Experience Level Encoded'] = self.exp_encoder.fit_transform(self.df['Experience Level'])
        
        # Create feature matrix using only Job Role and Experience Level
        X = self.df[['Job Role Encoded', 'Experience Level Encoded']]
        y_min = self.df['Min Salary']
        y_max = self.df['Max Salary']
        
        # Split data
        X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = train_test_split(
            X, y_min, y_max, test_size=0.2, random_state=42
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_min_test = y_min_test
        self.y_max_test = y_max_test
        
        # Train Random Forest models for Min and Max salary
        self.min_salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.min_salary_model.fit(X_train, y_min_train)
        
        self.max_salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.max_salary_model.fit(X_train, y_max_train)
        
        # Calculate predictions on test set
        self.y_min_pred = self.min_salary_model.predict(X_test)
        self.y_max_pred = self.max_salary_model.predict(X_test)
        
        # Create discretized versions for confusion matrix
        self.y_min_test_disc, self.min_salary_bands, self.min_salary_bins = discretize_salary(y_min_test)
        self.y_max_test_disc, self.max_salary_bands, self.max_salary_bins = discretize_salary(y_max_test)
        
        # Discretize predictions using the same bins
        self.y_min_pred_disc = np.digitize(self.y_min_pred, self.min_salary_bins[1:-1])
        self.y_max_pred_disc = np.digitize(self.y_max_pred, self.max_salary_bins[1:-1])
        
        # Evaluate models
        self.evaluate_models()
    
    def evaluate_models(self):
        """Evaluate the performance of the models and save metrics and visualizations"""
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics for min 
        min_r2 = r2_score(self.y_min_test, self.y_min_pred)
        min_mae = mean_absolute_error(self.y_min_test, self.y_min_pred)
        min_rmse = np.sqrt(mean_squared_error(self.y_min_test, self.y_min_pred))
        min_evs = explained_variance_score(self.y_min_test, self.y_min_pred)
        
        # Calculate metrics for max 
        max_r2 = r2_score(self.y_max_test, self.y_max_pred)
        max_mae = mean_absolute_error(self.y_max_test, self.y_max_pred)
        max_rmse = np.sqrt(mean_squared_error(self.y_max_test, self.y_max_pred))
        max_evs = explained_variance_score(self.y_max_test, self.y_max_pred)
        
        # Print metrics
        print("\n===== Model Evaluation Metrics =====")
        print("\nMinimum :")
        print(f"R² Score: {min_r2:.4f}")
        print(f"Mean Absolute Error: ₹{min_mae:.2f}")
        print(f"Root Mean Squared Error: ₹{min_rmse:.2f}")
        print(f"Explained Variance Score: {min_evs:.4f}")
        
        print("\nMaximum :")
        print(f"R² Score: {max_r2:.4f}")
        print(f"Mean Absolute Error: ₹{max_mae:.2f}")
        print(f"Root Mean Squared Error: ₹{max_rmse:.2f}")
        print(f"Explained Variance Score: {max_evs:.4f}")
        
        # Create a dictionary to store all metrics
        self.model_metrics = {
            'min_salary': {
                'r2': min_r2,
                'mae': min_mae,
                'rmse': min_rmse,
                'evs': min_evs
            },
            'max_salary': {
                'r2': max_r2,
                'mae': max_mae,
                'rmse': max_rmse,
                'evs': max_evs
            }
        }
        
        # Generate and save visualizations
        self.plot_prediction_error(self.y_min_test, self.y_min_pred, 'min', timestamp)
        self.plot_prediction_error(self.y_max_test, self.y_max_pred, 'max', timestamp)
        self.plot_feature_importance(timestamp)
        self.plot_precision_matrix(timestamp)
        self.plot_confusion_matrices(timestamp)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE', 'EVS'],
            'Min Salary ': [min_r2, min_mae, min_rmse, min_evs],
            'Max Salary ': [max_r2, max_mae, max_rmse, max_evs]
        })
        
        metrics_file = f"{self.results_dir}/model_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nMetrics saved to {metrics_file}")
    
    
    def find_similar_job_role(self, query_job, threshold=80):
        # Try fuzzy matching first
        best_match, score = find_best_match(query_job, self.unique_job_roles, threshold)
        
        if best_match:
            print(f"Fuzzy matched '{query_job}' to '{best_match}' with score {score}")
            return best_match
        
        # If no good fuzzy match, try embeddings
        query_job_processed = preprocess_text(query_job)
        query_vector = self.job_vectorizer.transform([query_job_processed]).toarray()[0]
        
        # Calculate cosine similarity with all job vectors
        similarities = {}
        for job in self.unique_job_roles:
            job_vector = self.job_to_vector[job]
            similarity = cosine_similarity([query_vector], [job_vector])[0][0]
            similarities[job] = similarity
        
        # Find the best match
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]
        
        # Only return if similarity is above threshold
        if best_score >= 0.5:  # Setting a cosine similarity threshold
            print(f"Embedding matched '{query_job}' to '{best_match}' with score {best_score:.4f}")
            return best_match
        
        return None
    
    def find_similar_experience_level(self, query_exp, threshold=80):
        # First check if this already contains a standardized level name
        if query_exp in ["Entry", "Mid", "Senior"]:
            return query_exp
            
        # If it's a standard pattern like "X years (Level)"
        if re.search(r'\d+\s*(?:year|yr).*\((Entry|Mid|Senior)\)', query_exp):
            level_match = re.search(r'\((Entry|Mid|Senior)\)', query_exp)
            if level_match:
                return level_match.group(1)
        
        # Check for years of experience pattern
        if re.search(r'\d+\s*(?:year|yr)', query_exp.lower()):
            # Convert years to experience level
            experience_level = years_to_experience_level(query_exp)
            print(f"Converted years '{query_exp}' to experience level: '{experience_level}'")
            return experience_level
        
        # Try fuzzy matching first
        best_match, score = find_best_match(query_exp, self.unique_exp_levels, threshold)
        
        if best_match:
            print(f"Fuzzy matched '{query_exp}' to '{best_match}' with score {score}")
            return best_match
        
        # If no good fuzzy match, try embeddings
        query_exp_processed = preprocess_text(query_exp)
        query_vector = self.exp_vectorizer.transform([query_exp_processed]).toarray()[0]
        
        # Calculate cosine similarity with all experience vectors
        similarities = {}
        for exp in self.unique_exp_levels:
            exp_vector = self.exp_to_vector[exp]
            similarity = cosine_similarity([query_vector], [exp_vector])[0][0]
            similarities[exp] = similarity
        
        # Find the best match
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]
        
        # Only return if similarity is above threshold
        if best_score >= 0.5:  # Setting a cosine similarity threshold
            print(f"Embedding matched '{query_exp}' to '{best_match}' with score {best_score:.4f}")
            return best_match
        
        # If all else fails, default to "Entry"
        print(f"Could not match '{query_exp}', defaulting to 'Entry'")
        return "Entry"
    
    def predict_salary(self, job_role, experience_level=None):
        # If experience level is not provided, default to Entry
        if experience_level is None or experience_level.strip() == "":
            experience_level = "Entry"
            print(f"No experience level provided. Using default: {experience_level}")
            
        # Find similar job role and experience level
        matched_job = self.find_similar_job_role(job_role)
        matched_exp = self.find_similar_experience_level(experience_level)
        
        if not matched_job:
            return f"Could not find a matching job role for '{job_role}'."
        
        if not matched_exp:
            return f"Could not find a matching experience level for '{experience_level}'."
        
        # Encode the matched values
        job_encoded = self.job_encoder.transform([matched_job])[0]
        exp_encoded = self.exp_encoder.transform([matched_exp])[0]
        
        # Make prediction
        features = np.array([[job_encoded, exp_encoded]])
        min_salary_pred = self.min_salary_model.predict(features)[0]
        max_salary_pred = self.max_salary_model.predict(features)[0]
        
        # Get the latest evaluation images
        eval_dir = self.results_dir
        precision_matrix_files = [f for f in os.listdir(eval_dir) if f.startswith('precision_matrix_')]
        prediction_error_files = [f for f in os.listdir(eval_dir) if f.startswith('min_salary_prediction_')]
        confusion_matrix_files = [f for f in os.listdir(eval_dir) if f.startswith('confusion_matrices_')]
        
        # Get the most recent files
        latest_precision_matrix = sorted(precision_matrix_files)[-1] if precision_matrix_files else None
        latest_prediction_error = sorted(prediction_error_files)[-1] if prediction_error_files else None
        latest_confusion_matrix = sorted(confusion_matrix_files)[-1] if confusion_matrix_files else None
        
        precision_matrix_path = f"{eval_dir}/{latest_precision_matrix}" if latest_precision_matrix else None
        prediction_error_path = f"{eval_dir}/{latest_prediction_error}" if latest_prediction_error else None
        confusion_matrix_path = f"{eval_dir}/{latest_confusion_matrix}" if latest_confusion_matrix else None
        
        # Return formatted result with paths to evaluation images
        return {
            "job_role_matched": matched_job,
            "experience_level_matched": matched_exp,
            "min_salary": min_salary_pred,
            "max_salary": max_salary_pred,
            "precision_matrix_path": precision_matrix_path,
            "prediction_error_path": prediction_error_path,
            "confusion_matrix_path": confusion_matrix_path,
            "model_metrics": self.model_metrics if hasattr(self, 'model_metrics') else None,
            "formatted_result": f"Predicted Salary Range for {matched_job} ({matched_exp}):\n\nMinimum Salary: ₹{min_salary_pred:,.2f}\nMaximum Salary: ₹{max_salary_pred:,.2f}"
        }
    
    def save_model(self, path='nlp_salary_model'):
        """Save the model and all its components"""
        os.makedirs(path, exist_ok=True)
        
        # Save the models
        joblib.dump(self.min_salary_model, f'{path}/min_salary_model.pkl')
        joblib.dump(self.max_salary_model, f'{path}/max_salary_model.pkl')
        
        # Save the encoders
        joblib.dump(self.job_encoder, f'{path}/job_encoder.pkl')
        joblib.dump(self.exp_encoder, f'{path}/exp_encoder.pkl')
        
        # Save the vectorizers
        joblib.dump(self.job_vectorizer, f'{path}/job_vectorizer.pkl')
        joblib.dump(self.exp_vectorizer, f'{path}/exp_vectorizer.pkl')
        
        # Save unique values
        joblib.dump(self.unique_job_roles, f'{path}/unique_job_roles.pkl')
        joblib.dump(self.unique_exp_levels, f'{path}/unique_exp_levels.pkl')
        
        # Save mappings
        joblib.dump(self.job_to_vector, f'{path}/job_to_vector.pkl')
        joblib.dump(self.exp_to_vector, f'{path}/exp_to_vector.pkl')
        
        # Save model metrics if available
        if hasattr(self, 'model_metrics'):
            joblib.dump(self.model_metrics, f'{path}/model_metrics.pkl')
        
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path='nlp_salary_model'):
        """Load a pre-trained model"""
        model = cls.__new__(cls)
        
        # Load the models
        model.min_salary_model = joblib.load(f'{path}/min_salary_model.pkl')
        model.max_salary_model = joblib.load(f'{path}/max_salary_model.pkl')
        
        # Load the encoders
        model.job_encoder = joblib.load(f'{path}/job_encoder.pkl')
        model.exp_encoder = joblib.load(f'{path}/exp_encoder.pkl')
        
        # Load the vectorizers
        model.job_vectorizer = joblib.load(f'{path}/job_vectorizer.pkl')
        model.exp_vectorizer = joblib.load(f'{path}/exp_vectorizer.pkl')
        
        # Load unique values
        model.unique_job_roles = joblib.load(f'{path}/unique_job_roles.pkl')
        model.unique_exp_levels = joblib.load(f'{path}/unique_exp_levels.pkl')
        
        # Load mappings
        model.job_to_vector = joblib.load(f'{path}/job_to_vector.pkl')
        model.exp_to_vector = joblib.load(f'{path}/exp_to_vector.pkl')
        
        # Create results directory
        model.results_dir = 'model_evaluation_results'
        os.makedirs(model.results_dir, exist_ok=True)
        
        # Load model metrics if available
        metrics_path = f'{path}/model_metrics.pkl'
        if os.path.exists(metrics_path):
            model.model_metrics = joblib.load(metrics_path)
        
        print(f"Model loaded from {path}")
        return model

    def plot_prediction_error(self, y_true, y_pred, salary_type, timestamp):
        """Plot prediction error for salary predictions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Salary (₹)')
        plt.ylabel('Predicted Salary (₹)')
        plt.title(f'{salary_type.capitalize()} Salary Prediction Error')
        
        # Save the plot
        plot_file = f"{self.results_dir}/{salary_type}_salary_prediction_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Prediction error plot saved to {plot_file}")

    def plot_feature_importance(self, timestamp):
        """Plot feature importance for the models."""
        plt.figure(figsize=(12, 6))
        
        # Get feature importance for both models
        min_importance = self.min_salary_model.feature_importances_
        max_importance = self.max_salary_model.feature_importances_
        
        # Plot min salary feature importance
        plt.subplot(1, 2, 1)
        plt.bar(range(len(min_importance)), min_importance)
        plt.title('Min Salary Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        # Plot max salary feature importance
        plt.subplot(1, 2, 2)
        plt.bar(range(len(max_importance)), max_importance)
        plt.title('Max Salary Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.results_dir}/feature_importance_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Feature importance plot saved to {plot_file}")

    def plot_precision_matrix(self, timestamp):
        """Plot precision matrix for salary predictions."""
        plt.figure(figsize=(10, 8))
        
        # Calculate precision matrix
        min_precision = np.corrcoef(self.y_min_test, self.y_min_pred)[0, 1]
        max_precision = np.corrcoef(self.y_max_test, self.y_max_pred)[0, 1]
        
        # Create heatmap
        precision_matrix = np.array([[min_precision, 0], [0, max_precision]])
        sns.heatmap(precision_matrix, annot=True, cmap='YlGnBu', 
                   xticklabels=['Min', 'Max'], yticklabels=['Min', 'Max'])
        plt.title('Salary Prediction Precision Matrix')
        
        # Save the plot
        plot_file = f"{self.results_dir}/precision_matrix_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Precision matrix plot saved to {plot_file}")

    def plot_confusion_matrices(self, timestamp):
        """Plot confusion matrices for discretized salary predictions."""
        plt.figure(figsize=(15, 6))
        
        # Calculate confusion matrices
        min_cm = confusion_matrix(self.y_min_test_disc, self.y_min_pred_disc)
        max_cm = confusion_matrix(self.y_max_test_disc, self.y_max_pred_disc)
        
        # Plot min salary confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(min_cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Min Salary Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot max salary confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(max_cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Max Salary Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.results_dir}/confusion_matrices_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Confusion matrices plot saved to {plot_file}")

# If running as a script
if __name__ == "__main__":
    # Check if model exists
    model_path = 'nlp_salary_model'
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Load pre-trained model
        print("Loading pre-trained model...")
        model = NLPSalaryPredictor.load_model(model_path)
    else:
        # Train new model
        print("Training new model...")
        model = NLPSalaryPredictor()
        model.save_model(model_path)
    
    # Interactive testing loop
    print("\nSalary Prediction Model with NLP")
    print("--------------------------------")
    print("This model can match similar job roles and experience levels using NLP techniques")
    print("Format examples:")
    print("1. 'Software Engineer, Senior'")
    print("2. 'Data Scientist' (defaults to Entry)")
    print("3. 'AI Engineer with 5 years experience'")
    print("4. 'salary for Machine Learning Engineer with 3 years'")
    print("Years of experience: 1 year = Entry, 3 years = Mid, 5+ years = Senior")
    
    while True:
        print("\nEnter 'q' to quit")
        
        # Get input in one line
        user_input = input("\nEnter query: ")
        if user_input.lower() == 'q':
            break
            
        # Parse natural language input to extract job role and experience
        job_role, experience_level = parse_user_input(user_input)
        
        print(f"Parsed job role: '{job_role}'")
        if experience_level:
            print(f"Parsed experience level: '{experience_level}'")
        
        # Make prediction
        result = model.predict_salary(job_role, experience_level)
        
        # Print result
        if isinstance(result, dict):
            print(f"\n{result['formatted_result']}")
        else:
            print(f"\n{result}") 