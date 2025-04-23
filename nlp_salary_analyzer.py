import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import joblib
import os
from datetime import datetime

# Set style for visualizations
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.2)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="whitegrid", rc=custom_params)

class SalaryDataset(Dataset):
    def __init__(self, texts, salaries, tokenizer, max_length=128):
        self.texts = texts
        self.salaries = salaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        salary = self.salaries[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'salary': torch.tensor(salary, dtype=torch.float32)
        }

class SalaryPredictor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.1):
        super(SalaryPredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.linear(pooled_output)

class SalaryAnalyzer:
    def __init__(self, data_dir="data/scraped", model_path="models/salary_predictor.pth"):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = SalaryPredictor().to(self.device)
        
        # Create models directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            print("Loaded pre-trained model")
        
        self.results_dir = Path("salary_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load or create dataset
        self.df = self._load_or_create_dataset()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.df['Median Salary'].values.reshape(-1, 1))
        joblib.dump(self.scaler, self.results_dir / "salary_scaler.pkl")
        
    def _load_or_create_dataset(self):
        """Load existing dataset or create from text files"""
        dataset_path = self.results_dir / "career_salary_data.csv"
        
        if dataset_path.exists():
            print("Loading existing dataset...")
            return pd.read_csv(dataset_path)
        
        print("Creating new dataset from text files...")
        career_data = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                career_name = os.path.splitext(filename)[0]
                filepath = self.data_dir / filename
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        
                        # Extract features
                        education_level = self._determine_education_level(content)
                        industry = self._determine_industry(career_name, content)
                        skills = self._extract_skills(content)
                        
                        # Simulate salary data
                        median_salary = self._simulate_salary(education_level, industry)
                        salary_range = (int(median_salary * 0.7), int(median_salary * 1.3))
                        
                        career_data.append({
                            'Career': career_name,
                            'Description': content,
                            'Education': education_level,
                            'Industry': industry,
                            'Skills': skills,
                            'Median Salary': median_salary,
                            'Salary Range Min': salary_range[0],
                            'Salary Range Max': salary_range[1]
                        })
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        df = pd.DataFrame(career_data)
        df.to_csv(dataset_path, index=False)
        return df
    
    def train_model(self, epochs=20, batch_size=16, learning_rate=1e-5):
        """Train the salary prediction model"""
        print("Preparing data for training...")
        
        # Split data
        X = self.df['Description'].values
        y = self.df['Median Salary'].values
        
        # Scale salaries
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = SalaryDataset(X_train, y_train, self.tokenizer)
        test_dataset = SalaryDataset(X_test, y_test, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
        
        # Metrics tracking
        train_losses = []
        test_losses = []
        learning_rates = []
        train_r2_scores = []
        test_r2_scores = []
        best_test_loss = float('inf')
        
        # Training loop
        print("Starting training...")
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            self.model.train()
            total_loss = 0
            train_predictions = []
            train_actuals = []
            
            # Training phase
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                salaries = batch['salary'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), salaries)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                train_actuals.extend(salaries.cpu().numpy())
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_r2 = r2_score(train_actuals, train_predictions)
            train_r2_scores.append(train_r2)
            
            # Evaluation phase
            self.model.eval()
            test_loss = 0
            test_predictions = []
            test_actuals = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} - Testing'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    salaries = batch['salary'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), salaries)
                    test_loss += loss.item()
                    test_predictions.extend(outputs.squeeze().cpu().numpy())
                    test_actuals.extend(salaries.cpu().numpy())
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            test_r2 = r2_score(test_actuals, test_predictions)
            test_r2_scores.append(test_r2)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Update learning rate
            scheduler.step(avg_test_loss)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Saved new best model with test loss: {best_test_loss:.4f}")
            
            # Calculate epoch time
            epoch_time = datetime.now() - epoch_start_time
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Train R²: {train_r2:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}, Test R²: {test_r2:.4f}')
            print(f'Learning Rate: {current_lr:.2e}')
            print(f'Time: {epoch_time.total_seconds():.2f}s')
            
            # Early stopping check
            if epoch > 5 and avg_test_loss > best_test_loss * 1.1:
                print("Early stopping triggered")
                break
        
        total_time = datetime.now() - start_time
        print(f"\nTotal training time: {total_time.total_seconds():.2f}s")
        
        # Plot training history
        self._plot_training_history(train_losses, test_losses, train_r2_scores, test_r2_scores, learning_rates)
        
        return train_losses, test_losses
    
    def _plot_training_history(self, train_losses, test_losses, train_r2_scores, test_r2_scores, learning_rates):
        """Plot detailed training history"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Loss curves
        ax1 = plt.subplot(2, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, test_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12) 
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        
        # 2. R² Score progression
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, train_r2_scores, 'g-', label='Training R²')
        ax2.plot(epochs, test_r2_scores, 'm-', label='Validation R²')
        ax2.set_title('R² Score Progression', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        
        # 3. Learning rate progression
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(epochs, learning_rates, 'c-')
        ax3.set_title('Learning Rate Progression', fontsize=14)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 4. Loss ratio (validation/training)
        ax4 = plt.subplot(2, 2, 4)
        loss_ratio = [test/train for test, train in zip(test_losses, train_losses)]
        ax4.plot(epochs, loss_ratio, 'y-')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax4.set_title('Validation/Training Loss Ratio', fontsize=14)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Ratio', fontsize=12)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create training summary report
        report = f"""
        Training Summary Report
        ======================
        
        Training Duration: {len(train_losses)} epochs
        
        Final Metrics:
        - Training Loss: {train_losses[-1]:.4f}
        - Validation Loss: {test_losses[-1]:.4f}
        - Training R² Score: {train_r2_scores[-1]:.4f}
        - Validation R² Score: {test_r2_scores[-1]:.4f}
        - Final Learning Rate: {learning_rates[-1]:.2e}
        
        Best Metrics:
        - Best Training Loss: {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses))+1})
        - Best Validation Loss: {min(test_losses):.4f} (Epoch {test_losses.index(min(test_losses))+1})
        - Best Training R² Score: {max(train_r2_scores):.4f} (Epoch {train_r2_scores.index(max(train_r2_scores))+1})
        - Best Validation R² Score: {max(test_r2_scores):.4f} (Epoch {test_r2_scores.index(max(test_r2_scores))+1})
        
        Learning Rate:
        - Initial: {learning_rates[0]:.2e}
        - Final: {learning_rates[-1]:.2e}
        - Number of adjustments: {sum(1 for i in range(1, len(learning_rates)) if learning_rates[i] != learning_rates[i-1])}
        """
        
        with open(self.results_dir / "training_summary.txt", "w") as f:
            f.write(report)
    
    def evaluate_model(self):
        """Evaluate model performance and generate insights"""
        print("Evaluating model performance...")
        
        # Prepare test data
        X = self.df['Description'].values
        y = self.df['Median Salary'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in X_test:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                output = self.model(input_ids, attention_mask)
                predictions.append(output.item())
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Create performance report
        performance = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([performance])
        metrics_df.to_csv(self.results_dir / "model_metrics.csv", index=False)
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.title('Actual vs Predicted Salaries')
        plt.xlabel('Actual Salary ($)')
        plt.ylabel('Predicted Salary ($)')
        plt.savefig(self.results_dir / "predictions_vs_actual.png", dpi=300)
        plt.close()
        
        # Plot error distribution
        errors = y_test - predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error ($)')
        plt.ylabel('Count')
        plt.savefig(self.results_dir / "error_distribution.png", dpi=300)
        plt.close()
        
        return performance
    
    def generate_insights(self):
        """Generate comprehensive insights from the dataset"""
        print("Generating insights...")
        
        # 1. Salary Distribution by Industry (Enhanced)
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Industry', y='Median Salary', data=self.df, palette='viridis')
        plt.title('Salary Distribution by Industry', fontsize=16)
        plt.xlabel('Industry', fontsize=14)
        plt.ylabel('Median Salary ($)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "salary_by_industry.png", dpi=300)
        plt.close()
        
        # 2. Education Impact (Enhanced)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Education', y='Median Salary', data=self.df, palette='muted')
        plt.title('Average Salary by Education Level', fontsize=16)
        plt.xlabel('Education Level', fontsize=14)
        plt.ylabel('Average Salary ($)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "salary_by_education.png", dpi=300)
        plt.close()
        
        # 3. Skills Analysis (Enhanced)
        all_skills = []
        for skills in self.df['Skills']:
            all_skills.extend(skills)
        
        skill_counts = pd.Series(all_skills).value_counts().head(20)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='rocket')
        plt.title('Top 20 Most Common Skills', fontsize=16)
        plt.xlabel('Frequency', fontsize=14)
        plt.ylabel('Skill', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "top_skills.png", dpi=300)
        plt.close()
        
        # 4. Salary Range Analysis (Enhanced)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Median Salary', y='Salary Range Max', 
                       data=self.df, hue='Industry', size='Salary Range Max',
                       palette='viridis', alpha=0.7)
        plt.title('Salary Range Analysis by Industry', fontsize=16)
        plt.xlabel('Median Salary ($)', fontsize=14)
        plt.ylabel('Maximum Salary ($)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "salary_range_analysis.png", dpi=300)
        plt.close()
        
        # 5. Industry-Education Interaction (Enhanced)
        pivot_table = self.df.pivot_table(
            values='Median Salary',
            index='Industry',
            columns='Education',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu',
                   cbar_kws={'label': 'Average Salary ($)'})
        plt.title('Average Salary by Industry and Education', fontsize=16)
        plt.xlabel('Education Level', fontsize=14)
        plt.ylabel('Industry', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "industry_education_heatmap.png", dpi=300)
        plt.close()
        
        # 6. Salary Distribution (New)
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.df, x='Median Salary', kde=True, bins=30)
        plt.title('Overall Salary Distribution', fontsize=16)
        plt.xlabel('Salary ($)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "salary_distribution.png", dpi=300)
        plt.close()
        
        # 7. Education Level Distribution (New)
        plt.figure(figsize=(10, 8))
        education_counts = self.df['Education'].value_counts()
        plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('pastel'), startangle=90)
        plt.title('Distribution of Education Levels', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.results_dir / "education_distribution.png", dpi=300)
        plt.close()
        
        # 8. Industry Distribution (New)
        plt.figure(figsize=(12, 8))
        industry_counts = self.df['Industry'].value_counts()
        sns.barplot(x=industry_counts.values, y=industry_counts.index, palette='husl')
        plt.title('Distribution of Industries', fontsize=16)
        plt.xlabel('Number of Careers', fontsize=14)
        plt.ylabel('Industry', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "industry_distribution.png", dpi=300)
        plt.close()
        
        # 9. Salary vs Education Level by Industry (New)
        plt.figure(figsize=(14, 10))
        sns.boxplot(x='Education', y='Median Salary', hue='Industry', data=self.df)
        plt.title('Salary Distribution by Education Level and Industry', fontsize=16)
        plt.xlabel('Education Level', fontsize=14)
        plt.ylabel('Median Salary ($)', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.results_dir / "salary_education_industry.png", dpi=300)
        plt.close()
        
        # 10. Top Paying Careers (New)
        top_careers = self.df.nlargest(10, 'Median Salary')
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Median Salary', y='Career', data=top_careers, palette='viridis')
        plt.title('Top 10 Highest Paying Careers', fontsize=16)
        plt.xlabel('Median Salary ($)', fontsize=14)
        plt.ylabel('Career', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / "top_paying_careers.png", dpi=300)
        plt.close()
        
        # Generate detailed summary statistics
        summary_stats = {
            'Total Careers': len(self.df),
            'Average Salary': self.df['Median Salary'].mean(),
            'Median Salary': self.df['Median Salary'].median(),
            'Highest Salary': self.df['Median Salary'].max(),
            'Lowest Salary': self.df['Median Salary'].min(),
            'Salary Standard Deviation': self.df['Median Salary'].std(),
            'Salary Range': f"${self.df['Median Salary'].min():,.2f} - ${self.df['Median Salary'].max():,.2f}",
            'Most Common Industry': self.df['Industry'].mode()[0],
            'Most Common Education Level': self.df['Education'].mode()[0],
            'Number of Industries': self.df['Industry'].nunique(),
            'Number of Education Levels': self.df['Education'].nunique(),
            'Average Salary Range': f"${self.df['Salary Range Min'].mean():,.2f} - ${self.df['Salary Range Max'].mean():,.2f}",
            'Top 3 Industries by Average Salary': self.df.groupby('Industry')['Median Salary'].mean().nlargest(3).to_dict(),
            'Top 3 Education Levels by Average Salary': self.df.groupby('Education')['Median Salary'].mean().nlargest(3).to_dict()
        }
        
        # Save summary statistics
        pd.DataFrame([summary_stats]).to_csv(self.results_dir / "summary_statistics.csv", index=False)
        
        # Create detailed report
        report = f"""
        Salary Analysis Report
        =====================
        
        Dataset Overview:
        - Total Careers: {summary_stats['Total Careers']}
        - Number of Industries: {summary_stats['Number of Industries']}
        - Number of Education Levels: {summary_stats['Number of Education Levels']}
        
        Salary Statistics:
        - Average Salary: ${summary_stats['Average Salary']:,.2f}
        - Median Salary: ${summary_stats['Median Salary']:,.2f}
        - Salary Range: {summary_stats['Salary Range']}
        - Standard Deviation: ${summary_stats['Salary Standard Deviation']:,.2f}
        
        Industry Analysis:
        - Most Common Industry: {summary_stats['Most Common Industry']}
        - Top 3 Industries by Average Salary:
        {chr(10).join(f'  - {industry}: ${salary:,.2f}' for industry, salary in summary_stats['Top 3 Industries by Average Salary'].items())}
        
        Education Analysis:
        - Most Common Education Level: {summary_stats['Most Common Education Level']}
        - Top 3 Education Levels by Average Salary:
        {chr(10).join(f'  - {education}: ${salary:,.2f}' for education, salary in summary_stats['Top 3 Education Levels by Average Salary'].items())}
        
        Salary Range Analysis:
        - Average Salary Range: {summary_stats['Average Salary Range']}
        """
        
        with open(self.results_dir / "detailed_report.txt", "w") as f:
            f.write(report)
        
        return summary_stats
    
    def predict_salary(self, job_description):
        """Predict salary for a given job description"""
        self.model.eval()
        
        encoding = self.tokenizer(
            job_description,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            predicted_salary_scaled = output.item()
            predicted_salary = self.scaler.inverse_transform([[predicted_salary_scaled]])[0][0]
        
        return predicted_salary
    
    def _determine_education_level(self, content):
        """Determine education level from content"""
        content_lower = content.lower()
        if "phd" in content_lower or "doctoral" in content_lower:
            return "Doctoral Degree"
        elif "master" in content_lower or "graduate degree" in content_lower:
            return "Master's Degree"
        elif "bachelor" in content_lower or "college degree" in content_lower:
            return "Bachelor's Degree"
        elif "associate" in content_lower or "technical degree" in content_lower:
            return "Associate's Degree"
        else:
            return "High School"
    
    def _determine_industry(self, career_name, content):
        """Determine industry from career name and content"""
        content_lower = content.lower()
        career_lower = career_name.lower()
        
        if any(term in career_lower for term in ["developer", "programmer", "software", "web", "it"]):
            return "Technology"
        elif any(term in career_lower for term in ["doctor", "nurse", "medical", "health"]):
            return "Healthcare"
        elif any(term in career_lower for term in ["accountant", "financial", "bank", "insurance"]):
            return "Finance"
        elif any(term in career_lower for term in ["teacher", "professor", "education"]):
            return "Education"
        elif any(term in career_lower for term in ["engineer", "architect"]):
            return "Engineering"
        else:
            return "Other"
    
    def _extract_skills(self, content):
        """Extract skills from content"""
        common_skills = [
            "Communication", "Programming", "Analysis", "Design", "Management",
            "Leadership", "Problem Solving", "Critical Thinking", "Teamwork", "Writing",
            "Research", "Database", "Python", "Java", "JavaScript", "SQL", "Excel",
            "Project Management", "Customer Service", "Sales", "Marketing", "Accounting"
        ]
        
        found_skills = []
        content_lower = content.lower()
        
        for skill in common_skills:
            if skill.lower() in content_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _simulate_salary(self, education_level, industry):
        """Simulate realistic salary based on education and industry"""
        education_base = {
            "High School": 35000,
            "Associate's Degree": 45000,
            "Bachelor's Degree": 65000,
            "Master's Degree": 85000,
            "Doctoral Degree": 110000
        }
        
        industry_multiplier = {
            "Technology": 1.3,
            "Healthcare": 1.2,
            "Finance": 1.25,
            "Education": 0.85,
            "Engineering": 1.2,
            "Other": 1.0
        }
        
        base = education_base.get(education_level, 50000)
        multiplier = industry_multiplier.get(industry, 1.0)
        variation = np.random.uniform(0.9, 1.1)
        
        return int(base * multiplier * variation)

def main():
    analyzer = SalaryAnalyzer()
    
    # Train model if not already trained
    if not analyzer.model_path.exists():
        print("Training model...")
        analyzer.train_model()
    
    # Evaluate model
    performance = analyzer.evaluate_model()
    print("\nModel Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate insights
    insights = analyzer.generate_insights()
    print("\nDataset Insights:")
    for key, value in insights.items():
        print(f"{key}: {value}")
    
    # Example prediction
    test_description = "Senior Software Engineer with 5 years of experience in Python and Machine Learning"
    predicted_salary = analyzer.predict_salary(test_description)
    print(f"\nExample Prediction for '{test_description}':")
    print(f"Predicted Salary: ${predicted_salary:,.2f}")

if __name__ == "__main__":
    main() 