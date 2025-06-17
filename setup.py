#!/usr/bin/env python3
"""
Smart Health Risk Analyzer - Setup Script

This script helps users set up the project environment and download required datasets.
Run this script after installing dependencies to prepare the project for use.

Usage:
    python setup.py [--download-data] [--setup-env] [--verify]
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import zipfile
import logging
from pathlib import Path
import requests
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthAnalyzerSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Dataset URLs (using Kaggle API or direct links when available)
        self.dataset_urls = {
            'heart': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'diabetes': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',
            'stroke': None,  # This would need to be downloaded manually from Kaggle
            'insurance': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
        }
        
        # Required columns for each dataset
        self.required_columns = {
            'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition'],
            'diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
            'stroke': ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'],
            'insurance': ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        }
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directory structure...")
        
        directories = [self.data_dir, self.models_dir, self.models_dir / "clustering", self.logs_dir]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"   ✓ {directory}")
        
        logger.info("✅ Directory structure created!")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        logger.info("📦 Checking dependencies...")
        
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'scikit-learn', 'xgboost', 
            'lightgbm', 'tensorflow', 'shap', 'lime', 'optuna', 
            'plotly', 'seaborn', 'matplotlib', 'imbalanced-learn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"   ✓ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"   ❌ {package}")
        
        if missing_packages:
            logger.error("❌ Missing required packages. Install with:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✅ All dependencies are installed!")
        return True
    
    def download_sample_data(self):
        """Download and prepare sample datasets"""
        logger.info("⬇️ Downloading sample datasets...")
        
        # Download insurance dataset
        if not (self.data_dir / "insurance.csv").exists():
            logger.info("   Downloading insurance dataset...")
            try:
                response = requests.get(self.dataset_urls['insurance'])
                with open(self.data_dir / "insurance.csv", 'w') as f:
                    f.write(response.text)
                logger.info("   ✓ insurance.csv downloaded")
            except Exception as e:
                logger.error(f"   ❌ Failed to download insurance.csv: {e}")
        
        # Download diabetes dataset
        if not (self.data_dir / "diabetes.csv").exists():
            logger.info("   Downloading diabetes dataset...")
            try:
                response = requests.get(self.dataset_urls['diabetes'])
                with open(self.data_dir / "diabetes.csv", 'w') as f:
                    f.write(response.text)
                logger.info("   ✓ diabetes.csv downloaded")
            except Exception as e:
                logger.error(f"   ❌ Failed to download diabetes.csv: {e}")
        
        # Create sample datasets for heart and stroke if not available
        self.create_sample_heart_data()
        self.create_sample_stroke_data()
        
        logger.info("✅ Sample datasets prepared!")
    
    def create_sample_heart_data(self):
        """Create sample heart disease dataset"""
        if (self.data_dir / "heart_cleveland_upload.csv").exists():
            return
        
        logger.info("   Creating sample heart disease dataset...")
        
        import pandas as pd
        import numpy as np
        
        # Create synthetic heart disease data
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'age': np.random.normal(54, 9, n_samples).astype(int),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.normal(132, 17, n_samples),
            'chol': np.random.normal(246, 51, n_samples),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.normal(149, 22, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.exponential(1, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'ca': np.random.choice([0, 1, 2, 3], n_samples),
            'thal': np.random.choice([0, 1, 2], n_samples)
        }
        
        # Create target variable with some correlation to features
        risk_score = (
            (data['age'] - 40) * 0.02 +
            data['sex'] * 0.3 +
            data['cp'] * 0.2 +
            (data['trestbps'] - 120) * 0.005 +
            (data['chol'] - 200) * 0.001 +
            data['exang'] * 0.4
        )
        
        data['condition'] = (risk_score + np.random.normal(0, 0.3, n_samples) > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / "heart_cleveland_upload.csv", index=False)
        logger.info("   ✓ heart_cleveland_upload.csv created")
    
    def create_sample_stroke_data(self):
        """Create sample stroke dataset"""
        if (self.data_dir / "healthcare-dataset-stroke-data.csv").exists():
            return
        
        logger.info("   Creating sample stroke dataset...")
        
        import pandas as pd
        import numpy as np
        
        # Create synthetic stroke data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'id': range(1, n_samples + 1),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'age': np.random.normal(43, 22, n_samples).clip(1, 82),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'heart_disease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'ever_married': np.random.choice(['No', 'Yes'], n_samples, p=[0.35, 0.65]),
            'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], n_samples),
            'Residence_type': np.random.choice(['Urban', 'Rural'], n_samples),
            'avg_glucose_level': np.random.normal(106, 45, n_samples).clip(55, 272),
            'bmi': np.random.normal(29, 8, n_samples).clip(10, 98),
            'smoking_status': np.random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown'], n_samples)
        }
        
        # Create target variable
        risk_score = (
            (data['age'] - 40) * 0.02 +
            data['hypertension'] * 1.5 +
            data['heart_disease'] * 2.0 +
            (data['avg_glucose_level'] - 100) * 0.005 +
            (data['bmi'] - 25) * 0.02
        )
        
        data['stroke'] = (risk_score + np.random.normal(0, 0.5, n_samples) > 1.0).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / "healthcare-dataset-stroke-data.csv", index=False)
        logger.info("   ✓ healthcare-dataset-stroke-data.csv created")
    
    def verify_datasets(self):
        """Verify that all datasets are properly formatted"""
        logger.info("🔍 Verifying datasets...")
        
        datasets = ['heart_cleveland_upload.csv', 'diabetes.csv', 'healthcare-dataset-stroke-data.csv', 'insurance.csv']
        
        all_valid = True
        
        for dataset_file in datasets:
            file_path = self.data_dir / dataset_file
            
            if not file_path.exists():
                logger.error(f"   ❌ {dataset_file} not found")
                all_valid = False
                continue
            
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                
                # Get dataset name from filename
                dataset_name = dataset_file.split('_')[0].split('-')[0]
                if 'cleveland' in dataset_file:
                    dataset_name = 'heart'
                elif 'stroke' in dataset_file:
                    dataset_name = 'stroke'
                
                # Check if required columns exist
                if dataset_name in self.required_columns:
                    required_cols = self.required_columns[dataset_name]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        logger.warning(f"   ⚠️ {dataset_file} missing columns: {missing_cols}")
                    else:
                        logger.info(f"   ✓ {dataset_file} ({len(df)} rows, {len(df.columns)} columns)")
                
            except Exception as e:
                logger.error(f"   ❌ Error reading {dataset_file}: {e}")
                all_valid = False
        
        if all_valid:
            logger.info("✅ All datasets verified!")
        else:
            logger.warning("⚠️ Some datasets have issues. The app may still work with available data.")
        
        return all_valid
    
    def create_example_config(self):
        """Create example configuration files"""
        logger.info("📝 Creating example configuration...")
        
        # Create .env example
        env_example = """# Smart Health Risk Analyzer Configuration
# Copy this file to .env and modify as needed

# Development settings
DEBUG=True
LOG_LEVEL=INFO

# Model settings
MODEL_CACHE_SIZE=100
ENABLE_GPU=False

# App settings
APP_PORT=8501
APP_HOST=localhost

# Security settings
SECRET_KEY=your-secret-key-here
ENABLE_AUTH=False
"""
        
        with open(self.base_dir / ".env.example", 'w') as f:
            f.write(env_example)
        
        logger.info("   ✓ .env.example created")
        
        # Create gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# Models and data
models/
*.pkl
*.h5
*.joblib

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
"""
        
        with open(self.base_dir / ".gitignore", 'w') as f:
            f.write(gitignore_content)
        
        logger.info("   ✓ .gitignore created")
        logger.info("✅ Configuration files created!")
    
    def run_quick_test(self):
        """Run a quick test to ensure everything works"""
        logger.info("🧪 Running quick system test...")
        
        try:
            # Test imports
            from data_processor import HealthDataProcessor
            from model_trainer import HealthModelTrainer
            from predictor import HealthRiskPredictor
            from clustering import HealthClustering
            
            logger.info("   ✓ All modules imported successfully")
            
            # Test data loading
            processor = HealthDataProcessor()
            datasets = processor.load_datasets()
            
            if datasets:
                logger.info(f"   ✓ Loaded {len(datasets)} datasets")
            else:
                logger.warning("   ⚠️ No datasets loaded")
            
            logger.info("✅ Quick test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
    
    def display_next_steps(self):
        """Display next steps for the user"""
        logger.info("\n🎉 Setup completed successfully!")
        
        print("\n" + "="*60)
        print("🚀 NEXT STEPS")
        print("="*60)
        print("\n1. Train the models:")
        print("   python train_models.py")
        print("\n2. Run the Streamlit app:")
        print("   streamlit run app.py")
        print("\n3. Open your browser and go to:")
        print("   http://localhost:8501")
        print("\n4. Optional: Customize settings in config.py")
        print("\n" + "="*60)
        print("📚 For more information, see README.md")
        print("🐛 Report issues at: https://github.com/your-repo/issues")
        print("="*60)
    
    def run_setup(self, download_data=True, setup_env=True, verify=True):
        """Run the complete setup process"""
        logger.info("🏥 Smart Health Risk Analyzer Setup")
        logger.info("="*50)
        
        success = True
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Create directories
        self.create_directories()
        
        # Download data if requested
        if download_data:
            self.download_sample_data()
        
        # Setup environment if requested
        if setup_env:
            self.create_example_config()
        
        # Verify datasets if requested
        if verify:
            self.verify_datasets()
        
        # Run quick test
        if not self.run_quick_test():
            success = False
        
        if success:
            self.display_next_steps()
        
        return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Setup Smart Health Risk Analyzer')
    parser.add_argument('--download-data', action='store_true', default=True,
                       help='Download sample datasets')
    parser.add_argument('--setup-env', action='store_true', default=True,
                       help='Create environment configuration files')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify datasets and installation')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data download')
    
    args = parser.parse_args()
    
    # Handle skip-data flag
    if args.skip_data:
        args.download_data = False
    
    # Initialize and run setup
    setup = HealthAnalyzerSetup()
    success = setup.run_setup(
        download_data=args.download_data,
        setup_env=args.setup_env,
        verify=args.verify
    )
    
    if success:
        sys.exit(0)
    else:
        logger.error("❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()