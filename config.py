"""
Configuration file for Smart Health Risk Analyzer

This file contains all configuration parameters for the application.
Modify these settings to customize the behavior of the system.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "clustering").mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    'heart': {
        'file': 'heart_cleveland_upload.csv',
        'target_column': 'condition',
        'task_type': 'classification',
        'description': 'Heart Disease Prediction (UCI Cleveland Dataset)',
        'features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    },
    'diabetes': {
        'file': 'diabetes.csv',
        'target_column': 'Outcome',
        'task_type': 'classification',
        'description': 'Diabetes Prediction (PIMA Indian Dataset)',
        'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    },
    'stroke': {
        'file': 'healthcare-dataset-stroke-data.csv',
        'target_column': 'stroke',
        'task_type': 'classification',
        'description': 'Stroke Prediction Dataset',
        'features': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    },
    'insurance': {
        'file': 'insurance.csv',
        'target_column': 'charges',
        'task_type': 'regression',
        'description': 'Healthcare Cost Prediction',
        'features': ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    }
}

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'num_leaves': [10, 31, 50, 100],
        'subsample': [0.6, 0.8, 1.0]
    },
    'logistic_regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
}

# Deep learning configurations
DL_CONFIGS = {
    'mlp': {
        'hidden_layers': [64, 32, 16],
        'dropout_rates': [0.3, 0.3, 0.2],
        'activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    },
    'lstm': {
        'lstm_units': [64, 32],
        'dropout_rates': [0.3, 0.3],
        'dense_units': [16],
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
        'sequence_length': 10
    }
}

# Clustering configurations
CLUSTERING_CONFIGS = {
    'kmeans': {
        'n_clusters_range': (2, 15),
        'random_state': 42,
        'n_init': 10
    },
    'gmm': {
        'n_components_range': (2, 15),
        'covariance_type': 'full',
        'random_state': 42
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    }
}

# Anomaly detection configurations
ANOMALY_CONFIGS = {
    'isolation_forest': {
        'contamination': 0.1,
        'random_state': 42,
        'n_estimators': 100
    },
    'lof': {
        'contamination': 0.1,
        'n_neighbors': 20
    }
}

# Risk threshold configurations
RISK_THRESHOLDS = {
    'heart': {'low': 0.3, 'moderate': 0.6, 'high': 0.8},
    'diabetes': {'low': 0.25, 'moderate': 0.55, 'high': 0.75},
    'stroke': {'low': 0.2, 'moderate': 0.5, 'high': 0.7}
}

# Feature engineering configurations
FEATURE_ENGINEERING = {
    'age_bins': {
        'bins': [0, 18, 30, 45, 60, 80, 120],
        'labels': ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior', 'Elderly']
    },
    'bmi_bins': {
        'bins': [0, 18.5, 25, 30, 40, 100],
        'labels': ['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese']
    },
    'glucose_bins': {
        'bins': [0, 90, 100, 125, 160, 300],
        'labels': ['Low', 'Normal', 'Pre-diabetic', 'Diabetic', 'Very High']
    },
    'cholesterol_bins': {
        'bins': [0, 200, 240, 300, 500],
        'labels': ['Normal', 'Borderline High', 'High', 'Very High']
    },
    'bp_bins': {
        'bins': [0, 120, 130, 140, 180, 300],
        'labels': ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis']
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring_metric': 'f1_weighted',
    'optuna_trials': 50,
    'early_stopping_rounds': 10
}

# Streamlit app configurations
APP_CONFIGS = {
    'page_title': 'Smart Health Risk Analyzer',
    'page_icon': '🏥',
    'layout': 'wide',
    'sidebar_width': 350,
    'chart_height': 400,
    'colors': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }
}

# Logging configurations
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_format': '%(asctime)s - %(levelname)s - %(message)s',
    'console_format': '%(levelname)s - %(message)s',
    'log_file_prefix': 'health_analyzer'
}

# Data validation configurations
VALIDATION_CONFIGS = {
    'age': {'min': 1, 'max': 120, 'type': 'int'},
    'bmi': {'min': 10.0, 'max': 80.0, 'type': 'float'},
    'blood_pressure': {'min': 60, 'max': 250, 'type': 'int'},
    'cholesterol': {'min': 100, 'max': 500, 'type': 'int'},
    'glucose': {'min': 50, 'max': 400, 'type': 'int'},
    'heart_rate': {'min': 40, 'max': 200, 'type': 'int'},
    'height': {'min': 100, 'max': 250, 'type': 'int'},
    'pregnancies': {'min': 0, 'max': 20, 'type': 'int'},
    'insulin': {'min': 0, 'max': 1000, 'type': 'int'},
    'skin_thickness': {'min': 0, 'max': 100, 'type': 'int'}
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'classification': {
        'accuracy': 0.7,
        'precision': 0.7,
        'recall': 0.7,
        'f1': 0.7,
        'auc': 0.7
    },
    'regression': {
        'r2': 0.6,
        'mae': 1000,  # Maximum allowed MAE for insurance prediction
        'rmse': 2000   # Maximum allowed RMSE for insurance prediction
    }
}

# Explanation configurations
EXPLANATION_CONFIGS = {
    'shap': {
        'max_display': 10,
        'sample_size': 100,
        'feature_perturbation': 'interventional'
    },
    'lime': {
        'num_features': 10,
        'num_samples': 5000,
        'mode': 'classification'
    }
}

# Health recommendations templates
RECOMMENDATION_TEMPLATES = {
    'age_based': {
        'young': ['Regular exercise routine', 'Healthy diet establishment', 'Preventive check-ups'],
        'middle': ['Stress management', 'Regular health monitoring', 'Weight management'],
        'senior': ['Fall prevention', 'Medication management', 'Regular specialist visits']
    },
    'risk_based': {
        'low': ['Maintain current lifestyle', 'Annual check-ups', 'Preventive care'],
        'moderate': ['Lifestyle modifications', 'Semi-annual check-ups', 'Risk factor monitoring'],
        'high': ['Immediate medical consultation', 'Intensive monitoring', 'Medication compliance'],
        'very_high': ['Emergency medical attention', 'Specialist referral', 'Intensive treatment']
    },
    'condition_specific': {
        'heart': {
            'diet': 'Heart-healthy diet (low sodium, high fiber)',
            'exercise': 'Cardiovascular exercise (150 min/week)',
            'lifestyle': 'Stress reduction and smoking cessation'
        },
        'diabetes': {
            'diet': 'Diabetic diet (low glycemic index foods)',
            'exercise': 'Regular physical activity and weight management',
            'lifestyle': 'Blood glucose monitoring and medication adherence'
        },
        'stroke': {
            'diet': 'DASH diet (low sodium, high potassium)',
            'exercise': 'Regular moderate exercise',
            'lifestyle': 'Blood pressure control and anticoagulant therapy if prescribed'
        }
    }
}

# Export configurations for easy access
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR',
    'DATASETS', 'MODEL_CONFIGS', 'DL_CONFIGS', 'CLUSTERING_CONFIGS',
    'ANOMALY_CONFIGS', 'RISK_THRESHOLDS', 'FEATURE_ENGINEERING',
    'TRAINING_CONFIGS', 'APP_CONFIGS', 'LOGGING_CONFIG',
    'VALIDATION_CONFIGS', 'PERFORMANCE_THRESHOLDS', 'EXPLANATION_CONFIGS',
    'RECOMMENDATION_TEMPLATES'
]