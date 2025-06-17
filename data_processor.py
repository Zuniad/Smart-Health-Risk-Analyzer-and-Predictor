import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class HealthDataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def load_datasets(self):
        """Load all health datasets"""
        datasets = {}
        try:
            datasets['heart'] = pd.read_csv('data/heart_cleveland_upload.csv')
            datasets['diabetes'] = pd.read_csv('data/diabetes.csv')
            datasets['stroke'] = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
            datasets['insurance'] = pd.read_csv('data/insurance.csv')
            print("✅ All datasets loaded successfully")
            return datasets
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return None
    
    def preprocess_heart_data(self, df):
        """Preprocess heart disease dataset based on notebook insights"""
        df = df.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering based on medical knowledge
        df['age_risk'] = pd.cut(df['age'], bins=[0, 45, 60, 100], labels=['Low', 'Medium', 'High'])
        df['chol_risk'] = pd.cut(df['chol'], bins=[0, 200, 240, 1000], labels=['Normal', 'Borderline', 'High'])
        df['bp_risk'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 300], labels=['Normal', 'Elevated', 'High'])
        
        # Encode categorical features
        categorical_cols = ['age_risk', 'chol_risk', 'bp_risk']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        # Select features (remove target)
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 
                       'age_risk', 'chol_risk', 'bp_risk']
        
        X = df[feature_cols]
        y = df['condition']
        
        return X, y
    
    def preprocess_diabetes_data(self, df):
        """Preprocess diabetes dataset with zero value handling"""
        df = df.copy()
        
        # Handle zero values (medical impossibilities)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            df[col] = df[col].replace(0, np.nan)
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        df[zero_cols] = imputer.fit_transform(df[zero_cols])
        
        # Feature engineering
        df['bmi_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['glucose_category'] = pd.cut(df['Glucose'], bins=[0, 100, 125, 300], 
                                       labels=['Normal', 'Prediabetic', 'Diabetic'])
        df['age_category'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], 
                                   labels=['Young', 'Middle', 'Senior'])
        
        # Encode categorical features
        cat_features = ['bmi_category', 'glucose_category', 'age_category']
        for col in cat_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                       'bmi_category', 'glucose_category', 'age_category']
        
        X = df[feature_cols]
        y = df['Outcome']
        
        return X, y
    
    def preprocess_stroke_data(self, df):
        """Preprocess stroke dataset with categorical encoding"""
        df = df.copy()
        
        # Drop ID column
        df = df.drop('id', axis=1)
        
        # Handle missing BMI values
        df['bmi'] = df['bmi'].replace('N/A', np.nan)
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 45, 65, 100], 
                                labels=['Young', 'Adult', 'Middle', 'Senior'])
        df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['glucose_group'] = pd.cut(df['avg_glucose_level'], bins=[0, 90, 160, 300], 
                                    labels=['Low', 'Normal', 'High'])
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 
                           'smoking_status', 'age_group', 'bmi_group', 'glucose_group']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 
                       'smoking_status', 'age_group', 'bmi_group', 'glucose_group']
        
        X = df[feature_cols]
        y = df['stroke']
        
        return X, y
    
    def preprocess_insurance_data(self, df):
        """Preprocess insurance dataset for cost prediction and clustering"""
        df = df.copy()
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], 
                                labels=['Young', 'Middle', 'Senior'])
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['family_size'] = df['children'].apply(lambda x: 'Large' if x >= 3 else 'Small')
        
        # Encode categorical variables
        categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category', 'family_size']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region',
                       'age_group', 'bmi_category', 'family_size']
        
        X = df[feature_cols]
        y = df['charges']
        
        return X, y
    
    def scale_features(self, X_train, X_test, dataset_name):
        """Scale features using StandardScaler"""
        if dataset_name not in self.scalers:
            self.scalers[dataset_name] = StandardScaler()
        
        X_train_scaled = self.scalers[dataset_name].fit_transform(X_train)
        X_test_scaled = self.scalers[dataset_name].transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def balance_data(self, X, y, method='smote'):
        """Balance dataset using SMOTE"""
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            return X_balanced, y_balanced
        return X, y
    
    def prepare_datasets(self):
        """Prepare all datasets for training"""
        datasets = self.load_datasets()
        if datasets is None:
            return None
        
        prepared_data = {}
        
        # Process each dataset
        for name, df in datasets.items():
            print(f"\n📊 Processing {name} dataset...")
            
            if name == 'heart':
                X, y = self.preprocess_heart_data(df)
            elif name == 'diabetes':
                X, y = self.preprocess_diabetes_data(df)
            elif name == 'stroke':
                X, y = self.preprocess_stroke_data(df)
            elif name == 'insurance':
                X, y = self.preprocess_insurance_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if name != 'insurance' else None
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, name)
            
            # Balance data for classification tasks
            if name != 'insurance':
                X_train_balanced, y_train_balanced = self.balance_data(X_train_scaled, y_train)
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            prepared_data[name] = {
                'X_train': X_train_balanced,
                'X_test': X_test_scaled,
                'y_train': y_train_balanced,
                'y_test': y_test,
                'feature_names': X.columns.tolist(),
                'original_X': X,
                'original_y': y
            }
            
            print(f"✅ {name}: {X_train_balanced.shape[0]} train samples, {X_test_scaled.shape[0]} test samples")
        
        return prepared_data
    
    def create_unified_health_profile(self, user_input):
        """Create unified health profile from user input"""
        # Map user input to standardized health metrics
        health_profile = {
            'age': user_input.get('age', 35),
            'gender': 1 if user_input.get('gender') == 'Male' else 0,
            'bmi': user_input.get('bmi', 25.0),
            'glucose': user_input.get('glucose', 100),
            'blood_pressure': user_input.get('blood_pressure', 120),
            'cholesterol': user_input.get('cholesterol', 200),
            'smoking': 1 if user_input.get('smoking') == 'Yes' else 0,
            'exercise': 1 if user_input.get('exercise') == 'Regular' else 0,
            'family_history': 1 if user_input.get('family_history') == 'Yes' else 0
        }
        
        return health_profile