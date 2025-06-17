import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class HealthRiskPredictor:
    def __init__(self, models, data_processor):
        self.models = models or {}  # Ensure models is not None
        self.data_processor = data_processor
        self.risk_thresholds = {
            'heart': {'low': 0.3, 'moderate': 0.6, 'high': 0.8},
            'diabetes': {'low': 0.25, 'moderate': 0.55, 'high': 0.75},
            'stroke': {'low': 0.2, 'moderate': 0.5, 'high': 0.7}
        }
        
        # Validate that we have models
        if not self.models:
            print("Warning: No models provided to predictor")
        else:
            print(f"Predictor initialized with models: {list(self.models.keys())}")
    
    def prepare_user_input(self, user_data, dataset_type):
        """Prepare user input for prediction"""
        if dataset_type == 'heart':
            # Map user input to heart disease features
            features = {
                'age': user_data.get('age', 50),
                'sex': 1 if user_data.get('gender') == 'Male' else 0,
                'cp': self._map_chest_pain(user_data.get('chest_pain', 'No pain')),
                'trestbps': user_data.get('blood_pressure', 120),
                'chol': user_data.get('cholesterol', 200),
                'fbs': 1 if user_data.get('glucose', 100) > 120 else 0,
                'restecg': self._map_ecg(user_data.get('ecg_results', 'Normal')),
                'thalach': user_data.get('max_heart_rate', 150),
                'exang': 1 if user_data.get('exercise_angina') == 'Yes' else 0,
                'oldpeak': user_data.get('st_depression', 0.0),
                'slope': self._map_slope(user_data.get('st_slope', 'Upsloping')),
                'ca': user_data.get('vessels_colored', 0),
                'thal': self._map_thal(user_data.get('thalassemia', 'Normal'))
            }
            
            # Add engineered features
            age = features['age']
            chol = features['chol']
            bp = features['trestbps']
            
            features['age_risk'] = 0 if age < 45 else (1 if age < 60 else 2)
            features['chol_risk'] = 0 if chol < 200 else (1 if chol < 240 else 2)
            features['bp_risk'] = 0 if bp < 120 else (1 if bp < 140 else 2)
            
        elif dataset_type == 'diabetes':
            features = {
                'Pregnancies': user_data.get('pregnancies', 0),
                'Glucose': user_data.get('glucose', 100),
                'BloodPressure': user_data.get('blood_pressure', 70),
                'SkinThickness': user_data.get('skin_thickness', 20),
                'Insulin': user_data.get('insulin', 80),
                'BMI': user_data.get('bmi', 25.0),
                'DiabetesPedigreeFunction': user_data.get('diabetes_pedigree', 0.5),
                'Age': user_data.get('age', 35)
            }
            
            # Add engineered features
            bmi = features['BMI']
            glucose = features['Glucose']
            age = features['Age']
            
            features['bmi_category'] = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
            features['glucose_category'] = 0 if glucose < 100 else (1 if glucose < 125 else 2)
            features['age_category'] = 0 if age < 30 else (1 if age < 50 else 2)
            
        elif dataset_type == 'stroke':
            features = {
                'gender': 1 if user_data.get('gender') == 'Male' else 0,
                'age': user_data.get('age', 50),
                'hypertension': 1 if user_data.get('hypertension') == 'Yes' else 0,
                'heart_disease': 1 if user_data.get('heart_disease') == 'Yes' else 0,
                'ever_married': 1 if user_data.get('married') == 'Yes' else 0,
                'work_type': self._map_work_type(user_data.get('work_type', 'Private')),
                'Residence_type': 1 if user_data.get('residence') == 'Urban' else 0,
                'avg_glucose_level': user_data.get('glucose', 100),
                'bmi': user_data.get('bmi', 25.0),
                'smoking_status': self._map_smoking(user_data.get('smoking', 'never smoked'))
            }
            
            # Add engineered features
            age = features['age']
            bmi = features['bmi']
            glucose = features['avg_glucose_level']
            
            features['age_group'] = 0 if age < 18 else (1 if age < 45 else (2 if age < 65 else 3))
            features['bmi_group'] = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
            features['glucose_group'] = 0 if glucose < 90 else (1 if glucose < 160 else 2)
        
        return pd.DataFrame([features])
    
    def _map_chest_pain(self, pain_type):
        mapping = {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
        return mapping.get(pain_type, 3)
    
    def _map_ecg(self, ecg_result):
        mapping = {'Normal': 0, 'ST-T abnormality': 1, 'LV hypertrophy': 2}
        return mapping.get(ecg_result, 0)
    
    def _map_slope(self, slope_type):
        mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        return mapping.get(slope_type, 0)
    
    def _map_thal(self, thal_type):
        mapping = {'Normal': 0, 'Fixed defect': 1, 'Reversible defect': 2}
        return mapping.get(thal_type, 0)
    
    def _map_work_type(self, work_type):
        mapping = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}
        return mapping.get(work_type, 3)
    
    def _map_smoking(self, smoking_status):
        mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
        return mapping.get(smoking_status, 1)
    
    def predict_disease_risk(self, user_data, disease_type):
        """Predict disease risk for user"""
        # Prepare input data
        input_df = self.prepare_user_input(user_data, disease_type)
        
        # Get the best model for this disease
        best_model_info = self.models.get(disease_type)
        if not best_model_info:
            return {'error': f'No model available for {disease_type}'}
        
        model = best_model_info['model']
        
        # Scale the input
        scaler = self.data_processor.scalers.get(disease_type)
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                # Scikit-learn model
                prob_array = model.predict_proba(input_scaled)
                risk_probability = float(prob_array[0][1])  # Convert to Python float
                pred_array = model.predict(input_scaled)
                prediction = int(pred_array[0])  # Convert to Python int
            elif hasattr(model, 'predict') and ('tensorflow' in str(type(model)) or 'keras' in str(type(model))):
                # TensorFlow model
                pred_array = model.predict(input_scaled, verbose=0)
                risk_probability = float(pred_array[0][0])  # Convert to Python float
                prediction = 1 if risk_probability > 0.5 else 0
            else:
                # Other models
                pred_array = model.predict(input_scaled)
                prediction_val = float(pred_array[0])  # Convert to Python float
                prediction = int(prediction_val)
                risk_probability = prediction_val if prediction_val <= 1 else 0.5
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
        
        # Ensure risk_probability is between 0 and 1
        risk_probability = max(0.0, min(1.0, risk_probability))
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_probability, disease_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(user_data, disease_type, risk_level)
        
        # Calculate confidence and ensure it's a Python float
        confidence = self._calculate_confidence(risk_probability)
        
        return {
            'risk_probability': float(risk_probability),
            'risk_percentage': float(risk_probability * 100),
            'risk_level': risk_level,
            'prediction': int(prediction),
            'recommendations': recommendations,
            'model_confidence': float(confidence)
        }
    
    def _get_risk_level(self, probability, disease_type):
        """Determine risk level based on probability"""
        thresholds = self.risk_thresholds.get(disease_type, {'low': 0.3, 'moderate': 0.6, 'high': 0.8})
        
        if probability < thresholds['low']:
            return 'Low'
        elif probability < thresholds['moderate']:
            return 'Moderate'
        elif probability < thresholds['high']:
            return 'High'
        else:
            return 'Very High'
    
    def _calculate_confidence(self, probability):
        """Calculate model confidence"""
        # Confidence is higher when probability is closer to 0 or 1
        distance_from_center = abs(float(probability) - 0.5)
        confidence = (distance_from_center * 2) * 100  # Convert to percentage
        return float(min(confidence, 95))  # Cap at 95% and ensure Python float
    
    def _generate_recommendations(self, user_data, disease_type, risk_level):
        """Generate personalized recommendations"""
        recommendations = []
        
        age = user_data.get('age', 50)
        bmi = user_data.get('bmi', 25)
        smoking = user_data.get('smoking', 'never smoked')
        exercise = user_data.get('exercise', 'Sometimes')
        
        # General recommendations
        if risk_level in ['High', 'Very High']:
            recommendations.append("🏥 Consult with a healthcare professional immediately")
            recommendations.append("📊 Schedule comprehensive health screening")
        
        if risk_level in ['Moderate', 'High', 'Very High']:
            recommendations.append("🔄 Regular health monitoring recommended")
        
        # Disease-specific recommendations
        if disease_type == 'heart':
            if user_data.get('cholesterol', 200) > 240:
                recommendations.append("💊 Monitor and manage cholesterol levels")
            if user_data.get('blood_pressure', 120) > 140:
                recommendations.append("🩺 Blood pressure management is crucial")
            if smoking in ['smokes', 'formerly smoked']:
                recommendations.append("🚭 Smoking cessation programs highly recommended")
                
        elif disease_type == 'diabetes':
            if user_data.get('glucose', 100) > 125:
                recommendations.append("🍎 Glucose level monitoring and dietary changes needed")
            if bmi > 30:
                recommendations.append("🏃‍♀️ Weight management through diet and exercise")
            recommendations.append("🥗 Follow a diabetic-friendly diet plan")
            
        elif disease_type == 'stroke':
            if user_data.get('hypertension') == 'Yes':
                recommendations.append("💉 Hypertension management is critical")
            if user_data.get('heart_disease') == 'Yes':
                recommendations.append("❤️ Monitor heart health closely")
            recommendations.append("🧘‍♀️ Stress management techniques")
        
        # Lifestyle recommendations
        if bmi > 30:
            recommendations.append("⚖️ Weight management program recommended")
        if exercise == 'Never':
            recommendations.append("🏋️‍♀️ Start a gentle exercise routine")
        if age > 65:
            recommendations.append("👴 Regular geriatric health check-ups")
        
        # Preventive measures
        recommendations.append("🥬 Adopt a heart-healthy diet")
        recommendations.append("💤 Maintain proper sleep schedule")
        recommendations.append("🧘 Practice stress reduction techniques")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def predict_all_risks(self, user_data):
        """Predict all disease risks for comprehensive health assessment"""
        results = {}
        
        for disease in ['heart', 'diabetes', 'stroke']:
            try:
                prediction = self.predict_disease_risk(user_data, disease)
                results[disease] = prediction
            except Exception as e:
                print(f"Error predicting {disease}: {e}")
                results[disease] = {'error': str(e)}
        
        # Calculate overall health score
        overall_score = self._calculate_overall_health_score(results)
        results['overall_health_score'] = overall_score
        
        return results
    
    def _calculate_overall_health_score(self, disease_results):
        """Calculate overall health score based on all disease risks"""
        total_risk = 0
        valid_predictions = 0
        
        for disease, result in disease_results.items():
            if 'risk_probability' in result:
                total_risk += result['risk_probability']
                valid_predictions += 1
        
        if valid_predictions == 0:
            return {'score': 50, 'level': 'Unknown'}
        
        average_risk = total_risk / valid_predictions
        health_score = max(0, min(100, (1 - average_risk) * 100))
        
        if health_score >= 80:
            level = 'Excellent'
        elif health_score >= 60:
            level = 'Good'
        elif health_score >= 40:
            level = 'Fair'
        else:
            level = 'Poor'
        
        return {
            'score': round(health_score, 1),
            'level': level,
            'average_risk': round(average_risk * 100, 1)
        }
    
    def get_feature_importance_explanation(self, user_data, disease_type):
        """Get explanation of which features contribute most to the prediction"""
        explanations = {
            'heart': {
                'age': 'Age is a significant risk factor for heart disease',
                'cholesterol': 'High cholesterol levels increase cardiovascular risk',
                'blood_pressure': 'Blood pressure directly affects heart health',
                'smoking': 'Smoking significantly increases heart disease risk'
            },
            'diabetes': {
                'glucose': 'Blood glucose level is the primary diabetes indicator',
                'bmi': 'Body weight affects insulin sensitivity',
                'age': 'Diabetes risk increases with age',
                'family_history': 'Genetic factors play a role in diabetes risk'
            },
            'stroke': {
                'age': 'Stroke risk increases significantly with age',
                'hypertension': 'High blood pressure is a major stroke risk factor',
                'heart_disease': 'Heart conditions increase stroke probability',
                'smoking': 'Smoking damages blood vessels and increases stroke risk'
            }
        }
        
        disease_explanations = explanations.get(disease_type, {})
        user_explanations = []
        
        for feature, explanation in disease_explanations.items():
            if feature in user_data or feature.replace('_', '') in user_data:
                user_explanations.append(explanation)
        
        return user_explanations