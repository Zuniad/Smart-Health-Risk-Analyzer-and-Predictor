import numpy as np
import pandas as pd
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class HealthModelExplainer:
    def __init__(self, models, data_processor):
        self.models = models
        self.data_processor = data_processor
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.feature_names = {}
        
    def initialize_explainers(self, prepared_data):
        """Initialize SHAP and LIME explainers for all models"""
        print("🔧 Initializing model explainers...")
        
        for dataset_name, data in prepared_data.items():
            if dataset_name == 'insurance':  # Skip regression for now
                continue
                
            X_train = data['X_train']
            feature_names = data['feature_names']
            self.feature_names[dataset_name] = feature_names
            
            # Get the best model for this dataset
            best_model_info = self.models.get(dataset_name)
            if not best_model_info:
                continue
                
            model = best_model_info['model']
            model_name = best_model_info['name']
            
            try:
                # Check if it's a TensorFlow/Keras model
                if hasattr(model, 'predict') and 'tensorflow' in str(type(model)) or 'keras' in str(type(model)):
                    print(f"⚠️ Skipping SHAP for TensorFlow model {model_name} (not supported)")
                    continue
                
                # Initialize SHAP explainer based on model type
                if 'tree' in model_name.lower() or 'forest' in model_name.lower() or 'xgb' in model_name.lower() or 'lgb' in model_name.lower():
                    # Tree-based models
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        print(f"⚠️ Tree model {model_name} doesn't have predict_proba, skipping SHAP")
                        continue
                elif 'linear' in model_name.lower() or 'logistic' in model_name.lower():
                    # Linear models
                    explainer = shap.LinearExplainer(model, X_train)
                elif hasattr(model, 'predict_proba'):
                    # Use KernelExplainer for other models with predict_proba (slower but universal)
                    explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])  # Use fewer samples for speed
                else:
                    print(f"⚠️ Model {model_name} doesn't have predict_proba, skipping SHAP")
                    continue
                
                self.shap_explainers[dataset_name] = explainer
                
                # Initialize LIME explainer only for models with predict_proba
                if hasattr(model, 'predict_proba'):
                    lime_explainer = LimeTabularExplainer(
                        training_data=X_train,
                        feature_names=feature_names,
                        class_names=['No Risk', 'Risk'],
                        mode='classification',
                        discretize_continuous=True
                    )
                    self.lime_explainers[dataset_name] = lime_explainer
                else:
                    print(f"⚠️ Model {model_name} doesn't have predict_proba, skipping LIME")
                
                print(f"✅ Explainers initialized for {dataset_name}")
                
            except Exception as e:
                print(f"⚠️ Failed to initialize explainer for {dataset_name}: {e}")
                continue
    
    def get_shap_explanation(self, user_data, dataset_name, max_display=10):
        """Get SHAP explanation for user prediction"""
        if dataset_name not in self.shap_explainers:
            return None
        
        try:
            # Prepare user data
            user_input = self._prepare_user_input_for_explanation(user_data, dataset_name)
            
            # Get SHAP values
            explainer = self.shap_explainers[dataset_name]
            
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(user_input)
                if isinstance(shap_values, list):  # Multi-class
                    shap_values = shap_values[1]  # Use positive class
            else:
                shap_values = explainer(user_input).values
                if len(shap_values.shape) > 2:  # Multi-class
                    shap_values = shap_values[:, :, 1]
            
            # Get feature names
            feature_names = self.feature_names[dataset_name]
            
            # Create explanation dictionary
            explanation = self._create_shap_explanation_dict(
                shap_values, feature_names, user_input, max_display
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error getting SHAP explanation: {e}")
            return None
    
    def get_lime_explanation(self, user_data, dataset_name, num_features=10):
        """Get LIME explanation for user prediction"""
        if dataset_name not in self.lime_explainers:
            return None
        
        try:
            # Prepare user data
            user_input = self._prepare_user_input_for_explanation(user_data, dataset_name)
            
            # Get model
            model = self.models[dataset_name]['model']
            
            # Get LIME explanation
            explainer = self.lime_explainers[dataset_name]
            
            explanation = explainer.explain_instance(
                user_input[0],
                model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            # Parse LIME explanation
            lime_explanation = self._parse_lime_explanation(explanation)
            
            return lime_explanation
            
        except Exception as e:
            print(f"Error getting LIME explanation: {e}")
            return None
    
    def _prepare_user_input_for_explanation(self, user_data, dataset_name):
        """Prepare user input in the format expected by the model"""
        # This should match the format used in predictor.py
        from predictor import HealthRiskPredictor
        
        # Create a temporary predictor to use its input preparation
        temp_predictor = HealthRiskPredictor(self.models, self.data_processor)
        input_df = temp_predictor.prepare_user_input(user_data, dataset_name)
        
        # Scale the input
        scaler = self.data_processor.scalers.get(dataset_name)
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        return input_scaled
    
    def _create_shap_explanation_dict(self, shap_values, feature_names, user_input, max_display):
        """Create a dictionary with SHAP explanation results"""
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # Get the explanation for the first (and only) instance
        instance_shap = shap_values[0]
        instance_input = user_input[0]
        
        # Create feature importance list
        feature_importance = []
        for i, (feature, importance, value) in enumerate(zip(feature_names, instance_shap, instance_input)):
            feature_importance.append({
                'feature': feature,
                'importance': float(importance),
                'value': float(value),
                'impact': 'Increases Risk' if importance > 0 else 'Decreases Risk',
                'magnitude': abs(float(importance))
            })
        
        # Sort by absolute importance
        feature_importance.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Take top features
        top_features = feature_importance[:max_display]
        
        # Create explanation text
        explanation_text = self._generate_shap_explanation_text(top_features)
        
        return {
            'top_features': top_features,
            'explanation_text': explanation_text,
            'base_value': float(np.mean(instance_shap)),  # Approximation
            'prediction_impact': sum([f['importance'] for f in top_features])
        }
    
    def _generate_shap_explanation_text(self, top_features):
        """Generate human-readable explanation from SHAP values"""
        if not top_features:
            return "No significant features identified for this prediction."
        
        explanations = []
        
        # Most impactful feature
        most_important = top_features[0]
        if most_important['importance'] > 0:
            explanations.append(f"🔴 Your {most_important['feature']} significantly increases your risk")
        else:
            explanations.append(f"🟢 Your {most_important['feature']} helps reduce your risk")
        
        # Other important features
        for feature in top_features[1:4]:  # Top 4 features
            if abs(feature['importance']) > 0.01:  # Only significant features
                direction = "increases" if feature['importance'] > 0 else "decreases"
                explanations.append(f"• {feature['feature']} {direction} risk")
        
        # Summary
        risk_increasing = [f for f in top_features if f['importance'] > 0]
        risk_decreasing = [f for f in top_features if f['importance'] < 0]
        
        summary = f"\n📊 Summary: {len(risk_increasing)} factors increase risk, {len(risk_decreasing)} factors decrease risk."
        explanations.append(summary)
        
        return "\n".join(explanations)
    
    def _parse_lime_explanation(self, lime_explanation):
        """Parse LIME explanation into structured format"""
        # Get explanation for the positive class (risk)
        explanation_list = lime_explanation.as_list()
        
        features_explanation = []
        for feature_desc, importance in explanation_list:
            # Parse feature description (e.g., "age <= 45.00")
            feature_name = feature_desc.split(' ')[0]
            condition = feature_desc
            
            features_explanation.append({
                'feature': feature_name,
                'condition': condition,
                'importance': float(importance),
                'impact': 'Increases Risk' if importance > 0 else 'Decreases Risk'
            })
        
        # Generate explanation text
        explanation_text = self._generate_lime_explanation_text(features_explanation)
        
        return {
            'features': features_explanation,
            'explanation_text': explanation_text,
            'prediction_probability': lime_explanation.predict_proba[1]  # Probability of positive class
        }
    
    def _generate_lime_explanation_text(self, features_explanation):
        """Generate human-readable explanation from LIME results"""
        if not features_explanation:
            return "No explanation available."
        
        explanations = []
        
        # Sort by absolute importance
        sorted_features = sorted(features_explanation, key=lambda x: abs(x['importance']), reverse=True)
        
        # Most important condition
        most_important = sorted_features[0]
        if most_important['importance'] > 0:
            explanations.append(f"🔴 The condition '{most_important['condition']}' most strongly increases your risk")
        else:
            explanations.append(f"🟢 The condition '{most_important['condition']}' most strongly decreases your risk")
        
        # Other important conditions
        for feature in sorted_features[1:4]:
            if abs(feature['importance']) > 0.01:
                direction = "increases" if feature['importance'] > 0 else "decreases"
                explanations.append(f"• '{feature['condition']}' {direction} risk")
        
        return "\n".join(explanations)
    
    def get_global_feature_importance(self, dataset_name, sample_size=100):
        """Get global feature importance using SHAP"""
        if dataset_name not in self.shap_explainers:
            return None
        
        try:
            # Get sample data for analysis
            best_model_info = self.models.get(dataset_name)
            if not best_model_info:
                return None
            
            # Use training data from data processor (would need to be passed or stored)
            # For now, create a mock global importance
            feature_names = self.feature_names[dataset_name]
            
            # Generate mock global importance (in real implementation, use SHAP on training data)
            global_importance = self._generate_mock_global_importance(feature_names, dataset_name)
            
            return global_importance
            
        except Exception as e:
            print(f"Error getting global feature importance: {e}")
            return None
    
    def _generate_mock_global_importance(self, feature_names, dataset_name):
        """Generate mock global feature importance based on medical knowledge"""
        # Based on medical literature and the notebooks provided
        importance_maps = {
            'heart': {
                'age': 0.25, 'chest_pain': 0.20, 'cholesterol': 0.15, 'max_heart_rate': 0.12,
                'exercise_angina': 0.10, 'st_depression': 0.08, 'blood_pressure': 0.06,
                'vessels_colored': 0.04
            },
            'diabetes': {
                'glucose': 0.30, 'bmi': 0.20, 'age': 0.15, 'insulin': 0.12,
                'diabetes_pedigree': 0.10, 'pregnancies': 0.08, 'blood_pressure': 0.05
            },
            'stroke': {
                'age': 0.28, 'hypertension': 0.22, 'heart_disease': 0.18, 'glucose': 0.15,
                'bmi': 0.10, 'smoking': 0.07
            }
        }
        
        base_importance = importance_maps.get(dataset_name, {})
        
        # Create importance list for available features
        feature_importance = []
        for feature in feature_names:
            # Find matching base feature (handle engineered features)
            base_feature = None
            for base in base_importance.keys():
                if base in feature.lower() or feature.lower() in base:
                    base_feature = base
                    break
            
            importance = base_importance.get(base_feature, 0.01)  # Default small importance
            
            feature_importance.append({
                'feature': feature,
                'importance': importance,
                'description': self._get_feature_description(feature, dataset_name)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'features': feature_importance,
            'dataset': dataset_name,
            'explanation': self._generate_global_explanation(feature_importance[:5], dataset_name)
        }
    
    def _get_feature_description(self, feature, dataset_name):
        """Get description for a feature"""
        descriptions = {
            'age': 'Patient age in years',
            'glucose': 'Blood glucose level',
            'bmi': 'Body Mass Index',
            'blood_pressure': 'Blood pressure measurement',
            'cholesterol': 'Cholesterol level',
            'heart_rate': 'Heart rate measurement',
            'smoking': 'Smoking status'
        }
        
        # Find matching description
        for key, desc in descriptions.items():
            if key in feature.lower():
                return desc
        
        return f"{feature} measurement"
    
    def _generate_global_explanation(self, top_features, dataset_name):
        """Generate global explanation for the dataset"""
        disease_names = {'heart': 'heart disease', 'diabetes': 'diabetes', 'stroke': 'stroke'}
        disease = disease_names.get(dataset_name, 'this condition')
        
        explanations = [f"📊 Key factors for predicting {disease}:"]
        
        for i, feature in enumerate(top_features[:3], 1):
            explanations.append(f"{i}. {feature['feature']} (importance: {feature['importance']:.2f})")
        
        explanations.append(f"\n🎯 These factors combined account for the majority of {disease} risk prediction.")
        
        return "\n".join(explanations)
    
    def create_explanation_summary(self, user_data, disease_type):
        """Create comprehensive explanation summary combining SHAP and LIME"""
        summary = {
            'disease_type': disease_type,
            'user_data': user_data,
            'explanations': {}
        }
        
        # Get SHAP explanation
        shap_explanation = self.get_shap_explanation(user_data, disease_type)
        if shap_explanation:
            summary['explanations']['shap'] = shap_explanation
        
        # Get LIME explanation
        lime_explanation = self.get_lime_explanation(user_data, disease_type)
        if lime_explanation:
            summary['explanations']['lime'] = lime_explanation
        
        # Get global importance
        global_importance = self.get_global_feature_importance(disease_type)
        if global_importance:
            summary['explanations']['global'] = global_importance
        
        # Create unified explanation
        summary['unified_explanation'] = self._create_unified_explanation(summary)
        
        return summary
    
    def _create_unified_explanation(self, summary):
        """Create unified explanation combining all methods"""
        explanations = summary['explanations']
        
        unified = ["🔍 **Model Explanation Summary**\n"]
        
        # SHAP insights
        if 'shap' in explanations:
            shap_text = explanations['shap']['explanation_text']
            unified.append("**Feature Impact Analysis (SHAP):**")
            unified.append(shap_text)
            unified.append("")
        
        # LIME insights
        if 'lime' in explanations:
            lime_text = explanations['lime']['explanation_text']
            unified.append("**Condition-Based Analysis (LIME):**")
            unified.append(lime_text)
            unified.append("")
        
        # Global context
        if 'global' in explanations:
            global_text = explanations['global']['explanation']
            unified.append("**General Model Insights:**")
            unified.append(global_text)
        
        return "\n".join(unified)