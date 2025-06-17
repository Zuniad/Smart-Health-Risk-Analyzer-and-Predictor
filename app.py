import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import HealthDataProcessor
from model_trainer import HealthModelTrainer
from predictor import HealthRiskPredictor
from clustering import HealthClustering
from explainer import HealthModelExplainer
from utils import (
    create_risk_gauge, create_feature_importance_chart, create_cluster_visualization,
    create_health_dashboard, generate_health_report, validate_user_input,
    format_risk_level_color, create_health_timeline, calculate_health_improvement_potential,
    create_improvement_potential_chart
)

# Page configuration
st.set_page_config(
    page_title="Smart Health Risk Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4f4f4f;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HealthApp:
    def __init__(self):
        self.data_processor = None
        self.model_trainer = None
        self.predictor = None
        self.clustering = None
        self.explainer = None
        self.models = None
        self.prepared_data = None
        
        # Initialize session state
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None
        if 'models' not in st.session_state:
            st.session_state.models = None
    
    def load_models(self):
        """Load or train models"""
        if st.session_state.models_loaded and st.session_state.predictor is not None:
            # Models already loaded and predictor exists in session state
            self.predictor = st.session_state.predictor
            self.models = st.session_state.models
            return True
        
        with st.spinner("🔄 Loading AI models and data..."):
            try:
                # Initialize components
                self.data_processor = HealthDataProcessor()
                self.model_trainer = HealthModelTrainer()
                self.clustering = HealthClustering()
                
                # Check if models exist
                if os.path.exists('models/') and len(os.listdir('models/')) > 0:
                    st.info("📁 Loading pre-trained models...")
                    
                    # Load models
                    models_loaded = self.model_trainer.load_models()
                    clustering_loaded = self.clustering.load_clustering_models()
                    
                    if not models_loaded:
                        st.warning("🔧 Could not load models, training new ones...")
                        return self.train_new_models()
                    
                    # Load prepared data
                    self.prepared_data = self.data_processor.prepare_datasets()
                    
                    if not self.prepared_data:
                        st.error("❌ Failed to prepare datasets")
                        return False
                    
                    # Get best models with validation
                    self.models = self.model_trainer.get_best_models()
                    
                    # Validate that we have required models
                    required_diseases = ['heart', 'diabetes', 'stroke']
                    missing_models = [disease for disease in required_diseases if disease not in self.models]
                    
                    if missing_models:
                        st.warning(f"⚠️ Missing models for: {missing_models}")
                        st.info("🔧 Training missing models...")
                        
                        # Train missing models
                        for disease in missing_models:
                            if disease in self.prepared_data:
                                st.info(f"Training {disease} model...")
                                self._train_single_model(disease, self.prepared_data[disease])
                        
                        # Refresh best models
                        self.models = self.model_trainer.get_best_models()
                    
                    if not self.models:
                        st.error("❌ No valid models found after loading")
                        return False
                    
                    st.success(f"✅ Loaded models for: {list(self.models.keys())}")
                    
                else:
                    st.warning("🔧 Training new models...")
                    return self.train_new_models()
                
                # Initialize predictor with validation
                try:
                    self.predictor = HealthRiskPredictor(self.models, self.data_processor)
                    if self.predictor is None:
                        st.error("❌ Failed to initialize predictor")
                        return False
                    
                    # Store in session state
                    st.session_state.predictor = self.predictor
                    st.session_state.models = self.models
                    
                    st.success("✅ Predictor initialized successfully")
                except Exception as e:
                    st.error(f"❌ Predictor initialization failed: {e}")
                    return False
                
                # Initialize explainer with error handling
                try:
                    self.explainer = HealthModelExplainer(self.models, self.data_processor)
                    if self.prepared_data:
                        self.explainer.initialize_explainers(self.prepared_data)
                    st.success("✅ Explainer initialized successfully")
                except Exception as e:
                    st.warning(f"⚠️ Could not initialize explainers: {e}")
                    self.explainer = None  # Set to None so we can handle it later
                
                st.session_state.models_loaded = True
                return True
                
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.error("Please ensure the data folder exists with the required CSV files.")
                
                # Show debug information
                st.error("Debug information:")
                st.error(f"Current working directory: {os.getcwd()}")
                st.error(f"Models directory exists: {os.path.exists('models/')}")
                if os.path.exists('models/'):
                    model_files = os.listdir('models/')
                    st.error(f"Model files: {model_files}")
                
                return False
    
    def _train_single_model(self, disease, data):
        """Train a single model for a specific disease"""
        try:
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            
            # Use the most reliable model for each disease
            if disease == 'heart':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42)
            elif disease == 'diabetes':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif disease == 'stroke':
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            else:
                return False
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate
            scores = self.model_trainer.evaluate_model(model, X_test, y_test, 'classification')
            
            # Store the model
            model_name = f"{disease}_emergency_model"
            self.model_trainer.models[model_name] = model
            self.model_trainer.model_scores[model_name] = scores
            
            return True
            
        except Exception as e:
            st.error(f"Failed to train {disease} model: {e}")
            return False
    
    def train_new_models(self):
        """Train new models"""
        # Prepare data
        self.prepared_data = self.data_processor.prepare_datasets()
        
        if self.prepared_data is None:
            st.error("Failed to load datasets. Please check data folder.")
            return
        
        # Train models
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training machine learning models...")
        progress_bar.progress(25)
        
        self.models = self.model_trainer.train_all_models(self.prepared_data)
        progress_bar.progress(75)
        
        status_text.text("Training clustering models...")
        clustering_data, _ = self.clustering.prepare_clustering_data(self.prepared_data)
        self.clustering.perform_clustering(clustering_data)
        self.clustering.perform_dimensionality_reduction(clustering_data)
        self.clustering.detect_anomalies(clustering_data)
        progress_bar.progress(90)
        
        # Save models
        status_text.text("Saving models...")
        self.model_trainer.save_models()
        self.clustering.save_clustering_models()
        progress_bar.progress(100)
        
        status_text.text("Training completed!")
        
        # Initialize predictor after training
        if self.models:
            self.predictor = HealthRiskPredictor(self.models, self.data_processor)
            st.session_state.predictor = self.predictor
            st.session_state.models = self.models
            st.session_state.models_loaded = True
    
    def render_sidebar(self):
        """Render sidebar with user inputs"""
        st.sidebar.markdown("## 👤 Patient Information")
        
        # Basic Information
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=35, key="age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1, key="bmi")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="height")
        
        st.sidebar.markdown("### 🩺 Health Metrics")
        
        # Health metrics
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=60, max_value=250, value=120, key="bp")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=400, value=100, key="glucose")
            
        with col2:
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=200, key="cholesterol")
            heart_rate = st.number_input("Resting Heart Rate", min_value=40, max_value=200, value=70, key="heart_rate")
        
        st.sidebar.markdown("### 🏥 Medical History")
        
        # Medical history
        hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"], key="hypertension")
        heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"], key="heart_disease")
        diabetes_family = st.sidebar.selectbox("Family History of Diabetes", ["No", "Yes"], key="diabetes_family")
        
        st.sidebar.markdown("### 🚭 Lifestyle Factors")
        
        # Lifestyle
        smoking = st.sidebar.selectbox("Smoking Status", 
                                     ["never smoked", "formerly smoked", "smokes"], key="smoking")
        exercise = st.sidebar.selectbox("Exercise Frequency", 
                                      ["Never", "Rarely", "Sometimes", "Regular", "Daily"], key="exercise")
        married = st.sidebar.selectbox("Marital Status", ["No", "Yes"], key="married")
        work_type = st.sidebar.selectbox("Work Type", 
                                       ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], key="work")
        
        # Additional inputs for specific conditions
        st.sidebar.markdown("### 🔬 Advanced Metrics (Optional)")
        
        with st.sidebar.expander("Heart Disease Specific"):
            chest_pain = st.selectbox("Chest Pain Type", 
                                    ["No pain", "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        with st.sidebar.expander("Diabetes Specific"):
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        
        # Compile user data
        user_data = {
            'name': 'Current User',
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'height': height,
            'blood_pressure': blood_pressure,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'heart_rate': heart_rate,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'diabetes_family': diabetes_family,
            'smoking': smoking,
            'exercise': exercise,
            'married': married,
            'work_type': work_type,
            'chest_pain': chest_pain,
            'exercise_angina': exercise_angina,
            'st_depression': st_depression,
            'pregnancies': pregnancies,
            'insulin': insulin,
            'skin_thickness': skin_thickness,
            'diabetes_pedigree': diabetes_pedigree
        }
        
        return user_data
    
    # def render_main_dashboard(self, user_data):
    #     """Render main dashboard"""
    #     st.markdown('<div class="main-header">🏥 Smart Health Risk Analyzer</div>', unsafe_allow_html=True)
    #     st.markdown('<div class="sub-header">AI-Powered Comprehensive Health Risk Assessment</div>', unsafe_allow_html=True)
        
    #     # Validate input
    #     validation_errors = validate_user_input(user_data)
    #     if validation_errors:
    #         st.sidebar.error("⚠️ Input Validation Errors:")
    #         for error in validation_errors:
    #             st.sidebar.error(f"• {error}")
    #         return
        
    #     # Check if predictor is available in session state
    #     if 'predictor' not in st.session_state or st.session_state.predictor is None:
    #         st.error("❌ Predictor not initialized. Please check model loading.")
    #         st.info("💡 Try reloading the page or retraining models.")
    #         return
        
    #     # Check if we have required models in session state
    #     if 'models' not in st.session_state or not st.session_state.models:
    #         st.error("❌ No models available for prediction.")
    #         st.info("💡 Please train models first or check model loading.")
    #         return
        
    #     # Use session state predictor and models
    #     predictor = st.session_state.predictor
    #     models = st.session_state.models
        
    #     # Display available models
    #     available_models = list(models.keys())
    #     st.info(f"📊 Available models: {', '.join(available_models)}")
        
    #     # Analysis button
    #     if st.button("🔍 Analyze Health Risks", type="primary", use_container_width=True):
    #         with st.spinner("🧠 AI is analyzing your health profile..."):
    #             try:
    #                 # Validate predictor one more time
    #                 if predictor is None:
    #                     st.error("❌ Predictor is not available")
    #                     return
                    
    #                 # Get predictions with error handling
    #                 predictions = predictor.predict_all_risks(user_data)
                    
    #                 if not predictions:
    #                     st.error("❌ No predictions returned")
    #                     return
                    
    #                 st.session_state.predictions = predictions
    #                 st.session_state.user_data = user_data
                    
    #                 # Get cluster assignment with error handling
    #                 try:
    #                     if hasattr(self, 'clustering') and self.clustering is not None:
    #                         cluster_info = self.clustering.assign_user_to_cluster(user_data)
    #                         st.session_state.cluster_info = cluster_info
    #                     else:
    #                         st.session_state.cluster_info = None
    #                 except Exception as e:
    #                     st.warning(f"⚠️ Clustering analysis failed: {e}")
    #                     st.session_state.cluster_info = None
                    
    #                 # Get anomaly detection with error handling
    #                 try:
    #                     if hasattr(self, 'clustering') and self.clustering is not None:
    #                         anomaly_info = self.clustering.detect_user_anomalies(user_data)
    #                         st.session_state.anomaly_info = anomaly_info
    #                     else:
    #                         st.session_state.anomaly_info = None
    #                 except Exception as e:
    #                     st.warning(f"⚠️ Anomaly detection failed: {e}")
    #                     st.session_state.anomaly_info = None
                    
    #                 # Get explanations with error handling
    #                 explanations = {}
    #                 if hasattr(self, 'explainer') and self.explainer is not None:
    #                     for disease in ['heart', 'diabetes', 'stroke']:
    #                         if disease in models:
    #                             try:
    #                                 exp = self.explainer.create_explanation_summary(user_data, disease)
    #                                 if exp:
    #                                     explanations[disease] = exp
    #                             except Exception as e:
    #                                 st.warning(f"⚠️ Explanation failed for {disease}: {e}")
                    
    #                 st.session_state.explanations = explanations
                    
    #                 st.success("✅ Analysis completed!")
                    
    #             except Exception as e:
    #                 st.error(f"❌ Analysis failed: {str(e)}")
                    
    #                 # Debug information
    #                 st.error("🔍 Debug Information:")
    #                 st.error(f"Predictor type: {type(predictor)}")
    #                 st.error(f"Models available: {list(models.keys()) if models else 'None'}")
    #                 st.error(f"User data keys: {list(user_data.keys())}")
                    
    #                 return
        
    #     # Display results if available
    #     if st.session_state.predictions:
    #         self.display_results()
    
    # def display_results(self):
    #     """Display analysis results"""
    #     predictions = st.session_state.predictions
    #     user_data = st.session_state.user_data
        
    #     # Overall Health Dashboard
    #     st.markdown("## 📊 Health Risk Dashboard")
        
    #     # Create health dashboard
    #     dashboard_fig = create_health_dashboard(predictions)
    #     if dashboard_fig:
    #         st.plotly_chart(dashboard_fig, use_container_width=True)
        
    #     # Individual Risk Analysis
    #     col1, col2, col3 = st.columns(3)
        
    #     diseases = ['heart', 'diabetes', 'stroke']
    #     disease_names = ['Heart Disease', 'Diabetes', 'Stroke']
        
    #     for i, (disease, name) in enumerate(zip(diseases, disease_names)):
    #         with [col1, col2, col3][i]:
    #             if disease in predictions and 'risk_percentage' in predictions[disease]:
    #                 result = predictions[disease]
    #                 risk_pct = result['risk_percentage']
    #                 risk_level = result['risk_level']
                    
    #                 # Risk gauge
    #                 gauge_fig = create_risk_gauge(risk_pct, f"{name} Risk")
    #                 st.plotly_chart(gauge_fig, use_container_width=True)
                    
    #                 # Risk level with color
    #                 color = format_risk_level_color(risk_level)
    #                 st.markdown(f"""
    #                 <div class="metric-container">
    #                     <h4>{name}</h4>
    #                     <p style="color: {color}; font-weight: bold; font-size: 1.2em;">
    #                         {risk_level} Risk ({risk_pct:.1f}%)
    #                     </p>
    #                     <p>Confidence: {result.get('model_confidence', 0):.1f}%</p>
    #                 </div>
    #                 """, unsafe_allow_html=True)
        
    #     # Detailed Analysis Tabs
    #     tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Recommendations", "🔍 AI Explanations", "👥 Patient Clustering", "⚠️ Anomaly Detection", "📈 Improvement Potential"])
        
    #     with tab1:
    #         self.display_recommendations(predictions)
        
    #     with tab2:
    #         self.display_explanations()
        
    #     with tab3:
    #         self.display_clustering_analysis()
        
    #     with tab4:
    #         self.display_anomaly_analysis()
        
    #     with tab5:
    #         self.display_improvement_potential()
        
    #     # Health Report
    #     st.markdown("## 📄 Generate Health Report")
    #     if st.button("📋 Generate Comprehensive Report", type="secondary"):
    #         self.generate_and_display_report()
    
    # def display_recommendations(self, predictions):
    #     """Display personalized recommendations"""
    #     st.markdown("### 💡 Personalized Health Recommendations")
        
    #     for disease, results in predictions.items():
    #         if disease == 'overall_health_score' or 'recommendations' not in results:
    #             continue
            
    #         risk_level = results.get('risk_level', 'Unknown')
    #         recommendations = results['recommendations']
            
    #         # Color-coded header
    #         color = format_risk_level_color(risk_level)
            
    #         st.markdown(f"""
    #         <div class="recommendation-box">
    #             <h4 style="color: {color};">{disease.title()} Recommendations ({risk_level} Risk)</h4>
    #         </div>
    #         """, unsafe_allow_html=True)
            
    #         for rec in recommendations:
    #             st.markdown(f"• {rec}")
            
    #         st.markdown("---")
    
    def display_results(self):
        """Display analysis results"""
        predictions = st.session_state.predictions
        user_data = st.session_state.user_data
        
        # Overall Health Dashboard
        st.markdown("## 📊 Health Risk Dashboard")
        
        # Create health dashboard
        try:
            dashboard_fig = create_health_dashboard(predictions)
            if dashboard_fig:
                st.plotly_chart(dashboard_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create dashboard visualization: {e}")
        
        # Individual Risk Analysis
        col1, col2, col3 = st.columns(3)
        
        diseases = ['heart', 'diabetes', 'stroke']
        disease_names = ['Heart Disease', 'Diabetes', 'Stroke']
        
        for i, (disease, name) in enumerate(zip(diseases, disease_names)):
            with [col1, col2, col3][i]:
                if disease in predictions and 'risk_percentage' in predictions[disease]:
                    result = predictions[disease]
                    
                    # Check for errors first
                    if 'error' in result:
                        st.error(f"❌ {name}: {result['error']}")
                        continue
                    
                    # Safely extract values and ensure they're Python scalars
                    try:
                        risk_pct = float(result.get('risk_percentage', 0))
                        risk_level = str(result.get('risk_level', 'Unknown'))
                        confidence = float(result.get('model_confidence', 0))
                        
                        # Risk gauge
                        try:
                            gauge_fig = create_risk_gauge(risk_pct, f"{name} Risk")
                            if gauge_fig:
                                st.plotly_chart(gauge_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create gauge for {name}: {e}")
                        
                        # Risk level with color
                        try:
                            color = format_risk_level_color(risk_level)
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>{name}</h4>
                                <p style="color: {color}; font-weight: bold; font-size: 1.2em;">
                                    {risk_level} Risk ({risk_pct:.1f}%)
                                </p>
                                <p>Confidence: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            # Fallback display
                            st.markdown(f"""
                            **{name}**  
                            Risk Level: {risk_level}  
                            Risk Percentage: {risk_pct:.1f}%  
                            Confidence: {confidence:.1f}%
                            """)
                            st.warning(f"Display error for {name}: {e}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {name} results: {e}")
                        st.write("Raw result:", result)
                else:
                    st.warning(f"⚠️ No valid prediction for {name}")
        
        # Detailed Analysis Tabs
        try:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Recommendations", "🔍 AI Explanations", "👥 Patient Clustering", "⚠️ Anomaly Detection", "📈 Improvement Potential"])
            
            with tab1:
                self.display_recommendations(predictions)
            
            with tab2:
                self.display_explanations()
            
            with tab3:
                self.display_clustering_analysis()
            
            with tab4:
                self.display_anomaly_analysis()
            
            with tab5:
                self.display_improvement_potential()
                
        except Exception as e:
            st.error(f"Error creating tabs: {e}")
            # Fallback: display recommendations only
            st.markdown("## 📋 Recommendations")
            self.display_recommendations(predictions)
        
        # Health Report
        st.markdown("## 📄 Generate Health Report")
        if st.button("📋 Generate Comprehensive Report", type="secondary"):
            try:
                self.generate_and_display_report()
            except Exception as e:
                st.error(f"Error generating report: {e}")
    
    def display_recommendations(self, predictions):
        """Display personalized recommendations"""
        st.markdown("### 💡 Personalized Health Recommendations")
        
        for disease, results in predictions.items():
            if disease == 'overall_health_score':
                continue
                
            # Check for errors
            if 'error' in results:
                st.error(f"❌ {disease.title()}: {results['error']}")
                continue
                
            if 'recommendations' not in results:
                continue
            
            try:
                risk_level = str(results.get('risk_level', 'Unknown'))
                recommendations = results['recommendations']
                
                # Color-coded header
                try:
                    color = format_risk_level_color(risk_level)
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4 style="color: {color};">{disease.title()} Recommendations ({risk_level} Risk)</h4>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    # Fallback
                    st.markdown(f"#### {disease.title()} Recommendations ({risk_level} Risk)")
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error displaying recommendations for {disease}: {e}")
    
    def render_main_dashboard(self, user_data):
        """Render main dashboard"""
        st.markdown('<div class="main-header">🏥 Smart Health Risk Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Comprehensive Health Risk Assessment</div>', unsafe_allow_html=True)
        
        # Validate input
        try:
            validation_errors = validate_user_input(user_data)
            if validation_errors:
                st.sidebar.error("⚠️ Input Validation Errors:")
                for error in validation_errors:
                    st.sidebar.error(f"• {error}")
                return
        except Exception as e:
            st.warning(f"Input validation failed: {e}")
        
        # Check if predictor is available in session state
        if 'predictor' not in st.session_state or st.session_state.predictor is None:
            st.error("❌ Predictor not initialized. Please check model loading.")
            st.info("💡 Try reloading the page or retraining models.")
            return
        
        # Check if we have required models in session state
        if 'models' not in st.session_state or not st.session_state.models:
            st.error("❌ No models available for prediction.")
            st.info("💡 Please train models first or check model loading.")
            return
        
        # Use session state predictor and models
        predictor = st.session_state.predictor
        models = st.session_state.models
        
        # Display available models
        available_models = list(models.keys())
        st.info(f"📊 Available models: {', '.join(available_models)}")
        
        # Analysis button
        if st.button("🔍 Analyze Health Risks", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is analyzing your health profile..."):
                try:
                    # Validate predictor one more time
                    if predictor is None:
                        st.error("❌ Predictor is not available")
                        return
                    
                    # Get predictions with error handling
                    predictions = predictor.predict_all_risks(user_data)
                    
                    if not predictions:
                        st.error("❌ No predictions returned")
                        return
                    
                    # Check if any predictions are valid
                    valid_predictions = any('risk_percentage' in pred for pred in predictions.values() 
                                          if isinstance(pred, dict) and 'error' not in pred)
                    
                    if not valid_predictions:
                        st.error("❌ No valid predictions generated")
                        st.write("Prediction results:", predictions)
                        return
                    
                    st.session_state.predictions = predictions
                    st.session_state.user_data = user_data
                    
                    # Get cluster assignment with error handling
                    try:
                        if hasattr(self, 'clustering') and self.clustering is not None:
                            cluster_info = self.clustering.assign_user_to_cluster(user_data)
                            st.session_state.cluster_info = cluster_info
                        else:
                            st.session_state.cluster_info = None
                    except Exception as e:
                        st.warning(f"⚠️ Clustering analysis failed: {e}")
                        st.session_state.cluster_info = None
                    
                    # Get anomaly detection with error handling
                    try:
                        if hasattr(self, 'clustering') and self.clustering is not None:
                            anomaly_info = self.clustering.detect_user_anomalies(user_data)
                            st.session_state.anomaly_info = anomaly_info
                        else:
                            st.session_state.anomaly_info = None
                    except Exception as e:
                        st.warning(f"⚠️ Anomaly detection failed: {e}")
                        st.session_state.anomaly_info = None
                    
                    # Get explanations with error handling
                    explanations = {}
                    if hasattr(self, 'explainer') and self.explainer is not None:
                        for disease in ['heart', 'diabetes', 'stroke']:
                            if disease in models:
                                try:
                                    exp = self.explainer.create_explanation_summary(user_data, disease)
                                    if exp:
                                        explanations[disease] = exp
                                except Exception as e:
                                    st.warning(f"⚠️ Explanation failed for {disease}: {e}")
                    
                    st.session_state.explanations = explanations
                    
                    st.success("✅ Analysis completed!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        st.error(f"Predictor type: {type(predictor)}")
                        st.error(f"Models available: {list(models.keys()) if models else 'None'}")
                        st.error(f"User data keys: {list(user_data.keys())}")
                        st.error(f"Error details: {str(e)}")
                    
                    return
        
        # Display results if available
        if st.session_state.predictions:
            self.display_results()
    # def display_explanations(self):
    #     """Display AI model explanations"""
    #     st.markdown("### 🤖 AI Model Explanations")
        
    #     if 'explanations' not in st.session_state:
    #         st.info("Run analysis to see AI explanations")
    #         return
        
    #     explanations = st.session_state.explanations
        
    #     for disease, exp_data in explanations.items():
    #         with st.expander(f"{disease.title()} Model Explanation"):
    #             if 'unified_explanation' in exp_data:
    #                 st.markdown(exp_data['unified_explanation'])
                
    #             # Feature importance chart
    #             if 'explanations' in exp_data and 'shap' in exp_data['explanations']:
    #                 shap_data = exp_data['explanations']['shap']
    #                 if 'top_features' in shap_data:
    #                     importance_fig = create_feature_importance_chart(
    #                         shap_data['top_features'], 
    #                         f"{disease.title()} Feature Importance"
    #                     )
    #                     if importance_fig:
    #                         st.plotly_chart(importance_fig, use_container_width=True)
    
    # def display_clustering_analysis(self):
    #     """Display clustering analysis"""
    #     st.markdown("### 👥 Patient Clustering Analysis")
        
    #     if 'cluster_info' not in st.session_state:
    #         st.info("Run analysis to see clustering results")
    #         return
        
    #     cluster_info = st.session_state.cluster_info
        
    #     if cluster_info and 'cluster_profile' in cluster_info:
    #         profile = cluster_info['cluster_profile']
            
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             st.markdown("#### Your Patient Group")
    #             st.markdown(f"**Cluster ID:** {cluster_info['cluster_id']}")
    #             st.markdown(f"**Similarity Score:** {cluster_info['similarity_score']:.2f}")
                
    #             if 'description' in profile:
    #                 st.markdown(f"**Group Description:** {profile['description']}")
                
    #             if 'risk_profile' in profile:
    #                 risk_profile = profile['risk_profile']
    #                 overall_risk = risk_profile.get('overall', 'Unknown')
    #                 color = format_risk_level_color(overall_risk)
                    
    #                 st.markdown(f"""
    #                 <div class="metric-container">
    #                     <h5>Group Risk Profile</h5>
    #                     <p style="color: {color}; font-weight: bold;">Overall Risk: {overall_risk}</p>
    #                     <p>Age Risk: {risk_profile.get('age_risk', 'N/A')}</p>
    #                     <p>BP Risk: {risk_profile.get('bp_risk', 'N/A')}</p>
    #                     <p>Cholesterol Risk: {risk_profile.get('cholesterol_risk', 'N/A')}</p>
    #                     <p>BMI Risk: {risk_profile.get('bmi_risk', 'N/A')}</p>
    #                 </div>
    #                 """, unsafe_allow_html=True)
            
    #         with col2:
    #             # Cluster visualization
    #             if hasattr(self.clustering, 'dimensionality_reducers'):
    #                 cluster_viz = create_cluster_visualization(
    #                     self.clustering.dimensionality_reducers,
    #                     cluster_info['cluster_id']
    #                 )
    #                 if cluster_viz:
    #                     st.plotly_chart(cluster_viz, use_container_width=True)
    
    # def display_anomaly_analysis(self):
    #     """Display anomaly detection results"""
    #     st.markdown("### ⚠️ Anomaly Detection Analysis")
        
    #     if 'anomaly_info' not in st.session_state:
    #         st.info("Run analysis to see anomaly detection results")
    #         return
        
    #     anomaly_info = st.session_state.anomaly_info
        
    #     is_anomaly = anomaly_info.get('is_anomaly', False)
    #     confidence = anomaly_info.get('confidence', 0)
    #     interpretation = anomaly_info.get('interpretation', '')
        
    #     if is_anomaly:
    #         st.markdown(f"""
    #         <div class="warning-box">
    #             <h4>⚠️ Anomaly Detected</h4>
    #             <p><strong>Confidence:</strong> {confidence:.2f}</p>
    #             <p>{interpretation}</p>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         st.markdown(f"""
    #         <div class="success-box">
    #             <h4>✅ Normal Health Profile</h4>
    #             <p>Your health profile appears normal compared to the population.</p>
    #             <p><strong>Confidence:</strong> {confidence:.2f}</p>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # def display_improvement_potential(self):
    #     """Display health improvement potential"""
    #     st.markdown("### 📈 Health Improvement Potential")
        
    #     predictions = st.session_state.predictions
    #     user_data = st.session_state.user_data
        
    #     improvements = calculate_health_improvement_potential(user_data, predictions)
        
    #     if improvements:
    #         # Create improvement chart
    #         improvement_fig = create_improvement_potential_chart(improvements)
    #         st.plotly_chart(improvement_fig, use_container_width=True)
            
    #         # Display specific improvements
    #         st.markdown("#### Potential Risk Reductions")
            
    #         for disease, improvement in improvements.items():
    #             current_risk = improvement['current_risk']
    #             improved_risk = improvement['improved_risk']
    #             reduction = improvement['potential_reduction']
                
    #             if reduction > 0:
    #                 st.markdown(f"""
    #                 <div class="metric-container">
    #                     <h5>{disease.title()}</h5>
    #                     <p><strong>Current Risk:</strong> {current_risk:.1f}%</p>
    #                     <p><strong>Potential Risk:</strong> {improved_risk:.1f}%</p>
    #                     <p style="color: green;"><strong>Potential Reduction:</strong> {reduction:.1f}%</p>
    #                 </div>
    #                 """, unsafe_allow_html=True)
            
    #         # Timeline projection
    #         timeline_fig = create_health_timeline(user_data, predictions)
    #         st.plotly_chart(timeline_fig, use_container_width=True)
    
    # def generate_and_display_report(self):
    #     """Generate and display comprehensive health report"""
    #     predictions = st.session_state.predictions
    #     user_data = st.session_state.user_data
    #     cluster_info = st.session_state.get('cluster_info')
    #     explanations = st.session_state.get('explanations')
        
    #     report = generate_health_report(user_data, predictions, cluster_info, explanations)
        
    #     st.markdown("### 📋 Comprehensive Health Report")
    #     st.markdown(report)
        
    #     # Download button
    #     st.download_button(
    #         label="📥 Download Report as Markdown",
    #         data=report,
    #         file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
    #         mime="text/markdown"
    #     )
    def display_explanations(self):
        """Display AI model explanations"""
        st.markdown("### 🤖 AI Model Explanations")
        
        # Safe check for explanations
        explanations = getattr(st.session_state, 'explanations', None)
        if not explanations:
            st.info("Run analysis to see AI explanations")
            return
        
        if not isinstance(explanations, dict):
            st.warning("Explanations data is not in expected format")
            return
        
        for disease, exp_data in explanations.items():
            if not exp_data or not isinstance(exp_data, dict):
                continue
                
            with st.expander(f"{disease.title()} Model Explanation"):
                try:
                    if 'unified_explanation' in exp_data:
                        st.markdown(exp_data['unified_explanation'])
                    
                    # Feature importance chart
                    if ('explanations' in exp_data and 
                        isinstance(exp_data['explanations'], dict) and
                        'shap' in exp_data['explanations']):
                        
                        shap_data = exp_data['explanations']['shap']
                        if isinstance(shap_data, dict) and 'top_features' in shap_data:
                            try:
                                importance_fig = create_feature_importance_chart(
                                    shap_data['top_features'], 
                                    f"{disease.title()} Feature Importance"
                                )
                                if importance_fig:
                                    st.plotly_chart(importance_fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create importance chart: {e}")
                                
                except Exception as e:
                    st.error(f"Error displaying explanation for {disease}: {e}")
    
    def display_clustering_analysis(self):
        """Display clustering analysis"""
        st.markdown("### 👥 Patient Clustering Analysis")
        
        # Safe check for cluster_info
        cluster_info = getattr(st.session_state, 'cluster_info', None)
        if not cluster_info:
            st.info("Run analysis to see clustering results")
            return
        
        if not isinstance(cluster_info, dict):
            st.warning("Clustering data is not in expected format")
            return
        
        try:
            if 'cluster_profile' in cluster_info and cluster_info['cluster_profile']:
                profile = cluster_info['cluster_profile']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Your Patient Group")
                    st.markdown(f"**Cluster ID:** {cluster_info.get('cluster_id', 'Unknown')}")
                    st.markdown(f"**Similarity Score:** {cluster_info.get('similarity_score', 0):.2f}")
                    
                    if isinstance(profile, dict) and 'description' in profile:
                        st.markdown(f"**Group Description:** {profile['description']}")
                    
                    if isinstance(profile, dict) and 'risk_profile' in profile:
                        risk_profile = profile['risk_profile']
                        if isinstance(risk_profile, dict):
                            overall_risk = risk_profile.get('overall', 'Unknown')
                            try:
                                color = format_risk_level_color(overall_risk)
                                st.markdown(f"""
                                <div class="metric-container">
                                    <h5>Group Risk Profile</h5>
                                    <p style="color: {color}; font-weight: bold;">Overall Risk: {overall_risk}</p>
                                    <p>Age Risk: {risk_profile.get('age_risk', 'N/A')}</p>
                                    <p>BP Risk: {risk_profile.get('bp_risk', 'N/A')}</p>
                                    <p>Cholesterol Risk: {risk_profile.get('cholesterol_risk', 'N/A')}</p>
                                    <p>BMI Risk: {risk_profile.get('bmi_risk', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.markdown(f"**Overall Risk:** {overall_risk}")
                                st.warning(f"Display formatting error: {e}")
                
                with col2:
                    # Cluster visualization
                    try:
                        if (hasattr(self, 'clustering') and 
                            self.clustering and 
                            hasattr(self.clustering, 'dimensionality_reducers')):
                            
                            cluster_viz = create_cluster_visualization(
                                self.clustering.dimensionality_reducers,
                                cluster_info.get('cluster_id', 0)
                            )
                            if cluster_viz:
                                st.plotly_chart(cluster_viz, use_container_width=True)
                        else:
                            st.info("Cluster visualization not available")
                    except Exception as e:
                        st.warning(f"Could not create cluster visualization: {e}")
            else:
                st.info("No detailed cluster profile available")
                
        except Exception as e:
            st.error(f"Error displaying clustering analysis: {e}")
    
    def display_anomaly_analysis(self):
        """Display anomaly detection results"""
        st.markdown("### ⚠️ Anomaly Detection Analysis")
        
        # Safe check for anomaly_info
        anomaly_info = getattr(st.session_state, 'anomaly_info', None)
        if not anomaly_info:
            st.info("Run analysis to see anomaly detection results")
            return
        
        if not isinstance(anomaly_info, dict):
            st.warning("Anomaly data is not in expected format")
            return
        
        try:
            is_anomaly = anomaly_info.get('is_anomaly', False)
            confidence = anomaly_info.get('confidence', 0)
            interpretation = anomaly_info.get('interpretation', 'No interpretation available')
            
            if is_anomaly:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Anomaly Detected</h4>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p>{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Normal Health Profile</h4>
                    <p>Your health profile appears normal compared to the population.</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error displaying anomaly analysis: {e}")
    
    def display_improvement_potential(self):
        """Display health improvement potential"""
        st.markdown("### 📈 Health Improvement Potential")
        
        # Safe checks for required data
        predictions = getattr(st.session_state, 'predictions', None)
        user_data = getattr(st.session_state, 'user_data', None)
        
        if not predictions or not user_data:
            st.info("Run analysis to see improvement potential")
            return
        
        try:
            improvements = calculate_health_improvement_potential(user_data, predictions)
            
            if improvements and isinstance(improvements, dict):
                # Create improvement chart
                try:
                    improvement_fig = create_improvement_potential_chart(improvements)
                    if improvement_fig:
                        st.plotly_chart(improvement_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create improvement chart: {e}")
                
                # Display specific improvements
                st.markdown("#### Potential Risk Reductions")
                
                for disease, improvement in improvements.items():
                    if not isinstance(improvement, dict):
                        continue
                        
                    try:
                        current_risk = improvement.get('current_risk', 0)
                        improved_risk = improvement.get('improved_risk', 0)
                        reduction = improvement.get('potential_reduction', 0)
                        
                        if reduction > 0:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h5>{disease.title()}</h5>
                                <p><strong>Current Risk:</strong> {current_risk:.1f}%</p>
                                <p><strong>Potential Risk:</strong> {improved_risk:.1f}%</p>
                                <p style="color: green;"><strong>Potential Reduction:</strong> {reduction:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Error displaying improvement for {disease}: {e}")
                
                # Timeline projection
                try:
                    timeline_fig = create_health_timeline(user_data, predictions)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create timeline projection: {e}")
            else:
                st.info("No improvement potential data available")
                
        except Exception as e:
            st.error(f"Error calculating improvement potential: {e}")
    
    def generate_and_display_report(self):
        """Generate and display comprehensive health report"""
        try:
            predictions = getattr(st.session_state, 'predictions', {})
            user_data = getattr(st.session_state, 'user_data', {})
            cluster_info = getattr(st.session_state, 'cluster_info', None)
            explanations = getattr(st.session_state, 'explanations', None)
            
            if not predictions or not user_data:
                st.error("No analysis data available to generate report")
                return
            
            report = generate_health_report(user_data, predictions, cluster_info, explanations)
            
            st.markdown("### 📋 Comprehensive Health Report")
            st.markdown(report)
            
            # Download button
            st.download_button(
                label="📥 Download Report as Markdown",
                data=report,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error generating report: {e}")
            
    def render_about_page(self):
        """Render about page"""
        st.markdown("# 🏥 About Smart Health Risk Analyzer")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        The **Smart Health Risk Analyzer** is an AI-powered comprehensive health assessment platform that leverages 
        cutting-edge machine learning and deep learning techniques to predict multiple health risks and provide 
        personalized insights.
        
        ## 🧠 AI Technologies Used
        
        ### Machine Learning Models
        - **Logistic Regression**: Linear baseline for interpretability
        - **Random Forest**: Ensemble method for robust predictions
        - **Gradient Boosting**: Advanced boosting for high accuracy
        - **XGBoost**: Optimized gradient boosting framework
        - **LightGBM**: Fast gradient boosting with low memory usage
        - **Support Vector Machines**: Kernel-based classification
        - **K-Nearest Neighbors**: Instance-based learning
        - **Naive Bayes**: Probabilistic classification
        
        ### Deep Learning Models
        - **Multi-Layer Perceptron (MLP)**: Neural networks for tabular data
        - **LSTM Networks**: Time-series health data analysis
        - **Ensemble Models**: Voting classifiers combining multiple algorithms
        
        ### Unsupervised Learning
        - **K-Means Clustering**: Patient segmentation
        - **Gaussian Mixture Models**: Probabilistic clustering
        - **DBSCAN**: Density-based clustering
        - **PCA**: Dimensionality reduction for visualization
        - **UMAP**: Non-linear dimensionality reduction
        
        ### Model Explainability
        - **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
        - **LIME (Local Interpretable Model-agnostic Explanations)**: Local model explanations
        
        ### Anomaly Detection
        - **Isolation Forest**: Unsupervised outlier detection
        - **Local Outlier Factor**: Local density-based anomaly detection
        
        ## 📊 Datasets
        
        The system is trained on multiple validated medical datasets:
        
        1. **Heart Disease Dataset (UCI)**: Cleveland heart disease data
        2. **Diabetes Dataset (PIMA Indian)**: Diabetes prediction data
        3. **Stroke Prediction Dataset**: Stroke risk factors
        4. **Insurance Dataset**: Healthcare cost analysis
        
        ## 🔧 Technical Features
        
        - **Automated Data Preprocessing**: Missing value imputation, feature engineering
        - **Hyperparameter Optimization**: Optuna-based automated tuning
        - **Real-time Predictions**: Instant health risk assessment
        - **Interactive Visualizations**: Plotly-based dynamic charts
        - **Comprehensive Reporting**: Automated health report generation
        - **Model Persistence**: Trained model storage and loading
        
        ## ⚠️ Disclaimer
        
        This application is for **educational and research purposes only**. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals 
        for medical decisions.
        
        ## 👨‍💻 About the Developer
        
        This project showcases expertise in:
        - Machine Learning & Deep Learning
        - Healthcare Data Science
        - Model Explainability & Interpretability
        - Full-Stack ML Application Development
        - Advanced Data Visualization
        """)
    
    def run(self):
        """Main application runner"""
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        page = st.sidebar.selectbox("Choose a page", ["🏠 Health Analysis", "ℹ️ About"])
        
        if page == "🏠 Health Analysis":
            # Load models first
            if not self.load_models():
                st.error("Failed to load models. Please check your setup.")
                return
            
            # Get user input
            user_data = self.render_sidebar()
            
            # Render main dashboard
            self.render_main_dashboard(user_data)
            
        elif page == "ℹ️ About":
            self.render_about_page()

# Run the app
if __name__ == "__main__":
    app = HealthApp()
    app.run()