import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

class HealthModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.model_scores = {}
        
    def create_mlp_model(self, input_dim, task_type='classification'):
        """Create Multi-Layer Perceptron for tabular data"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        loss = 'binary_crossentropy' if task_type == 'classification' else 'mse'
        metrics = ['accuracy'] if task_type == 'classification' else ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def create_lstm_model(self, input_dim, sequence_length=10):
        """Create LSTM model for time-series health data"""
        # Adjust sequence_length to make dimensions compatible
        if input_dim < sequence_length:
            sequence_length = input_dim
        else:
            # Find the largest divisor of input_dim that's <= 10
            for seq_len in range(min(10, input_dim), 0, -1):
                if input_dim % seq_len == 0:
                    sequence_length = seq_len
                    break
        
        features_per_timestep = input_dim // sequence_length
        
        model = Sequential([
            Reshape((sequence_length, features_per_timestep), input_shape=(input_dim,)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def get_classical_models(self, task_type='classification'):
        """Get classical ML models"""
        if task_type == 'classification':
            return {
                'logistic_regression': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'lightgbm': lgb.LGBMClassifier(random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'ada_boost': AdaBoostClassifier(random_state=42)
            }
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            
            return {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'xgboost': xgb.XGBRegressor(random_state=42),
                'lightgbm': lgb.LGBMRegressor(random_state=42),
                'svr': SVR()
            }
    
    def optimize_hyperparameters(self, model_name, X_train, y_train, X_val, y_val, task_type='classification'):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
                if task_type == 'classification':
                    model = RandomForestClassifier(**params, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(**params, random_state=42)
                    
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                if task_type == 'classification':
                    model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                else:
                    model = xgb.XGBRegressor(**params, random_state=42)
                    
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                if task_type == 'classification':
                    model = lgb.LGBMClassifier(**params, random_state=42)
                else:
                    model = lgb.LGBMRegressor(**params, random_state=42)
            else:
                return 0
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            if task_type == 'classification':
                return accuracy_score(y_val, predictions)
            else:
                return r2_score(y_val, predictions)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
    
    def evaluate_model(self, model, X_test, y_test, task_type='classification'):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        if task_type == 'classification':
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            else:
                auc = 0.5
                
            return {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1': f1_score(y_test, predictions, average='weighted'),
                'auc': auc
            }
        else:
            return {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
    
    def train_deep_learning_models(self, X_train, y_train, X_test, y_test, dataset_name, task_type='classification'):
        """Train deep learning models"""
        input_dim = X_train.shape[1]
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # MLP Model
        try:
            mlp_model = self.create_mlp_model(input_dim, task_type)
            
            mlp_model.fit(X_train, y_train, 
                         validation_split=0.2,
                         epochs=100, 
                         batch_size=32, 
                         callbacks=[early_stopping],
                         verbose=0)
            
            # Save MLP model
            self.models[f'{dataset_name}_mlp'] = mlp_model
            
            # Evaluate MLP
            mlp_predictions = (mlp_model.predict(X_test) > 0.5).astype(int) if task_type == 'classification' else mlp_model.predict(X_test)
            mlp_scores = self.evaluate_model(type('MockModel', (), {
                'predict': lambda self, X: mlp_predictions.flatten(),
                'predict_proba': lambda self, X: np.column_stack([1-mlp_model.predict(X), mlp_model.predict(X)])
            })(), X_test, y_test, task_type)
            
            self.model_scores[f'{dataset_name}_mlp'] = mlp_scores
            
        except Exception as e:
            print(f"Warning: MLP training failed for {dataset_name}: {e}")
        
        # LSTM Model (if applicable and dimensions are compatible)
        if input_dim >= 4:  # Only create LSTM if we have at least 4 features
            try:
                lstm_model = self.create_lstm_model(input_dim)
                lstm_model.fit(X_train, y_train,
                              validation_split=0.2,
                              epochs=50,
                              batch_size=32,
                              callbacks=[early_stopping],
                              verbose=0)
                
                self.models[f'{dataset_name}_lstm'] = lstm_model
                
                # Evaluate LSTM
                lstm_predictions = (lstm_model.predict(X_test) > 0.5).astype(int)
                lstm_scores = self.evaluate_model(type('MockModel', (), {
                    'predict': lambda self, X: lstm_predictions.flatten(),
                    'predict_proba': lambda self, X: np.column_stack([1-lstm_model.predict(X), lstm_model.predict(X)])
                })(), X_test, y_test, task_type)
                
                self.model_scores[f'{dataset_name}_lstm'] = lstm_scores
                
            except Exception as e:
                print(f"Warning: LSTM training failed for {dataset_name}: {e}")
                # Continue without LSTM model
        
        # Return MLP model as primary (or None if failed)
        return self.models.get(f'{dataset_name}_mlp')
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test, dataset_name, task_type='classification'):
        """Train ensemble models"""
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        base_models = self.get_classical_models(task_type)
        
        # Select top 3 models for ensemble
        if task_type == 'classification':
            ensemble_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
                ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
            ]
            
            # Soft voting classifier
            ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        else:
            from sklearn.ensemble import RandomForestRegressor
            ensemble_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42))
            ]
            
            ensemble = VotingRegressor(estimators=ensemble_models)
        
        ensemble.fit(X_train, y_train)
        self.models[f'{dataset_name}_ensemble'] = ensemble
        
        # Evaluate ensemble
        ensemble_scores = self.evaluate_model(ensemble, X_test, y_test, task_type)
        self.model_scores[f'{dataset_name}_ensemble'] = ensemble_scores
        
        return ensemble
    
    def train_all_models(self, prepared_data):
        """Train all models for all datasets"""
        print("🚀 Starting comprehensive model training...")
        
        for dataset_name, data in prepared_data.items():
            print(f"\n📈 Training models for {dataset_name} dataset...")
            
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            
            task_type = 'regression' if dataset_name == 'insurance' else 'classification'
            
            # Train classical ML models
            classical_models = self.get_classical_models(task_type)
            
            dataset_best_score = 0
            dataset_best_model = None
            dataset_best_name = None
            
            for model_name, model in classical_models.items():
                print(f"  🔧 Training {model_name}...")
                
                try:
                    # Hyperparameter optimization for key models
                    if model_name in ['random_forest', 'xgboost', 'lightgbm']:
                        # Use validation split for optimization
                        val_size = int(0.2 * len(X_train))
                        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                        X_train_opt, y_train_opt = X_train[:-val_size], y_train[:-val_size]
                        
                        try:
                            best_params = self.optimize_hyperparameters(
                                model_name, X_train_opt, y_train_opt, X_val, y_val, task_type
                            )
                            self.best_params[f'{dataset_name}_{model_name}'] = best_params
                            
                            # Update model with best parameters
                            model.set_params(**best_params)
                        except Exception as e:
                            print(f"    Warning: Hyperparameter optimization failed for {model_name}: {e}")
                    
                    model.fit(X_train, y_train)
                    full_model_name = f'{dataset_name}_{model_name}'
                    self.models[full_model_name] = model
                    
                    # Evaluate model
                    scores = self.evaluate_model(model, X_test, y_test, task_type)
                    self.model_scores[full_model_name] = scores
                    
                    # Track best model for this dataset
                    score = scores.get('f1', scores.get('r2', 0))
                    if score > dataset_best_score:
                        dataset_best_score = score
                        dataset_best_model = model
                        dataset_best_name = full_model_name
                        
                except Exception as e:
                    print(f"  ❌ Error training {model_name}: {e}")
                    continue
            
            # Train deep learning models (if successful classical training)
            if dataset_best_model is not None:
                print(f"  🧠 Training deep learning models...")
                try:
                    self.train_deep_learning_models(X_train, y_train, X_test, y_test, dataset_name, task_type)
                except Exception as e:
                    print(f"  Warning: Deep learning training failed for {dataset_name}: {e}")
                
                # Train ensemble models
                print(f"  🏆 Training ensemble models...")
                try:
                    self.train_ensemble_models(X_train, y_train, X_test, y_test, dataset_name, task_type)
                except Exception as e:
                    print(f"  Warning: Ensemble training failed for {dataset_name}: {e}")
        
        print("\n✅ Model training completed!")
        return self.get_best_models()
    
    def get_best_models(self):
        """Get best performing model for each dataset"""
        best_models = {}
        
        for dataset in ['heart', 'diabetes', 'stroke', 'insurance']:
            dataset_scores = {k: v for k, v in self.model_scores.items() if k.startswith(dataset)}
            
            # Skip if no models found for this dataset
            if not dataset_scores:
                print(f"Warning: No models found for {dataset} dataset")
                continue
            
            try:
                if dataset == 'insurance':
                    # For regression, use R2 score
                    best_model_name = max(dataset_scores.keys(), 
                                        key=lambda x: dataset_scores[x].get('r2', 0))
                else:
                    # For classification, use F1 score
                    best_model_name = max(dataset_scores.keys(), 
                                        key=lambda x: dataset_scores[x].get('f1', 0))
                
                # Verify the model exists
                if best_model_name in self.models:
                    best_models[dataset] = {
                        'model': self.models[best_model_name],
                        'scores': dataset_scores[best_model_name],
                        'name': best_model_name
                    }
                else:
                    print(f"Warning: Best model {best_model_name} not found in trained models")
                    
            except Exception as e:
                print(f"Error selecting best model for {dataset}: {e}")
                continue
        
        return best_models
    
    def save_models(self, filepath='models/'):
        """Save all trained models"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'mlp' in model_name or 'lstm' in model_name:
                model.save(f'{filepath}{model_name}.h5')
            else:
                joblib.dump(model, f'{filepath}{model_name}.pkl')
        
        # Save scores and parameters
        joblib.dump(self.model_scores, f'{filepath}model_scores.pkl')
        joblib.dump(self.best_params, f'{filepath}best_params.pkl')
        
        print(f"💾 Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models"""
        import os
        
        if not os.path.exists(filepath):
            print(f"❌ Models directory {filepath} does not exist")
            return False
        
        loaded_count = 0
        
        for filename in os.listdir(filepath):
            if filename.endswith('.pkl') and 'model_scores' not in filename and 'best_params' not in filename:
                try:
                    model_name = filename.replace('.pkl', '')
                    self.models[model_name] = joblib.load(f'{filepath}{filename}')
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
                    
            elif filename.endswith('.h5'):
                try:
                    model_name = filename.replace('.h5', '')
                    self.models[model_name] = tf.keras.models.load_model(f'{filepath}{filename}')
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
        
        # Load scores and parameters
        try:
            if os.path.exists(f'{filepath}model_scores.pkl'):
                self.model_scores = joblib.load(f'{filepath}model_scores.pkl')
        except Exception as e:
            print(f"Warning: Failed to load model scores: {e}")
            self.model_scores = {}
            
        try:
            if os.path.exists(f'{filepath}best_params.pkl'):
                self.best_params = joblib.load(f'{filepath}best_params.pkl')
        except Exception as e:
            print(f"Warning: Failed to load best params: {e}")
            self.best_params = {}
        
        if loaded_count > 0:
            print(f"📁 Models loaded from {filepath}")
            print(f"✅ Successfully loaded {loaded_count} models")
            return True
        else:
            print(f"❌ No models could be loaded from {filepath}")
            return False