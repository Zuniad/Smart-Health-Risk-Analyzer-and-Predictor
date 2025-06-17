#!/usr/bin/env python3
"""
Smart Health Risk Analyzer - Model Training Script

This script trains all ML/DL models for the health risk prediction system.
Run this script to train models on your datasets before using the Streamlit app.

Usage:
    python train_models.py [--retrain] [--models heart,diabetes,stroke] [--verbose]
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import HealthDataProcessor
from model_trainer import HealthModelTrainer
from clustering import HealthClustering

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    def __init__(self, retrain=False, models_to_train=None, verbose=False):
        self.retrain = retrain
        self.models_to_train = models_to_train or ['heart', 'diabetes', 'stroke', 'insurance']
        self.verbose = verbose
        
        # Initialize components
        self.data_processor = HealthDataProcessor()
        self.model_trainer = HealthModelTrainer()
        self.clustering = HealthClustering()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/clustering', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def check_data_availability(self):
        """Check if required datasets are available"""
        logger.info("🔍 Checking data availability...")
        
        required_files = [
            'data/heart_cleveland_upload.csv',
            'data/diabetes.csv',
            'data/healthcare-dataset-stroke-data.csv',
            'data/insurance.csv'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error("❌ Missing required data files:")
            for file in missing_files:
                logger.error(f"   - {file}")
            logger.error("\nPlease ensure all required CSV files are in the 'data/' directory.")
            return False
        
        logger.info("✅ All required data files found!")
        return True
    
    def check_existing_models(self):
        """Check if models already exist"""
        if not self.retrain and os.path.exists('models/') and len(os.listdir('models/')) > 0:
            logger.info("📁 Existing models found.")
            
            if not self.retrain:
                response = input("Do you want to retrain models? (y/N): ").lower()
                if response != 'y':
                    logger.info("Using existing models. Use --retrain flag to force retraining.")
                    return False
            
        return True
    
    def prepare_data(self):
        """Prepare all datasets"""
        logger.info("📊 Preparing datasets...")
        start_time = time.time()
        
        try:
            prepared_data = self.data_processor.prepare_datasets()
            
            if prepared_data is None:
                logger.error("❌ Failed to prepare datasets")
                return None
            
            # Log dataset statistics
            for dataset_name, data in prepared_data.items():
                if dataset_name in self.models_to_train:
                    train_samples = data['X_train'].shape[0]
                    test_samples = data['X_test'].shape[0]
                    features = data['X_train'].shape[1]
                    
                    logger.info(f"   {dataset_name}: {train_samples} train, {test_samples} test, {features} features")
            
            prep_time = time.time() - start_time
            logger.info(f"✅ Data preparation completed in {prep_time:.2f}s")
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {str(e)}")
            return None
    
    def train_classification_models(self, prepared_data):
        """Train classification models for health prediction"""
        logger.info("🤖 Training classification models...")
        start_time = time.time()
        
        try:
            # Filter datasets for training
            filtered_data = {k: v for k, v in prepared_data.items() 
                           if k in self.models_to_train and k != 'insurance'}
            
            if not filtered_data:
                logger.warning("No classification datasets selected for training")
                return None
            
            # Train models with error handling
            try:
                best_models = self.model_trainer.train_all_models(filtered_data)
            except Exception as e:
                logger.error(f"Error in main training: {e}")
                # Try to train models individually
                logger.info("Attempting individual model training...")
                best_models = self._train_models_individually(filtered_data)
            
            # Log model performance
            if best_models:
                logger.info("📈 Model Performance Summary:")
                for dataset_name, model_info in best_models.items():
                    model_name = model_info['name']
                    scores = model_info['scores']
                    
                    if 'f1' in scores:
                        logger.info(f"   {dataset_name}: {model_name} - F1: {scores['f1']:.3f}, Accuracy: {scores['accuracy']:.3f}")
                    elif 'r2' in scores:
                        logger.info(f"   {dataset_name}: {model_name} - R2: {scores['r2']:.3f}")
            
            training_time = time.time() - start_time
            logger.info(f"✅ Classification training completed in {training_time:.2f}s")
            
            return best_models
            
        except Exception as e:
            logger.error(f"❌ Classification training failed: {str(e)}")
            return None
    
    def _train_models_individually(self, filtered_data):
        """Fallback method to train models individually"""
        logger.info("🔄 Training models individually as fallback...")
        
        best_models = {}
        
        for dataset_name, data in filtered_data.items():
            logger.info(f"   Training models for {dataset_name}...")
            
            try:
                X_train, X_test = data['X_train'], data['X_test']
                y_train, y_test = data['y_train'], data['y_test']
                
                task_type = 'regression' if dataset_name == 'insurance' else 'classification'
                
                # Train classical models only (skip deep learning for stability)
                classical_models = self.model_trainer.get_classical_models(task_type)
                
                best_score = 0
                best_model = None
                best_model_name = None
                
                for model_name, model in classical_models.items():
                    try:
                        model.fit(X_train, y_train)
                        scores = self.model_trainer.evaluate_model(model, X_test, y_test, task_type)
                        
                        # Store model and scores
                        full_model_name = f'{dataset_name}_{model_name}'
                        self.model_trainer.models[full_model_name] = model
                        self.model_trainer.model_scores[full_model_name] = scores
                        
                        # Track best model
                        score = scores.get('f1', scores.get('r2', 0))
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_model_name = full_model_name
                        
                        logger.info(f"     ✓ {model_name}: {score:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"     ❌ {model_name} failed: {e}")
                
                if best_model:
                    best_models[dataset_name] = {
                        'model': best_model,
                        'scores': self.model_trainer.model_scores[best_model_name],
                        'name': best_model_name
                    }
                    logger.info(f"   ✅ Best model for {dataset_name}: {best_model_name}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to train models for {dataset_name}: {e}")
        
        return best_models if best_models else None
    
    def train_clustering_models(self, prepared_data):
        """Train clustering and anomaly detection models"""
        logger.info("🔬 Training clustering models...")
        start_time = time.time()
        
        try:
            # Prepare clustering data
            clustering_data, unified_df = self.clustering.prepare_clustering_data(prepared_data)
            
            if clustering_data is None:
                logger.error("Failed to prepare clustering data")
                return False
            
            logger.info(f"   Clustering data shape: {clustering_data.shape}")
            
            # Perform clustering
            clustering_results, best_method = self.clustering.perform_clustering(clustering_data)
            logger.info(f"   Best clustering method: {best_method}")
            
            # Create cluster profiles
            best_labels = clustering_results[best_method]['labels']
            feature_names = clustering_data.columns.tolist()
            cluster_profiles = self.clustering.create_cluster_profiles(clustering_data.values, best_labels, feature_names)
            
            logger.info(f"   Created {len(cluster_profiles)} cluster profiles")
            
            # Dimensionality reduction
            dim_reducers = self.clustering.perform_dimensionality_reduction(clustering_data)
            logger.info("   Dimensionality reduction completed")
            
            # Anomaly detection
            anomaly_results = self.clustering.detect_anomalies(clustering_data)
            
            anomaly_count = len(anomaly_results['isolation_forest']['anomaly_indices'])
            total_samples = len(clustering_data)
            anomaly_rate = (anomaly_count / total_samples) * 100
            
            logger.info(f"   Detected {anomaly_count}/{total_samples} anomalies ({anomaly_rate:.1f}%)")
            
            clustering_time = time.time() - start_time
            logger.info(f"✅ Clustering training completed in {clustering_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Clustering training failed: {str(e)}")
            return False
    
    def save_models(self):
        """Save all trained models"""
        logger.info("💾 Saving models...")
        start_time = time.time()
        
        try:
            # Save classification models
            self.model_trainer.save_models()
            
            # Save clustering models
            self.clustering.save_clustering_models()
            
            save_time = time.time() - start_time
            logger.info(f"✅ Models saved successfully in {save_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {str(e)}")
            return False
    
    def validate_models(self, prepared_data, best_models):
        """Validate trained models"""
        logger.info("🔍 Validating trained models...")
        
        try:
            validation_results = {}
            
            for dataset_name, model_info in best_models.items():
                model = model_info['model']
                scores = model_info['scores']
                
                # Basic validation - check if scores are reasonable
                if 'accuracy' in scores and scores['accuracy'] > 0.5:
                    validation_results[dataset_name] = 'PASS'
                    logger.info(f"   ✅ {dataset_name}: Model validation passed")
                else:
                    validation_results[dataset_name] = 'FAIL'
                    logger.warning(f"   ⚠️ {dataset_name}: Model validation failed")
            
            # Overall validation
            passed = sum(1 for result in validation_results.values() if result == 'PASS')
            total = len(validation_results)
            
            logger.info(f"📊 Validation Summary: {passed}/{total} models passed validation")
            
            if passed == total:
                logger.info("✅ All models passed validation!")
                return True
            else:
                logger.warning("⚠️ Some models failed validation. Check training logs.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model validation failed: {str(e)}")
            return False
    
    def generate_training_report(self, prepared_data, best_models, training_time):
        """Generate comprehensive training report"""
        logger.info("📋 Generating training report...")
        
        try:
            report_lines = [
                "# Smart Health Risk Analyzer - Training Report",
                f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Training Time:** {training_time:.2f} seconds",
                "",
                "## Dataset Summary",
                ""
            ]
            
            # Dataset information
            for dataset_name, data in prepared_data.items():
                if dataset_name in self.models_to_train:
                    train_samples = data['X_train'].shape[0]
                    test_samples = data['X_test'].shape[0]
                    features = data['X_train'].shape[1]
                    
                    report_lines.extend([
                        f"### {dataset_name.title()} Dataset",
                        f"- Training samples: {train_samples}",
                        f"- Test samples: {test_samples}",
                        f"- Features: {features}",
                        ""
                    ])
            
            # Model performance
            report_lines.extend([
                "## Model Performance",
                ""
            ])
            
            for dataset_name, model_info in best_models.items():
                model_name = model_info['name']
                scores = model_info['scores']
                
                report_lines.extend([
                    f"### {dataset_name.title()}",
                    f"- **Best Model:** {model_name}",
                ])
                
                for metric, value in scores.items():
                    report_lines.append(f"- **{metric.title()}:** {value:.4f}")
                
                report_lines.append("")
            
            # Save report
            report_content = "\n".join(report_lines)
            report_filename = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            
            with open(report_filename, 'w') as f:
                f.write(report_content)
            
            logger.info(f"📄 Training report saved as: {report_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return False
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting Smart Health Risk Analyzer training pipeline...")
        pipeline_start = time.time()
        
        # Step 1: Check data availability
        if not self.check_data_availability():
            return False
        
        # Step 2: Check existing models
        if not self.check_existing_models():
            return True  # Use existing models
        
        # Step 3: Prepare data
        prepared_data = self.prepare_data()
        if prepared_data is None:
            return False
        
        # Step 4: Train classification models
        best_models = self.train_classification_models(prepared_data)
        if best_models is None:
            return False
        
        # Step 5: Train clustering models
        if not self.train_clustering_models(prepared_data):
            return False
        
        # Step 6: Save models
        if not self.save_models():
            return False
        
        # Step 7: Validate models
        if not self.validate_models(prepared_data, best_models):
            logger.warning("⚠️ Model validation failed, but continuing...")
        
        # Step 8: Generate report
        total_time = time.time() - pipeline_start
        self.generate_training_report(prepared_data, best_models, total_time)
        
        logger.info(f"🎉 Training pipeline completed successfully in {total_time:.2f}s!")
        logger.info("You can now run the Streamlit app: streamlit run app.py")
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Smart Health Risk Analyzer models')
    parser.add_argument('--retrain', action='store_true', 
                       help='Force retrain even if models exist')
    parser.add_argument('--models', type=str, default='heart,diabetes,stroke,insurance',
                       help='Comma-separated list of models to train')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Parse models list
    models_to_train = [model.strip() for model in args.models.split(',')]
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = ModelTrainingPipeline(
        retrain=args.retrain,
        models_to_train=models_to_train,
        verbose=args.verbose
    )
    
    success = pipeline.run_training_pipeline()
    
    if success:
        logger.info("✅ Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()