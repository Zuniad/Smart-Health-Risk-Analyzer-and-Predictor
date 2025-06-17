import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import umap
import joblib
import warnings
warnings.filterwarnings('ignore')

class HealthClustering:
    def __init__(self):
        self.clustering_models = {}
        self.dimensionality_reducers = {}
        self.anomaly_detectors = {}
        self.cluster_profiles = {}
        self.scaler = StandardScaler()
        
    def prepare_clustering_data(self, datasets):
        """Prepare unified dataset for clustering analysis"""
        combined_data = []
        
        for dataset_name, data in datasets.items():
            if dataset_name == 'insurance':  # Skip regression dataset
                continue
                
            df = pd.DataFrame(data['original_X'])
            df['dataset_source'] = dataset_name
            df['target'] = data['original_y']
            
            # Add common health metrics
            if dataset_name == 'heart':
                df['health_age'] = df['age']
                df['health_bp'] = df.get('trestbps', 120)
                df['health_chol'] = df.get('chol', 200)
                df['health_bmi'] = 25.0  # Default BMI
                
            elif dataset_name == 'diabetes':
                df['health_age'] = df['Age']
                df['health_bp'] = df['BloodPressure']
                df['health_chol'] = 200  # Default cholesterol
                df['health_bmi'] = df['BMI']
                
            elif dataset_name == 'stroke':
                df['health_age'] = df['age']
                df['health_bp'] = 120  # Default BP
                df['health_chol'] = 200  # Default cholesterol
                df['health_bmi'] = df['bmi']
            
            combined_data.append(df)
        
        # Combine all datasets
        unified_df = pd.concat(combined_data, ignore_index=True)
        
        # Select common features for clustering
        common_features = ['health_age', 'health_bp', 'health_chol', 'health_bmi']
        clustering_features = unified_df[common_features].fillna(unified_df[common_features].median())
        
        return clustering_features, unified_df
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:  # Avoid silhouette score error
                sil_score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return optimal_k, inertias, silhouette_scores
    
    def perform_clustering(self, X, n_clusters=None):
        """Perform multiple clustering algorithms"""
        X_scaled = self.scaler.fit_transform(X)
        
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_clusters(X_scaled)
        
        clustering_results = {}
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        clustering_results['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'silhouette_score': silhouette_score(X_scaled, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0
        }
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        clustering_results['gmm'] = {
            'model': gmm,
            'labels': gmm_labels,
            'silhouette_score': silhouette_score(X_scaled, gmm_labels) if len(np.unique(gmm_labels)) > 1 else 0
        }
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        clustering_results['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'silhouette_score': silhouette_score(X_scaled, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else 0
        }
        
        # Select best clustering method
        best_method = max(clustering_results.keys(), 
                         key=lambda x: clustering_results[x]['silhouette_score'])
        
        self.clustering_models = clustering_results
        return clustering_results, best_method
    
    def perform_dimensionality_reduction(self, X):
        """Perform PCA and UMAP for visualization"""
        X_scaled = self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = umap_reducer.fit_transform(X_scaled)
        
        self.dimensionality_reducers = {
            'pca': {'model': pca, 'data': X_pca, 'explained_variance': pca.explained_variance_ratio_},
            'umap': {'model': umap_reducer, 'data': X_umap}
        }
        
        return self.dimensionality_reducers
    
    def detect_anomalies(self, X):
        """Detect anomalies using multiple methods"""
        X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(X)
        
        anomaly_results = {}
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_anomalies = iso_forest.fit_predict(X_scaled)
        iso_scores = iso_forest.score_samples(X_scaled)
        
        anomaly_results['isolation_forest'] = {
            'model': iso_forest,
            'anomalies': iso_anomalies,
            'scores': iso_scores,
            'anomaly_indices': np.where(iso_anomalies == -1)[0]
        }
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.1)
        lof_anomalies = lof.fit_predict(X_scaled)
        lof_scores = lof.negative_outlier_factor_
        
        anomaly_results['lof'] = {
            'model': lof,
            'anomalies': lof_anomalies,
            'scores': lof_scores,
            'anomaly_indices': np.where(lof_anomalies == -1)[0]
        }
        
        self.anomaly_detectors = anomaly_results
        return anomaly_results
    
    def create_cluster_profiles(self, X, labels, feature_names):
        """Create detailed profiles for each cluster"""
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = labels
        
        profiles = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points from DBSCAN
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'characteristics': {},
                'risk_profile': self._assess_cluster_risk(cluster_data)
            }
            
            # Calculate statistics for each feature
            for feature in feature_names:
                feature_data = cluster_data[feature]
                profile['characteristics'][feature] = {
                    'mean': feature_data.mean(),
                    'median': feature_data.median(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max()
                }
            
            # Generate cluster description
            profile['description'] = self._generate_cluster_description(profile, cluster_id)
            
            profiles[f'cluster_{cluster_id}'] = profile
        
        self.cluster_profiles = profiles
        return profiles
    
    def _assess_cluster_risk(self, cluster_data):
        """Assess health risk level of a cluster"""
        age_risk = 'High' if cluster_data['health_age'].mean() > 60 else ('Medium' if cluster_data['health_age'].mean() > 45 else 'Low')
        
        bp_risk = 'High' if cluster_data['health_bp'].mean() > 140 else ('Medium' if cluster_data['health_bp'].mean() > 120 else 'Low')
        
        chol_risk = 'High' if cluster_data['health_chol'].mean() > 240 else ('Medium' if cluster_data['health_chol'].mean() > 200 else 'Low')
        
        bmi_risk = 'High' if cluster_data['health_bmi'].mean() > 30 else ('Medium' if cluster_data['health_bmi'].mean() > 25 else 'Low')
        
        # Calculate overall risk
        risk_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        total_risk = sum([risk_scores[risk] for risk in [age_risk, bp_risk, chol_risk, bmi_risk]])
        
        if total_risk <= 6:
            overall_risk = 'Low'
        elif total_risk <= 9:
            overall_risk = 'Medium'
        else:
            overall_risk = 'High'
        
        return {
            'overall': overall_risk,
            'age_risk': age_risk,
            'bp_risk': bp_risk,
            'cholesterol_risk': chol_risk,
            'bmi_risk': bmi_risk,
            'risk_score': total_risk
        }
    
    def _generate_cluster_description(self, profile, cluster_id):
        """Generate human-readable cluster description"""
        characteristics = profile['characteristics']
        risk_profile = profile['risk_profile']
        
        avg_age = characteristics['health_age']['mean']
        avg_bp = characteristics['health_bp']['mean']
        avg_chol = characteristics['health_chol']['mean']
        avg_bmi = characteristics['health_bmi']['mean']
        
        description = f"Cluster {cluster_id} ({profile['percentage']:.1f}% of patients): "
        
        # Age description
        if avg_age < 30:
            age_desc = "Young adults"
        elif avg_age < 50:
            age_desc = "Middle-aged adults"
        elif avg_age < 65:
            age_desc = "Older adults"
        else:
            age_desc = "Elderly patients"
        
        # Health status description
        if risk_profile['overall'] == 'Low':
            health_desc = "with generally good health metrics"
        elif risk_profile['overall'] == 'Medium':
            health_desc = "with moderate health concerns"
        else:
            health_desc = "with significant health risks"
        
        # Specific concerns
        concerns = []
        if risk_profile['bp_risk'] == 'High':
            concerns.append("elevated blood pressure")
        if risk_profile['cholesterol_risk'] == 'High':
            concerns.append("high cholesterol")
        if risk_profile['bmi_risk'] == 'High':
            concerns.append("obesity")
        
        concern_desc = f". Main concerns: {', '.join(concerns)}" if concerns else ""
        
        return description + age_desc + " " + health_desc + concern_desc
    
    def assign_user_to_cluster(self, user_data, method='kmeans'):
        """Assign a new user to existing clusters"""
        if method not in self.clustering_models:
            return None
        
        # Prepare user data
        user_features = [
            user_data.get('age', 35),
            user_data.get('blood_pressure', 120),
            user_data.get('cholesterol', 200),
            user_data.get('bmi', 25.0)
        ]
        
        user_array = np.array(user_features).reshape(1, -1)
        user_scaled = self.scaler.transform(user_array)
        
        # Predict cluster
        model = self.clustering_models[method]['model']
        cluster_id = model.predict(user_scaled)[0]
        
        # Get cluster profile
        cluster_profile = self.cluster_profiles.get(f'cluster_{cluster_id}', {})
        
        return {
            'cluster_id': cluster_id,
            'cluster_profile': cluster_profile,
            'similarity_score': self._calculate_similarity(user_features, cluster_id)
        }
    
    def _calculate_similarity(self, user_features, cluster_id):
        """Calculate how similar user is to cluster centroid"""
        if f'cluster_{cluster_id}' not in self.cluster_profiles:
            return 0.5
        
        cluster_profile = self.cluster_profiles[f'cluster_{cluster_id}']
        feature_names = ['health_age', 'health_bp', 'health_chol', 'health_bmi']
        
        similarities = []
        for i, feature in enumerate(feature_names):
            cluster_mean = cluster_profile['characteristics'][feature]['mean']
            cluster_std = cluster_profile['characteristics'][feature]['std']
            
            if cluster_std == 0:
                similarity = 1.0 if user_features[i] == cluster_mean else 0.0
            else:
                # Calculate normalized distance
                distance = abs(user_features[i] - cluster_mean) / cluster_std
                similarity = max(0, 1 - distance / 3)  # 3-sigma rule
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def detect_user_anomalies(self, user_data):
        """Detect if user data represents an anomaly"""
        if not self.anomaly_detectors:
            return {'is_anomaly': False, 'confidence': 0.5}
        
        # Prepare user data
        user_features = np.array([
            user_data.get('age', 35),
            user_data.get('blood_pressure', 120),
            user_data.get('cholesterol', 200),
            user_data.get('bmi', 25.0)
        ]).reshape(1, -1)
        
        user_scaled = self.scaler.transform(user_features)
        
        anomaly_results = {}
        
        # Check with Isolation Forest
        if 'isolation_forest' in self.anomaly_detectors:
            iso_model = self.anomaly_detectors['isolation_forest']['model']
            iso_prediction = iso_model.predict(user_scaled)[0]
            iso_score = iso_model.score_samples(user_scaled)[0]
            
            anomaly_results['isolation_forest'] = {
                'is_anomaly': iso_prediction == -1,
                'anomaly_score': iso_score
            }
        
        # Combine results
        is_anomaly = any([result['is_anomaly'] for result in anomaly_results.values()])
        
        # Calculate confidence (normalized anomaly score)
        avg_score = np.mean([result['anomaly_score'] for result in anomaly_results.values()])
        confidence = max(0, min(1, abs(avg_score)))
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'details': anomaly_results,
            'interpretation': self._interpret_anomaly(is_anomaly, confidence, user_data)
        }
    
    def _interpret_anomaly(self, is_anomaly, confidence, user_data):
        """Interpret anomaly detection results"""
        if not is_anomaly:
            return "Your health profile is within normal ranges compared to the population."
        
        interpretations = []
        
        if confidence > 0.8:
            interpretations.append("⚠️ Your health profile shows unusual patterns that warrant attention.")
        elif confidence > 0.6:
            interpretations.append("🔍 Some of your health metrics appear atypical.")
        else:
            interpretations.append("⚪ Your health profile shows minor deviations from typical patterns.")
        
        # Specific anomaly explanations
        age = user_data.get('age', 35)
        bp = user_data.get('blood_pressure', 120)
        chol = user_data.get('cholesterol', 200)
        bmi = user_data.get('bmi', 25)
        
        if age < 25 and (bp > 140 or chol > 240):
            interpretations.append("🏥 High blood pressure or cholesterol at young age is unusual.")
        
        if age > 65 and bp < 90:
            interpretations.append("💊 Unusually low blood pressure for your age group.")
        
        if bmi < 16 or bmi > 45:
            interpretations.append("⚖️ Extreme BMI values detected.")
        
        interpretations.append("👨‍⚕️ Consider consulting with a healthcare professional for evaluation.")
        
        return " ".join(interpretations)
    
    def save_clustering_models(self, filepath='models/clustering/'):
        """Save clustering models and results"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        joblib.dump(self.clustering_models, f'{filepath}clustering_models.pkl')
        joblib.dump(self.dimensionality_reducers, f'{filepath}dimensionality_reducers.pkl')
        joblib.dump(self.anomaly_detectors, f'{filepath}anomaly_detectors.pkl')
        joblib.dump(self.cluster_profiles, f'{filepath}cluster_profiles.pkl')
        joblib.dump(self.scaler, f'{filepath}scaler.pkl')
        
        print(f"💾 Clustering models saved to {filepath}")
    
    def load_clustering_models(self, filepath='models/clustering/'):
        """Load clustering models and results"""
        try:
            self.clustering_models = joblib.load(f'{filepath}clustering_models.pkl')
            self.dimensionality_reducers = joblib.load(f'{filepath}dimensionality_reducers.pkl')
            self.anomaly_detectors = joblib.load(f'{filepath}anomaly_detectors.pkl')
            self.cluster_profiles = joblib.load(f'{filepath}cluster_profiles.pkl')
            self.scaler = joblib.load(f'{filepath}scaler.pkl')
            print(f"📁 Clustering models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading clustering models: {e}")
            return False