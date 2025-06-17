import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

def create_risk_gauge(risk_percentage, title="Health Risk"):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'lightgreen'},
                {'range': [25, 50], 'color': 'yellow'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_feature_importance_chart(feature_data, title="Feature Importance"):
    """Create horizontal bar chart for feature importance"""
    if not feature_data:
        return None
    
    features = [item['feature'] for item in feature_data]
    importances = [item['magnitude'] if 'magnitude' in item else abs(item['importance']) for item in feature_data]
    colors = ['red' if item.get('importance', 0) > 0 else 'green' for item in feature_data]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features[::-1],  # Reverse for better visualization
            x=importances[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{imp:.3f}" for imp in importances[::-1]],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(300, len(features) * 30),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_cluster_visualization(cluster_data, user_cluster=None):
    """Create 2D visualization of patient clusters"""
    if 'pca' not in cluster_data:
        return None
    
    pca_data = cluster_data['pca']['data']
    labels = cluster_data.get('labels', np.zeros(len(pca_data)))
    
    # Create base scatter plot
    fig = px.scatter(
        x=pca_data[:, 0], 
        y=pca_data[:, 1],
        color=labels.astype(str),
        title="Patient Health Clusters",
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    
    # Highlight user cluster if provided
    if user_cluster is not None:
        user_points = pca_data[labels == user_cluster]
        fig.add_trace(go.Scatter(
            x=user_points[:, 0],
            y=user_points[:, 1],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='Your Cluster',
            showlegend=True
        ))
    
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_health_dashboard(risk_results):
    """Create comprehensive health dashboard"""
    if not risk_results:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Heart Disease Risk', 'Diabetes Risk', 'Stroke Risk', 'Overall Health Score'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Add risk gauges
    diseases = ['heart', 'diabetes', 'stroke']
    positions = [(1, 1), (1, 2), (2, 1)]
    
    for disease, (row, col) in zip(diseases, positions):
        if disease in risk_results and 'risk_percentage' in risk_results[disease]:
            risk_pct = risk_results[disease]['risk_percentage']
            
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = risk_pct,
                title = {'text': f"{disease.title()} Risk"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': 'lightgreen'},
                        {'range': [25, 50], 'color': 'yellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ), row=row, col=col)
    
    # Add overall health score
    if 'overall_health_score' in risk_results:
        health_score = risk_results['overall_health_score']['score']
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 25], 'color': 'red'},
                    {'range': [25, 50], 'color': 'orange'},
                    {'range': [50, 75], 'color': 'yellow'},
                    {'range': [75, 100], 'color': 'lightgreen'}
                ],
            }
        ), row=2, col=2)
    
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_comparison_chart(user_risks, population_avg):
    """Create comparison chart between user risks and population average"""
    categories = list(user_risks.keys())
    user_values = [user_risks[cat]['risk_percentage'] for cat in categories if 'risk_percentage' in user_risks[cat]]
    pop_values = [population_avg.get(cat, 30) for cat in categories]  # Default 30% population average
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Risk',
        x=categories,
        y=user_values,
        marker_color='red'
    ))
    
    fig.add_trace(go.Bar(
        name='Population Average',
        x=categories,
        y=pop_values,
        marker_color='blue'
    ))
    
    fig.update_layout(
        title='Your Risk vs Population Average',
        xaxis_title='Health Conditions',
        yaxis_title='Risk Percentage',
        barmode='group',
        height=400
    )
    
    return fig

def generate_health_report(user_data, risk_results, cluster_info=None, explanations=None):
    """Generate comprehensive health report"""
    report = []
    
    # Header
    report.append("# 🏥 Comprehensive Health Risk Analysis Report")
    report.append(f"**Generated for:** {user_data.get('name', 'Patient')}")
    report.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## 📋 Executive Summary")
    
    if 'overall_health_score' in risk_results:
        health_score = risk_results['overall_health_score']
        report.append(f"**Overall Health Score:** {health_score['score']}/100 ({health_score['level']})")
        report.append(f"**Average Risk Level:** {health_score['average_risk']}%")
    
    # Individual Risk Assessments
    report.append("\n## 🎯 Individual Risk Assessments")
    
    for disease, results in risk_results.items():
        if disease == 'overall_health_score':
            continue
            
        if 'risk_percentage' in results:
            risk_pct = results['risk_percentage']
            risk_level = results['risk_level']
            confidence = results.get('model_confidence', 'N/A')
            
            report.append(f"\n### {disease.title()} Risk")
            report.append(f"- **Risk Level:** {risk_level} ({risk_pct:.1f}%)")
            report.append(f"- **Model Confidence:** {confidence:.1f}%" if confidence != 'N/A' else "")
            
            # Recommendations
            if 'recommendations' in results:
                report.append("- **Key Recommendations:**")
                for rec in results['recommendations'][:5]:
                    report.append(f"  - {rec}")
    
    # Patient Cluster Analysis
    if cluster_info:
        report.append("\n## 👥 Patient Profile Analysis")
        cluster_profile = cluster_info.get('cluster_profile', {})
        if cluster_profile:
            report.append(f"**Patient Group:** {cluster_profile.get('description', 'N/A')}")
            report.append(f"**Similarity Score:** {cluster_info.get('similarity_score', 0):.2f}")
            
            risk_profile = cluster_profile.get('risk_profile', {})
            if risk_profile:
                report.append(f"**Group Risk Level:** {risk_profile.get('overall', 'N/A')}")
    
    # Model Explanations
    if explanations:
        report.append("\n## 🔍 Model Explanation")
        if 'unified_explanation' in explanations:
            report.append(explanations['unified_explanation'])
    
    # Lifestyle Recommendations
    report.append("\n## 💡 Personalized Lifestyle Recommendations")
    
    age = user_data.get('age', 35)
    bmi = user_data.get('bmi', 25)
    smoking = user_data.get('smoking', 'never smoked')
    
    # General recommendations based on user profile
    if age > 50:
        report.append("- 🏥 Schedule regular health check-ups (every 6 months)")
    if bmi > 30:
        report.append("- ⚖️ Consider weight management program")
    if smoking == 'smokes':
        report.append("- 🚭 Smoking cessation is critical for health improvement")
    
    report.append("- 🥗 Maintain a balanced, heart-healthy diet")
    report.append("- 🏃‍♀️ Engage in regular physical activity (150 min/week)")
    report.append("- 😴 Prioritize quality sleep (7-9 hours nightly)")
    report.append("- 🧘‍♀️ Practice stress management techniques")
    
    # Disclaimer
    report.append("\n---")
    report.append("\n## ⚠️ Important Disclaimer")
    report.append("This analysis is for informational purposes only and should not replace professional medical advice. ")
    report.append("Please consult with qualified healthcare professionals for medical decisions.")
    
    return "\n".join(report)

def validate_user_input(user_data):
    """Validate user input data"""
    errors = []
    
    # Age validation
    age = user_data.get('age')
    if age is None or age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years")
    
    # BMI validation
    bmi = user_data.get('bmi')
    if bmi is None or bmi < 10 or bmi > 80:
        errors.append("BMI must be between 10 and 80")
    
    # Blood pressure validation
    bp = user_data.get('blood_pressure')
    if bp is not None and (bp < 60 or bp > 250):
        errors.append("Blood pressure must be between 60 and 250 mmHg")
    
    # Cholesterol validation
    chol = user_data.get('cholesterol')
    if chol is not None and (chol < 100 or chol > 500):
        errors.append("Cholesterol must be between 100 and 500 mg/dL")
    
    # Glucose validation
    glucose = user_data.get('glucose')
    if glucose is not None and (glucose < 50 or glucose > 400):
        errors.append("Glucose level must be between 50 and 400 mg/dL")
    
    return errors

def format_risk_level_color(risk_level):
    """Return color code for risk level"""
    colors = {
        'Low': 'green',
        'Moderate': 'orange',
        'High': 'red',
        'Very High': 'darkred'
    }
    return colors.get(risk_level, 'gray')

def create_health_timeline(user_data, predictions):
    """Create timeline showing health risk progression"""
    # This is a mock timeline - in real application, would use historical data
    ages = list(range(max(20, user_data.get('age', 35) - 10), min(90, user_data.get('age', 35) + 20), 5))
    
    # Mock progression based on current risk and age
    current_age = user_data.get('age', 35)
    
    fig = go.Figure()
    
    for disease, results in predictions.items():
        if disease == 'overall_health_score':
            continue
            
        if 'risk_percentage' in results:
            current_risk = results['risk_percentage']
            
            # Project risk progression (simplified model)
            progression = []
            for age in ages:
                age_factor = (age - current_age) * 0.5  # Risk increases with age
                projected_risk = min(100, max(0, current_risk + age_factor))
                progression.append(projected_risk)
            
            fig.add_trace(go.Scatter(
                x=ages,
                y=progression,
                mode='lines+markers',
                name=f'{disease.title()} Risk',
                line=dict(width=3)
            ))
    
    fig.update_layout(
        title='Projected Health Risk Timeline',
        xaxis_title='Age (years)',
        yaxis_title='Risk Percentage',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def calculate_health_improvement_potential(user_data, risk_results):
    """Calculate potential health improvement with lifestyle changes"""
    improvements = {}
    
    for disease, results in risk_results.items():
        if disease == 'overall_health_score' or 'risk_percentage' not in results:
            continue
        
        current_risk = results['risk_percentage']
        potential_reduction = 0
        
        # Calculate reduction potential based on modifiable risk factors
        if user_data.get('smoking') == 'smokes':
            potential_reduction += 15  # Smoking cessation
        
        if user_data.get('bmi', 25) > 30:
            potential_reduction += 10  # Weight loss
        
        if user_data.get('exercise') == 'Never':
            potential_reduction += 8   # Regular exercise
        
        if user_data.get('blood_pressure', 120) > 140:
            potential_reduction += 12  # BP control
        
        improved_risk = max(0, current_risk - potential_reduction)
        
        improvements[disease] = {
            'current_risk': current_risk,
            'improved_risk': improved_risk,
            'potential_reduction': potential_reduction,
            'improvement_percentage': (potential_reduction / current_risk * 100) if current_risk > 0 else 0
        }
    
    return improvements

def create_improvement_potential_chart(improvements):
    """Create chart showing health improvement potential"""
    diseases = list(improvements.keys())
    current_risks = [improvements[d]['current_risk'] for d in diseases]
    improved_risks = [improvements[d]['improved_risk'] for d in diseases]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Risk',
        x=diseases,
        y=current_risks,
        marker_color='red',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name='Potential Risk (with improvements)',
        x=diseases,
        y=improved_risks,
        marker_color='green',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Health Improvement Potential',
        xaxis_title='Health Conditions',
        yaxis_title='Risk Percentage',
        barmode='group',
        height=400
    )
    
    return fig