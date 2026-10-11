"""
Model Training and Evaluation Pipeline for Hospital Readmission Prediction System
Trains multiple clinical predictive models (XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression),
benchmarks them, and stores all models and metrics for multi-model comparison in the UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def train_and_evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'hospital_readmissions.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run data_generator.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    cat_features = [
        'gender', 'admission_type', 'admission_source',
        'primary_diagnosis', 'high_risk_medication', 'discharge_disposition'
    ]
    
    num_features = [
        'age', 'length_of_stay', 'num_inpatient_visits_prior_year',
        'num_emergency_visits_prior_year', 'num_outpatient_visits_prior_year',
        'charlson_comorbidity_index', 'num_medications', 'num_lab_procedures',
        'hemoglobin_g_dl', 'creatinine_mg_dl', 'blood_glucose_mg_dl',
        'systolic_bp', 'oxygen_saturation'
    ]
    
    X = df[num_features + cat_features]
    y = df['readmitted_30d']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )
    
    preprocessor.fit(X_train)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names
    
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
            verbose=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    fitted_models = {}
    feature_importances_dict = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_proc, y_train)
        fitted_models[name] = model
        
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        stride = max(1, len(fpr) // 50)
        roc_pts = [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])} for i in range(0, len(fpr), stride)]
        if roc_pts[-1]['fpr'] < 1.0:
            roc_pts.append({'fpr': 1.0, 'tpr': 1.0})
            
        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm,
            'roc_curve': roc_pts
        }
        
        print(f" -> {name}: Accuracy = {acc*100:.2f}%, ROC-AUC = {auc:.4f}, F1 = {f1:.4f}")
        
        # Calculate feature importances / coefficients
        feat_list = []
        if hasattr(model, 'feature_importances_'):
            vals = model.feature_importances_
            sorted_idx = np.argsort(vals)[::-1]
            for idx in sorted_idx[:15]:
                raw = all_feature_names[idx]
                clean = raw.replace('num_', '').replace('_prior_year', '').replace('_', ' ').title()
                feat_list.append({
                    'feature': clean,
                    'raw_feature': raw,
                    'importance': float(vals[idx])
                })
        elif hasattr(model, 'coef_'):
            vals = np.abs(model.coef_[0])
            sorted_idx = np.argsort(vals)[::-1]
            for idx in sorted_idx[:15]:
                raw = all_feature_names[idx]
                clean = raw.replace('num_', '').replace('_prior_year', '').replace('_', ' ').title()
                feat_list.append({
                    'feature': clean,
                    'raw_feature': raw,
                    'importance': float(vals[idx])
                })
        feature_importances_dict[name] = feat_list

    # Save artifacts
    artifacts = {
        'models': fitted_models,
        'default_model_name': 'XGBoost',
        'preprocessor': preprocessor,
        'cat_features': cat_features,
        'num_features': num_features,
        'all_feature_names': all_feature_names,
        'target_classes': ['Not Readmitted', 'Readmitted within 30 Days']
    }
    
    artifacts_path = os.path.join(script_dir, 'model_artifacts.joblib')
    joblib.dump(artifacts, artifacts_path)
    print(f"Saved multi-model pipeline artifacts to {artifacts_path}")
    
    summary_data = {
        'default_model': 'XGBoost',
        'metrics': results,
        'feature_importances': feature_importances_dict,
        'dataset_summary': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(num_features) + len(cat_features),
            'readmission_rate': float(df['readmitted_30d'].mean())
        }
    }
    
    summary_path = os.path.join(script_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation metrics to {summary_path}")

if __name__ == '__main__':
    train_and_evaluate()
