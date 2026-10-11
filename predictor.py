"""
Clinical Prediction and Risk Stratification Engine
Provides inference, risk scoring, feature attribution, and personalized clinical discharge planning.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from validator import validate_individual_patient, validate_batch_dataframe, REQUIRED_COLUMNS

class HospitalPredictor:
    def __init__(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is None:
            artifacts_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.artifacts_path = os.path.join(artifacts_dir, 'model_artifacts.joblib')
        self.summary_path = os.path.join(artifacts_dir, 'evaluation_summary.json')
        
        if not os.path.exists(self.artifacts_path):
            raise FileNotFoundError(f"Model artifacts not found at {self.artifacts_path}. Run train_model.py first.")
            
        self.artifacts = joblib.load(self.artifacts_path)
        self.models = self.artifacts['models']
        self.preprocessor = self.artifacts['preprocessor']
        self.cat_features = self.artifacts['cat_features']
        self.num_features = self.artifacts['num_features']
        self.all_feature_names = self.artifacts['all_feature_names']
        self.default_model_name = self.artifacts.get('default_model_name', 'XGBoost')
        
        with open(self.summary_path, 'r') as f:
            self.evaluation_summary = json.load(f)
            
    def get_available_models(self) -> List[str]:
        return list(self.models.keys())
        
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        return self.evaluation_summary['metrics'].get(model_name, {})

    def predict_single(self, patient_dict: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validates and scores a single patient profile.
        Strict validation: fails immediately if any required field is missing.
        """
        is_valid, errors, cleaned = validate_individual_patient(patient_dict)
        if not is_valid:
            return {
                'success': False,
                'errors': errors,
                'message': f"Validation Failed: {len(errors)} required field(s) are missing or out of valid clinical bounds."
            }
            
        active_model_name = model_name if (model_name and model_name in self.models) else self.default_model_name
        model = self.models[active_model_name]
        
        # Prepare DataFrame
        df_patient = pd.DataFrame([cleaned])[self.num_features + self.cat_features]
        X_proc = self.preprocessor.transform(df_patient)
        
        prob = float(model.predict_proba(X_proc)[0, 1])
        prediction = int(prob >= 0.50)
        
        # Risk Stratification
        if prob < 0.30:
            risk_tier = 'Low Risk'
            risk_badge = '🟢 LOW READMISSION RISK'
            risk_color = '#10B981' # emerald green
        elif prob < 0.65:
            risk_tier = 'Moderate Risk'
            risk_badge = '🟡 MODERATE READMISSION RISK'
            risk_color = '#F59E0B' # amber
        else:
            risk_tier = 'High Risk'
            risk_badge = '🔴 HIGH READMISSION RISK'
            risk_color = '#EF4444' # rose red
            
        # Clinical Risk Factors (Attribution)
        risk_factors = []
        positive_factors = []
        
        if cleaned['length_of_stay'] >= 6:
            risk_factors.append({
                'factor': 'Prolonged Length of Stay',
                'value': f"{cleaned['length_of_stay']} days",
                'impact': 'High',
                'explanation': 'Hospital stays > 5 days are heavily associated with clinical frailty and post-discharge complications.'
            })
            
        if cleaned['num_inpatient_visits_prior_year'] >= 1:
            risk_factors.append({
                'factor': 'Frequent Inpatient Re-admissions',
                'value': f"{cleaned['num_inpatient_visits_prior_year']} prior admission(s)",
                'impact': 'Critical',
                'explanation': 'Past-year hospitalizations are the primary driver of the LACE index.'
            })
            
        if cleaned['num_emergency_visits_prior_year'] >= 1:
            risk_factors.append({
                'factor': 'Emergency Department Utilization',
                'value': f"{cleaned['num_emergency_visits_prior_year']} ED visit(s)",
                'impact': 'High',
                'explanation': 'Frequent ED visits indicate unmanaged acute symptom exacerbations.'
            })
            
        if cleaned['charlson_comorbidity_index'] >= 3:
            risk_factors.append({
                'factor': 'High Chronic Comorbidity Burden',
                'value': f"CCI Score {cleaned['charlson_comorbidity_index']}",
                'impact': 'High',
                'explanation': 'Multiple chronic illnesses compound physiological vulnerability post-discharge.'
            })
            
        if cleaned['creatinine_mg_dl'] > 1.4:
            risk_factors.append({
                'factor': 'Elevated Serum Creatinine',
                'value': f"{cleaned['creatinine_mg_dl']} mg/dL",
                'impact': 'Moderate',
                'explanation': 'Impaired renal filtration increases medication toxicity and metabolic stress.'
            })
            
        if cleaned['hemoglobin_g_dl'] < 10.5:
            risk_factors.append({
                'factor': 'Significant Anemia',
                'value': f"{cleaned['hemoglobin_g_dl']} g/dL",
                'impact': 'Moderate',
                'explanation': 'Low oxygen carrying capacity impedes physical recovery.'
            })
            
        if cleaned['oxygen_saturation'] < 93:
            risk_factors.append({
                'factor': 'Sub-optimal SpO2 Oxygenation',
                'value': f"{cleaned['oxygen_saturation']}%",
                'impact': 'High',
                'explanation': 'Hypoxemia indicates active respiratory or cardiac insufficiency.'
            })
            
        if cleaned['high_risk_medication'] == 'Yes':
            risk_factors.append({
                'factor': 'High-Risk Pharmacotherapy',
                'value': 'Active (Anticoagulant / Insulin / Opioid)',
                'impact': 'Moderate',
                'explanation': 'High-alert medications require strict post-discharge titration and adherence.'
            })

        # Protective/Safe factors
        if cleaned['num_inpatient_visits_prior_year'] == 0:
            positive_factors.append('No previous hospital admissions in past 12 months')
        if cleaned['length_of_stay'] <= 3:
            positive_factors.append('Short acute hospital stay (≤ 3 days)')
        if cleaned['creatinine_mg_dl'] <= 1.2:
            positive_factors.append('Normal kidney function (Serum Creatinine in target range)')
        if cleaned['oxygen_saturation'] >= 96:
            positive_factors.append('Stable respiratory status (SpO2 ≥ 96%)')

        # Personalized Clinical Discharge Recommendations
        recommendations = []
        if risk_tier == 'High Risk':
            recommendations.append('📅 Schedule mandatory Outpatient Primary Care / Specialist visit within 48-72 hours.')
            recommendations.append('📞 Assign Clinical Nurse Navigator for a follow-up telehealth call within 24-48 hours.')
            recommendations.append('💊 Perform comprehensive Pharmacist-led Medication Reconciliation prior to discharge.')
            if cleaned['discharge_disposition'] in ['Home (Routine)', 'Home Health Service']:
                recommendations.append('🏡 Request in-home physical therapy & skilled nursing health assessment.')
        elif risk_tier == 'Moderate Risk':
            recommendations.append('📅 Schedule standard clinic follow-up appointment within 7-10 days.')
            recommendations.append('📞 Post-discharge welfare call at Day 3 to review symptom control and medication intake.')
            recommendations.append('📋 Provide patient-friendly bilingual discharge summary and red-flag symptom warnings.')
        else:
            recommendations.append('📅 Standard routine follow-up with primary care physician in 14-21 days.')
            recommendations.append('✅ Patient cleared for standard discharge pathway with routine medication instructions.')

        # Biomarker radar baseline comparison
        radar_metrics = {
            'Length of Stay (Days)': {'patient': cleaned['length_of_stay'], 'normal': 3.0, 'max': 15.0},
            'Prior Inpatient Visits': {'patient': cleaned['num_inpatient_visits_prior_year'], 'normal': 0.0, 'max': 5.0},
            'Comorbidity Index (CCI)': {'patient': cleaned['charlson_comorbidity_index'], 'normal': 1.0, 'max': 6.0},
            'Creatinine (mg/dL)': {'patient': cleaned['creatinine_mg_dl'], 'normal': 0.9, 'max': 3.0},
            'Medication Count': {'patient': cleaned['num_medications'], 'normal': 10.0, 'max': 30.0}
        }
        
        return {
            'success': True,
            'model_used': active_model_name,
            'probability': prob,
            'probability_pct': round(prob * 100, 1),
            'prediction': prediction,
            'prediction_label': 'Likely Readmission (< 30 Days)' if prediction == 1 else 'Low Readmission Risk (Safe Discharge)',
            'risk_tier': risk_tier,
            'risk_badge': risk_badge,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'positive_factors': positive_factors,
            'recommendations': recommendations,
            'cleaned_patient': cleaned,
            'radar_metrics': radar_metrics
        }

    def predict_batch(self, df: pd.DataFrame, model_name: Optional[str] = None, strict_mode: bool = True) -> Dict[str, Any]:
        """
        Validates and processes batch patient data.
        If strict_mode is True and any field is missing or invalid, analysis is rejected.
        """
        validation_report = validate_batch_dataframe(df)
        
        if validation_report['missing_columns']:
            return {
                'success': False,
                'validation_report': validation_report,
                'message': f"Batch Rejected: The uploaded dataset is missing {len(validation_report['missing_columns'])} required clinical columns."
            }
            
        if strict_mode and not validation_report['is_perfect']:
            return {
                'success': False,
                'validation_report': validation_report,
                'message': f"Batch Rejected in Strict Mode: Found {validation_report['invalid_rows_count']} rows with missing/out-of-range clinical fields. Every field must be complete."
            }
            
        # Target df to score
        scoring_df = validation_report['valid_df'] if not strict_mode else df.copy()
        
        if scoring_df is None or len(scoring_df) == 0:
            return {
                'success': False,
                'validation_report': validation_report,
                'message': "No valid patient records available for batch analysis."
            }
            
        active_model_name = model_name if (model_name and model_name in self.models) else self.default_model_name
        model = self.models[active_model_name]
        
        # Preprocess
        X_df = scoring_df[self.num_features + self.cat_features]
        X_proc = self.preprocessor.transform(X_df)
        
        probabilities = model.predict_proba(X_proc)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        
        # Risk tier tagging
        risk_tiers = []
        action_priorities = []
        for p in probabilities:
            if p < 0.30:
                risk_tiers.append('Low Risk')
                action_priorities.append('Standard Discharge')
            elif p < 0.65:
                risk_tiers.append('Moderate Risk')
                action_priorities.append('Care Coordination')
            else:
                risk_tiers.append('High Risk')
                action_priorities.append('Immediate Intervention')
                
        results_df = scoring_df.copy()
        results_df['Readmission_Probability_%'] = np.round(probabilities * 100, 1)
        results_df['Predicted_30d_Readmit'] = predictions
        results_df['Risk_Tier'] = risk_tiers
        results_df['Triage_Action'] = action_priorities
        
        # Aggregate dashboard summary metrics
        total = len(results_df)
        high_risk_count = int((results_df['Risk_Tier'] == 'High Risk').sum())
        mod_risk_count = int((results_df['Risk_Tier'] == 'Moderate Risk').sum())
        low_risk_count = int((results_df['Risk_Tier'] == 'Low Risk').sum())
        pred_readmit_count = int((results_df['Predicted_30d_Readmit'] == 1).sum())
        
        return {
            'success': True,
            'model_used': active_model_name,
            'validation_report': validation_report,
            'results_df': results_df,
            'summary': {
                'total_patients': total,
                'predicted_readmissions': pred_readmit_count,
                'predicted_readmission_rate': round((pred_readmit_count / total) * 100, 1),
                'high_risk_count': high_risk_count,
                'high_risk_pct': round((high_risk_count / total) * 100, 1),
                'moderate_risk_count': mod_risk_count,
                'moderate_risk_pct': round((mod_risk_count / total) * 100, 1),
                'low_risk_count': low_risk_count,
                'low_risk_pct': round((low_risk_count / total) * 100, 1)
            }
        }
