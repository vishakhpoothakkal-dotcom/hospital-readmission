"""
Clinical Data Validation Engine for Hospital Readmission Prediction System
Enforces strict medical-grade validation on both individual patient inputs
and batch dataset uploads. Ensures analysis CANNOT proceed if any field is missing or invalid.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'age',
    'gender',
    'admission_type',
    'admission_source',
    'length_of_stay',
    'num_inpatient_visits_prior_year',
    'num_emergency_visits_prior_year',
    'num_outpatient_visits_prior_year',
    'primary_diagnosis',
    'charlson_comorbidity_index',
    'num_medications',
    'num_lab_procedures',
    'hemoglobin_g_dl',
    'creatinine_mg_dl',
    'blood_glucose_mg_dl',
    'systolic_bp',
    'oxygen_saturation',
    'high_risk_medication',
    'discharge_disposition'
]

FIELD_LABELS = {
    'age': 'Patient Age (years)',
    'gender': 'Patient Gender',
    'admission_type': 'Admission Type',
    'admission_source': 'Admission Source',
    'length_of_stay': 'Length of Stay (days)',
    'num_inpatient_visits_prior_year': 'Prior Inpatient Admissions (1 yr)',
    'num_emergency_visits_prior_year': 'Prior Emergency Visits (1 yr)',
    'num_outpatient_visits_prior_year': 'Prior Outpatient Visits (1 yr)',
    'primary_diagnosis': 'Primary Diagnosis Department',
    'charlson_comorbidity_index': 'Charlson Comorbidity Index',
    'num_medications': 'Number of Prescribed Medications',
    'num_lab_procedures': 'Number of Lab Procedures Performed',
    'hemoglobin_g_dl': 'Hemoglobin Level (g/dL)',
    'creatinine_mg_dl': 'Serum Creatinine (mg/dL)',
    'blood_glucose_mg_dl': 'Blood Glucose (mg/dL)',
    'systolic_bp': 'Systolic Blood Pressure (mmHg)',
    'oxygen_saturation': 'Oxygen Saturation SpO2 (%)',
    'high_risk_medication': 'High-Risk Medication Prescribed',
    'discharge_disposition': 'Discharge Disposition'
}

PHYSIOLOGICAL_RANGES = {
    'age': (18, 115),
    'length_of_stay': (1, 90),
    'num_inpatient_visits_prior_year': (0, 30),
    'num_emergency_visits_prior_year': (0, 30),
    'num_outpatient_visits_prior_year': (0, 50),
    'charlson_comorbidity_index': (0, 20),
    'num_medications': (1, 70),
    'num_lab_procedures': (1, 200),
    'hemoglobin_g_dl': (4.0, 24.0),
    'creatinine_mg_dl': (0.2, 20.0),
    'blood_glucose_mg_dl': (30, 800),
    'systolic_bp': (50, 260),
    'oxygen_saturation': (50, 100)
}

ALLOWED_CATEGORIES = {
    'gender': ['Male', 'Female', 'Other'],
    'admission_type': ['Emergency', 'Urgent', 'Elective'],
    'admission_source': ['Emergency Department', 'Physician Referral', 'Transfer from Hospital', 'Clinic Referral'],
    'primary_diagnosis': [
        'Cardiology (Heart Failure/AMI)',
        'Pulmonology (COPD/Pneumonia)',
        'Endocrinology (Diabetes)',
        'Nephrology (Kidney Disease)',
        'Neurology (Stroke/TIA)',
        'Gastroenterology',
        'Oncology',
        'General Surgery'
    ],
    'high_risk_medication': ['Yes', 'No'],
    'discharge_disposition': [
        'Home (Routine)',
        'Home Health Service',
        'Skilled Nursing Facility (SNF)',
        'Long-Term Care Hospital'
    ]
}

def validate_individual_patient(patient_dict: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Strict validation of single patient profile.
    Returns:
        is_valid: bool (True only if EVERY single required field is present and within range)
        errors: List of specific clinical error strings
        cleaned_data: Sanitized dictionary ready for model prediction
    """
    errors = []
    cleaned_data = {}
    
    # Check for missing keys or empty/None values
    for col in REQUIRED_COLUMNS:
        friendly_label = FIELD_LABELS.get(col, col)
        
        if col not in patient_dict:
            errors.append(f"Missing Field: '{friendly_label}' is absent from submission.")
            continue
            
        val = patient_dict[col]
        
        # Check None, NaN, empty strings, or unselected placeholders
        if val is None or (isinstance(val, (str, list)) and len(str(val).strip()) == 0) or str(val).strip() in ['Select...', '-- Select --', 'None']:
            errors.append(f"Incomplete Field: '{friendly_label}' must be specified. Cannot be left blank.")
            continue
            
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            errors.append(f"Incomplete Field: '{friendly_label}' has invalid NaN value.")
            continue
            
        # Range checks for numerical fields
        if col in PHYSIOLOGICAL_RANGES:
            min_val, max_val = PHYSIOLOGICAL_RANGES[col]
            try:
                num_val = float(val)
                if num_val < min_val or num_val > max_val:
                    errors.append(
                        f"Out-of-Range Clinical Value: '{friendly_label}' = {val} is outside valid physiological boundary [{min_val} to {max_val}]."
                    )
                else:
                    # Cast integer types
                    if col in ['age', 'length_of_stay', 'num_inpatient_visits_prior_year',
                               'num_emergency_visits_prior_year', 'num_outpatient_visits_prior_year',
                               'charlson_comorbidity_index', 'num_medications', 'num_lab_procedures',
                               'blood_glucose_mg_dl', 'systolic_bp', 'oxygen_saturation']:
                        cleaned_data[col] = int(num_val)
                    else:
                        cleaned_data[col] = round(num_val, 2)
            except (ValueError, TypeError):
                errors.append(f"Data Type Error: '{friendly_label}' must be a numerical value, got '{val}'.")
                
        # Category checks
        elif col in ALLOWED_CATEGORIES:
            str_val = str(val).strip()
            if str_val not in ALLOWED_CATEGORIES[col]:
                errors.append(
                    f"Invalid Selection: '{friendly_label}' = '{str_val}' is not an authorized clinical category."
                )
            else:
                cleaned_data[col] = str_val
                
    is_valid = (len(errors) == 0) and (len(cleaned_data) == len(REQUIRED_COLUMNS))
    return is_valid, errors, cleaned_data

def validate_batch_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Strict batch dataset validation.
    Checks column presence, missing values per row, and invalid records.
    """
    report = {
        'total_rows': len(df),
        'missing_columns': [],
        'missing_values_per_col': {},
        'invalid_rows_count': 0,
        'valid_rows_count': 0,
        'row_errors': [],  # list of {row_index, patient_id, errors}
        'is_perfect': False,
        'valid_df': None,
        'invalid_df': None
    }
    
    # 1. Column check
    df_cols = [c.lower().strip() for c in df.columns]
    missing_cols = [col for col in REQUIRED_COLUMNS if col.lower() not in df_cols]
    if missing_cols:
        report['missing_columns'] = [FIELD_LABELS.get(c, c) for c in missing_cols]
        return report

    # 2. Check missing values per column
    for col in REQUIRED_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report['missing_values_per_col'][FIELD_LABELS.get(col, col)] = null_count
            
    # 3. Row-by-row validation
    valid_indices = []
    invalid_indices = []
    
    for idx, row in df.iterrows():
        p_dict = row.to_dict()
        is_v, row_errs, _ = validate_individual_patient(p_dict)
        pid = row.get('patient_id', f'Row {idx + 1}')
        if is_v:
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)
            if len(report['row_errors']) < 25: # keep preview compact
                report['row_errors'].append({
                    'row_idx': int(idx + 1),
                    'patient_id': str(pid),
                    'errors': row_errs
                })
                
    report['valid_rows_count'] = len(valid_indices)
    report['invalid_rows_count'] = len(invalid_indices)
    report['is_perfect'] = (len(invalid_indices) == 0 and len(report['missing_columns']) == 0)
    
    if len(valid_indices) > 0:
        report['valid_df'] = df.loc[valid_indices].copy()
    if len(invalid_indices) > 0:
        report['invalid_df'] = df.loc[invalid_indices].copy()
        
    return report

if __name__ == '__main__':
    # Test valid case
    test_patient = {
        'age': 65,
        'gender': 'Male',
        'admission_type': 'Emergency',
        'admission_source': 'Emergency Department',
        'length_of_stay': 4,
        'num_inpatient_visits_prior_year': 1,
        'num_emergency_visits_prior_year': 0,
        'num_outpatient_visits_prior_year': 2,
        'primary_diagnosis': 'Cardiology (Heart Failure/AMI)',
        'charlson_comorbidity_index': 2,
        'num_medications': 12,
        'num_lab_procedures': 35,
        'hemoglobin_g_dl': 13.5,
        'creatinine_mg_dl': 1.1,
        'blood_glucose_mg_dl': 130,
        'systolic_bp': 128,
        'oxygen_saturation': 97,
        'high_risk_medication': 'No',
        'discharge_disposition': 'Home (Routine)'
    }
    
    v_ok, errs, _ = validate_individual_patient(test_patient)
    print("Full patient valid:", v_ok, "Errors:", errs)
    
    # Test missing field case
    test_missing = test_patient.copy()
    del test_missing['blood_glucose_mg_dl']
    test_missing['length_of_stay'] = None
    v_bad, errs_bad, _ = validate_individual_patient(test_missing)
    print("Missing fields valid:", v_bad, "Errors count:", len(errs_bad))
    for e in errs_bad:
        print(" ->", e)
