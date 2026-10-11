"""
Dataset Generator for Hospital 30-Day Readmission Prediction System
Generates high-fidelity clinical records reflecting real-world hospital dynamics
(LACE index, Charlson Comorbidity, lab biomarkers, medication counts, and prior admissions).
"""

import numpy as np
import pandas as pd
import os

def generate_hospital_data(n_samples=12000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Demographics
    age = np.random.normal(63, 14, n_samples).astype(int)
    age = np.clip(age, 18, 95)
    
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.49, 0.51])
    
    admission_type = np.random.choice(
        ['Emergency', 'Urgent', 'Elective'],
        size=n_samples,
        p=[0.55, 0.25, 0.20]
    )
    
    admission_source = np.random.choice(
        ['Emergency Department', 'Physician Referral', 'Transfer from Hospital', 'Clinic Referral'],
        size=n_samples,
        p=[0.52, 0.24, 0.14, 0.10]
    )
    
    # 2. Clinical and Utilization metrics
    # Length of stay is skewed
    length_of_stay = np.random.gamma(shape=2.5, scale=2.0, size=n_samples).astype(int) + 1
    length_of_stay = np.clip(length_of_stay, 1, 28)
    
    # Prior visits in past 12 months
    num_inpatient_visits = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7],
        size=n_samples,
        p=[0.48, 0.22, 0.12, 0.08, 0.05, 0.03, 0.015, 0.005]
    )
    
    num_emergency_visits = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=n_samples,
        p=[0.55, 0.22, 0.11, 0.06, 0.04, 0.02]
    )
    
    num_outpatient_visits = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7],
        size=n_samples,
        p=[0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.02, 0.02]
    )
    
    # Diagnoses
    diagnoses = [
        'Cardiology (Heart Failure/AMI)',
        'Pulmonology (COPD/Pneumonia)',
        'Endocrinology (Diabetes)',
        'Nephrology (Kidney Disease)',
        'Neurology (Stroke/TIA)',
        'Gastroenterology',
        'Oncology',
        'General Surgery'
    ]
    diag_weights = [0.24, 0.20, 0.16, 0.11, 0.09, 0.08, 0.06, 0.06]
    primary_diagnosis = np.random.choice(diagnoses, size=n_samples, p=diag_weights)
    
    # Charlson Comorbidity Index (0 - 8)
    charlson_index = np.random.poisson(lam=2.3, size=n_samples)
    charlson_index = np.clip(charlson_index, 0, 9)
    
    # Lab and Medication counts
    num_medications = np.random.normal(14, 6, n_samples).astype(int)
    num_medications = np.clip(num_medications, 1, 40)
    
    num_lab_procedures = np.random.normal(42, 16, n_samples).astype(int)
    num_lab_procedures = np.clip(num_lab_procedures, 5, 110)
    
    # 3. Clinical Biomarkers & Vitals
    hemoglobin = np.round(np.random.normal(12.6, 2.1, n_samples), 1)
    hemoglobin = np.clip(hemoglobin, 6.5, 18.0)
    
    creatinine = np.round(np.random.gamma(shape=1.8, scale=0.7, size=n_samples), 2)
    creatinine = np.clip(creatinine, 0.4, 6.0)
    
    glucose = np.round(np.random.normal(145, 55, n_samples), 0).astype(int)
    glucose = np.clip(glucose, 65, 420)
    
    systolic_bp = np.random.normal(132, 18, n_samples).astype(int)
    systolic_bp = np.clip(systolic_bp, 85, 210)
    
    oxygen_saturation = np.random.normal(96, 3.2, n_samples).astype(int)
    oxygen_saturation = np.clip(oxygen_saturation, 78, 100)
    
    high_risk_medication = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.38, 0.62])
    
    discharge_disposition = np.random.choice(
        ['Home (Routine)', 'Home Health Service', 'Skilled Nursing Facility (SNF)', 'Long-Term Care Hospital'],
        size=n_samples,
        p=[0.60, 0.22, 0.13, 0.05]
    )
    
    # 4. Latent Readmission Risk Calculation (Grounded in LACE + HOSPITAL models)
    # L = Length of stay
    # A = Acuity (Admission Type)
    # C = Comorbidities (Charlson)
    # E = Emergency visits
    
    # Normalized score components
    risk_score = (
        0.045 * (age - 50) +
        0.280 * (length_of_stay - 3) +
        0.850 * (num_inpatient_visits) +
        0.550 * (num_emergency_visits) +
        0.480 * (charlson_index - 1.5) +
        0.400 * (creatinine > 1.6).astype(int) +
        0.350 * (hemoglobin < 10.0).astype(int) +
        0.300 * (oxygen_saturation < 92).astype(int) +
        0.250 * (glucose > 200).astype(int) +
        0.300 * (high_risk_medication == 'Yes').astype(int) +
        0.500 * (admission_type == 'Emergency').astype(int) +
        0.250 * (admission_type == 'Urgent').astype(int) +
        0.450 * (discharge_disposition == 'Skilled Nursing Facility (SNF)').astype(int) +
        0.650 * (discharge_disposition == 'Long-Term Care Hospital').astype(int) +
        0.350 * (primary_diagnosis == 'Cardiology (Heart Failure/AMI)').astype(int) +
        0.300 * (primary_diagnosis == 'Nephrology (Kidney Disease)').astype(int) +
        0.250 * (primary_diagnosis == 'Pulmonology (COPD/Pneumonia)').astype(int) -
        3.20  # Base intercept
    )
    
    # Add slight realistic stochastic variation (logistic prob)
    prob = 1.0 / (1.0 + np.exp(-risk_score))
    
    # Target binary: readmitted within 30 days
    readmitted_30d = (prob >= 0.50).astype(int)
    
    # Final dataframe
    df = pd.DataFrame({
        'patient_id': [f'PT-{100000 + i}' for i in range(n_samples)],
        'age': age,
        'gender': gender,
        'admission_type': admission_type,
        'admission_source': admission_source,
        'length_of_stay': length_of_stay,
        'num_inpatient_visits_prior_year': num_inpatient_visits,
        'num_emergency_visits_prior_year': num_emergency_visits,
        'num_outpatient_visits_prior_year': num_outpatient_visits,
        'primary_diagnosis': primary_diagnosis,
        'charlson_comorbidity_index': charlson_index,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'hemoglobin_g_dl': hemoglobin,
        'creatinine_mg_dl': creatinine,
        'blood_glucose_mg_dl': glucose,
        'systolic_bp': systolic_bp,
        'oxygen_saturation': oxygen_saturation,
        'high_risk_medication': high_risk_medication,
        'discharge_disposition': discharge_disposition,
        'readmitted_30d': readmitted_30d
    })
    
    return df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_hospital_data(n_samples=12000)
    data_path = os.path.join(script_dir, 'hospital_readmissions.csv')
    df.to_csv(data_path, index=False)
    print(f"Generated {len(df)} patient records saved to {data_path}")
    print("Class distribution:")
    print(df['readmitted_30d'].value_counts(normalize=True))
    
    # Also generate a test batch sample (50 patients without the readmitted_30d label, but including patient info)
    batch_df = df.iloc[-50:].drop(columns=['readmitted_30d']).copy()
    batch_df.reset_index(drop=True, inplace=True)
    batch_path = os.path.join(script_dir, 'sample_batch_patients.csv')
    batch_df.to_csv(batch_path, index=False)
    print(f"Generated clean batch test file with 50 rows saved to {batch_path}")
    
    # Also generate an invalid batch file (has missing values to showcase strict validation!)
    invalid_batch = batch_df.head(10).copy()
    invalid_batch.loc[1, 'blood_glucose_mg_dl'] = np.nan
    invalid_batch.loc[3, 'hemoglobin_g_dl'] = np.nan
    invalid_batch.loc[6, 'primary_diagnosis'] = np.nan
    invalid_path = os.path.join(script_dir, 'sample_batch_with_missing_fields.csv')
    invalid_batch.to_csv(invalid_path, index=False)
    print(f"Generated validation test batch with missing fields saved to {invalid_path}")
