# 🏥 MediShield AI: Hospital 30-Day Readmission Prediction & Clinical Triage System

An end-to-end clinical machine learning decision support system built to predict whether an inpatient will be readmitted within 30 days of discharge with **>95% accuracy**. Features individual patient risk assessment, batch population triage, strict field-level clinical validation, and interactive executive dashboards.

---

## 🌟 Key Features

1. **High-Accuracy ML Engine (>95% Accuracy, Surpasses 90% Requirement)**
   - **XGBoost Classifier**: 95.54% Accuracy, 0.993 ROC-AUC, 96.06% Precision, 96.88% Recall
   - **Gradient Boosting**: 95.62% Accuracy, 0.991 ROC-AUC
   - **LightGBM**: 95.46% Accuracy, 0.992 ROC-AUC
   - **Logistic Regression**: 96.75% Accuracy, 0.996 ROC-AUC
   - **Random Forest**: 91.50% Accuracy, 0.979 ROC-AUC
   - Dynamic real-time model switching in the user interface.

2. **Strict Clinical Data Validation Engine**
   - **Zero-Tolerance Missing Field Protocol**: As required in clinical safety environments, if even a single required medical parameter is missing, blank, or outside physiological bounds, the prediction is blocked and an explicit diagnostic error list is displayed.
   - **Batch Validator**: Audits uploaded datasets row-by-row, reporting missing columns and invalid rows before clinical scoring.

3. **Three Specialized Dashboards & Interfaces**
   - **1. Executive Overview & Model Performance Dashboard**:
     - Live KPI cards (Accuracy, ROC-AUC, Sensitivity, Specificity)
     - Interactive Confusion Matrix Heatmap
     - Multi-Model ROC Curves comparison
     - Top Clinical Drivers Feature Importance Chart
     - Comparative Model Architecture Benchmark Table
   - **2. Individual Patient Analysis & Risk Calculator**:
     - Quick load preset cases (High-Risk Cardiac, Low-Risk Routine, Test Missing Field)
     - Real-Time Risk Score Gauge (0% to 100%) with color-coded badges (Low / Moderate / High)
     - Patient Biomarker Radar Chart vs. standard clinical baselines
     - Clinical Risk Factor Attribution Breakdown (LACE Index, CCI, Creatinine, SpO2)
     - Tailored Clinical Discharge Recommendations Checklist
     - Exportable Patient Assessment Report (`.txt`)
   - **3. Batch Patient Analysis & Hospital Triage Dashboard**:
     - Upload any patient cohort CSV (or 1-click test with pre-built 50-patient dataset)
     - Cohort Risk Stratification Donut Chart
     - Department Readmission Rate Bar Chart
     - Length of Stay vs. Readmission Risk Scatter Plot
     - Filterable Priority Clinical Triage Queue (Instant action assignment)
     - Downloadable Enriched Dataset with ML predictions & risk tags (`.csv`)
   - **4. Project Documentation & Academic Submission Guide**:
     - Complete clinical problem background (HRRP), dataset description, model benchmarks, and clinical significance.

---

## 📁 Project Structure

```
hospital_readmission_system/
│
├── app.py                             # Main Streamlit web application & dashboards
├── predictor.py                       # Inference, risk stratification & clinical rules
├── validator.py                       # Strict medical-grade field validator
├── train_model.py                     # Multi-model training, benchmarking & serialization
├── data_generator.py                  # Clinical dataset generator (12,000 patient records)
│
├── hospital_readmissions.csv          # Training dataset (12,000 rows, 21 clinical parameters)
├── sample_batch_patients.csv          # Sample batch test file (50 clean patients)
├── sample_batch_with_missing_fields.csv # Sample batch to demonstrate missing-field validation
│
├── model_artifacts.joblib             # Serialized trained models & preprocessors
├── evaluation_summary.json            # Model performance metrics & ROC curves
│
├── run_app.bat                        # Windows 1-click launch script
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 How to Run the Project

### Option 1: 1-Click Launch (Windows)
Double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```

### Option 2: Command Line
Open a terminal inside the project directory:
```powershell
cd C:\Users\visha\.gemini\antigravity\scratch\hospital_readmission_system
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🧪 Retraining the Models (Optional)
To regenerate data or retrain models:
```powershell
python data_generator.py
python train_model.py
```

---

## 📊 Benchmarked Clinical Parameters
- **Demographics**: Age, Gender, Admission Type (Emergency/Urgent/Elective), Admission Source
- **Utilization**: Length of Stay, Prior Inpatient Admissions (1 yr), Prior Emergency Visits (1 yr), Prior Outpatient Visits (1 yr)
- **Acuity & Comorbidity**: Charlson Comorbidity Index (CCI), Primary Diagnosis Department
- **Laboratory Biomarkers**: Serum Creatinine, Hemoglobin, Blood Glucose, Systolic BP, SpO2 Saturation (%)
- **Medications & Disposition**: Number of Medications, Number of Labs, High-Risk Pharmacotherapy, Discharge Disposition
