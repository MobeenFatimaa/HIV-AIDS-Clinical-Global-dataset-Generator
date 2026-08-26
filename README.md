# HIV/AIDS Clinical Global Dataset Generator

A Python-based synthetic data generation pipeline for creating **500,000+ patient-level HIV/AIDS records across 33 features**, covering demographics, healthcare access, treatment, laboratory biomarkers, co-infections, comorbidities, and clinical outcomes.

- **Kaggle dataset:** https://www.kaggle.com/datasets/mobeenfatimah/hivaids-clinical-global-dataset
## Key Features

- 500,000+ synthetic patient records
- 33 clinical, demographic, and treatment features
- Vectorized dataset generation using NumPy and Pandas
- Automated validation using Polars
- Synthetic relationships between ART adherence, CD4 count, viral load, viral suppression, co-infections, and mortality
- Suitable for machine learning and healthcare analytics

## Dataset Structure

| Category | Features |
|---|---|
| Demographics | `Patient_ID`, `Country`, `Income_Group`, `Age`, `Gender`, `Urban_Residence` |
| Healthcare Access | `Education_Level`, `Insurance_Status`, `Distance_To_Clinic_KM` |
| Clinical | `Diagnosis_Year`, `Baseline_CD4_Count`, `Baseline_Viral_Load` |
| Treatment | `ART_Status`, `ART_Regimen`, `ART_Adherence_Pct`, `Side_Effects_Reported`, `Drug_Resistance_Mutation` |
| Laboratory | `Current_CD4_Count`, `CD4_CD8_Ratio`, `Current_Viral_Load`, `Viral_Suppression_Flag` |
| Co-infections | `TB_Coinfection`, `Hepatitis_B_Coinfection`, `Hepatitis_C_Coinfection`, `Pneumocystis_Pneumonia`, `Kaposi_Sarcoma` |
| Comorbidities | `Hypertension`, `Diabetes_Type2` |
| Outcomes | `Hospitalizations_Last_Year`, `Mortality_5Yr_Outcome` |

## Repository Structure

```text
├── generate.py
├── validate.py
├── pulse-of-pandemics-hiv-aids-global-biomarker.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
python generate.py
```

### Validate Dataset

```bash
python validate.py
```

## Machine Learning Applications

The dataset can be used for:

- Binary classification of `Mortality_5Yr_Outcome`
- Viral suppression prediction
- ART regimen classification
- Risk stratification
- Exploratory data analysis
- Healthcare analytics
- Feature engineering and model benchmarking

## Important Note

This is a **100% synthetic dataset** and does not contain real patient information. It is intended for educational, research, machine learning, and data science purposes only. It must not be used for medical diagnosis or clinical decision-making.

## Dataset & Profiles

- **Kaggle:** https://www.kaggle.com/mobeenfatimah
- **LinkedIn:** https://www.linkedin.com/in/mobeen-fatima-599a35347/
- 
## Author

**Mobeen Fatima**  
BS Computer Science (Specialized AI)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
