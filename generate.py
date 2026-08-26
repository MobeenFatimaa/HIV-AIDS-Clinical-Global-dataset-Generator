import time
import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 500_000
OUTPUT_FILE = "global_hiv_synthetic_microdata_v2.csv"

start_time = time.time()
print(f"Generating expanded dataset with {N_ROWS:,} rows and 32 columns...")

# 1. Demographics & Geography
countries_income = {
    "South Africa": ("Upper-middle", 0.22), "Nigeria": ("Lower-middle", 0.18),
    "India": ("Lower-middle", 0.15), "Kenya": ("Lower-middle", 0.12),
    "Uganda": ("Low", 0.10), "Brazil": ("Upper-middle", 0.08),
    "United States": ("High", 0.05), "Thailand": ("Upper-middle", 0.05),
    "Ukraine": ("Lower-middle", 0.05)
}
country_names = list(countries_income.keys())
country_probs = [v[1] for v in countries_income.values()]
income_map = {k: v[0] for k, v in countries_income.items()}

country = np.random.choice(country_names, size=N_ROWS, p=country_probs)
income_group = np.vectorize(income_map.get)(country)
age = np.random.gamma(shape=9, scale=4, size=N_ROWS).clip(15, 80).astype(int)
gender = np.random.choice(["Female", "Male"], size=N_ROWS, p=[0.53, 0.47])
urban_residence = np.random.choice([1, 0], size=N_ROWS, p=[0.58, 0.42])

# 2. Socioeconomic & Healthcare Access
education_level = np.random.choice(["None", "Primary", "Secondary", "Tertiary"], size=N_ROWS, p=[0.15, 0.35, 0.38, 0.12])
insurance_status = np.where(income_group == "High", 
                            np.random.choice([1, 0], size=N_ROWS, p=[0.88, 0.12]),
                            np.random.choice([1, 0], size=N_ROWS, p=[0.35, 0.65]))
distance_to_clinic_km = np.random.exponential(scale=12, size=N_ROWS).clip(0.5, 150).round(1)

# 3. Clinical Timeline & Baseline Markers
diagnosis_year = np.random.randint(1990, 2025, size=N_ROWS)
baseline_cd4 = np.random.gamma(shape=2.5, scale=100, size=N_ROWS).clip(10, 800).astype(int)
baseline_viral_load = np.random.lognormal(mean=11.2, sigma=1.0, size=N_ROWS).clip(5000, 1000000).astype(int)

# 4. ART Treatment & Regimen Details
art_prob_base = np.where(np.isin(income_group, ["High", "Upper-middle"]), 0.82, 0.65)
art_prob_year = (diagnosis_year - 1990) * 0.008
art_status = (np.random.rand(N_ROWS) < np.clip(art_prob_base + art_prob_year, 0.20, 0.95)).astype(int)

regimens = ["TLD (TDF/3TC/DTG)", "TLE (TDF/3TC/EFV)", "AZT/3TC/NVP", "Second-Line (PI-based)", "None"]
regimen_choice = np.where(
    art_status == 1,
    np.random.choice(regimens[:-1], size=N_ROWS, p=[0.55, 0.25, 0.12, 0.08]),
    "None"
)

art_adherence_pct = np.where(art_status == 1, np.random.beta(a=7, b=2, size=N_ROWS) * 100, 0.0).round(1)
side_effects_reported = np.where(art_status == 1, np.random.choice([1, 0], size=N_ROWS, p=[0.24, 0.76]), 0)

# 5. Current Lab Markers (Dynamic Dependencies)
current_cd4 = np.where(
    art_status == 1,
    baseline_cd4 + (art_adherence_pct * 3.5) + np.random.normal(50, 30, size=N_ROWS),
    baseline_cd4 - np.random.normal(120, 40, size=N_ROWS)
).clip(5, 1600).astype(int)

cd4_cd8_ratio = (current_cd4 / np.random.normal(800, 150, size=N_ROWS)).clip(0.01, 2.5).round(2)

suppressed_vl = np.random.exponential(scale=25, size=N_ROWS).clip(0, 199)
unsuppressed_vl = np.random.lognormal(mean=10.5, sigma=1.2, size=N_ROWS).clip(1000, 750000)
current_viral_load = np.where(
    (art_status == 1) & (art_adherence_pct >= 80),
    suppressed_vl,
    unsuppressed_vl
).astype(int)

viral_suppression_flag = (current_viral_load < 200).astype(int)
hemoglobin_g_dl = (np.random.normal(13.5, 1.8, size=N_ROWS) - np.where(current_cd4 < 200, 1.5, 0)).clip(5.0, 18.0).round(1)
creatinine_mg_dl = np.random.normal(0.95, 0.25, size=N_ROWS).clip(0.4, 4.5).round(2)
alt_liver_enzyme_u_l = np.random.gamma(shape=4, scale=8, size=N_ROWS).clip(8, 250).astype(int)

# 6. Opportunistic Infections & Comorbidities
tb_risk = np.where(current_cd4 < 200, 0.35, 0.06)
tb_coinfection = (np.random.rand(N_ROWS) < tb_risk).astype(int)

hep_b_coinfection = (np.random.rand(N_ROWS) < 0.08).astype(int)
hep_c_coinfection = (np.random.rand(N_ROWS) < 0.05).astype(int)

pneumocystis_pneumonia = ((current_cd4 < 200) & (np.random.rand(N_ROWS) < 0.22)).astype(int)
kaposi_sarcoma = ((current_cd4 < 150) & (np.random.rand(N_ROWS) < 0.09)).astype(int)

hypertension = (np.random.rand(N_ROWS) < (0.10 + (age / 180))).astype(int)
diabetes_type2 = (np.random.rand(N_ROWS) < (0.04 + (age / 250))).astype(int)

# 7. Targeted Outcomes
drug_resistance_mutation = ((art_status == 1) & (art_adherence_pct < 60) & (np.random.rand(N_ROWS) < 0.30)).astype(int)
hospitalizations_last_year = np.where(current_cd4 < 200, np.random.poisson(1.5, size=N_ROWS), np.random.poisson(0.2, size=N_ROWS)).clip(0, 10)

mortality_risk = np.clip(
    0.01 
    + np.where(current_cd4 < 200, 0.22, 0.0) 
    + np.where(tb_coinfection == 1, 0.15, 0.0) 
    + np.where(drug_resistance_mutation == 1, 0.10, 0.0)
    - np.where(viral_suppression_flag == 1, 0.08, 0.0),
    0.005, 0.90
)
mortality_5yr_outcome = (np.random.rand(N_ROWS) < mortality_risk).astype(int)

# Build DataFrame
df_expanded = pd.DataFrame({
    "Patient_ID": [f"HIV_{i:08d}" for i in range(1, N_ROWS + 1)],
    "Country": country,
    "Income_Group": income_group,
    "Age": age,
    "Gender": gender,
    "Urban_Residence": urban_residence,
    "Education_Level": education_level,
    "Insurance_Status": insurance_status,
    "Distance_To_Clinic_KM": distance_to_clinic_km,
    "Diagnosis_Year": diagnosis_year,
    "Baseline_CD4_Count": baseline_cd4,
    "Baseline_Viral_Load": baseline_viral_load,
    "ART_Status": art_status,
    "ART_Regimen": regimen_choice,
    "ART_Adherence_Pct": art_adherence_pct,
    "Side_Effects_Reported": side_effects_reported,
    "Current_CD4_Count": current_cd4,
    "CD4_CD8_Ratio": cd4_cd8_ratio,
    "Current_Viral_Load": current_viral_load,
    "Viral_Suppression_Flag": viral_suppression_flag,
    "Hemoglobin_g_dL": hemoglobin_g_dl,
    "Creatinine_mg_dL": creatinine_mg_dl,
    "ALT_Liver_Enzyme_U_L": alt_liver_enzyme_u_l,
    "TB_Coinfection": tb_coinfection,
    "Hepatitis_B_Coinfection": hep_b_coinfection,
    "Hepatitis_C_Coinfection": hep_c_coinfection,
    "Pneumocystis_Pneumonia": pneumocystis_pneumonia,
    "Kaposi_Sarcoma": kaposi_sarcoma,
    "Hypertension": hypertension,
    "Diabetes_Type2": diabetes_type2,
    "Drug_Resistance_Mutation": drug_resistance_mutation,
    "Hospitalizations_Last_Year": hospitalizations_last_year,
    "Mortality_5Yr_Outcome": mortality_5yr_outcome
})

df_expanded.to_csv(OUTPUT_FILE, index=False)
elapsed = time.time() - start_time
print(f"Generated {df_expanded.shape[1]} columns across {len(df_expanded):,} rows in {elapsed:.2f} seconds.")