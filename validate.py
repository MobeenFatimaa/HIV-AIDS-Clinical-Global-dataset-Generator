import time
import polars as pl

def validate_hiv_dataset_v2(file_path: str):
    print(f"--- Starting Validation on: {file_path} ---")
    start_time = time.time()
    
    df = pl.read_csv(file_path)
    total_rows, total_cols = df.shape
    print(f"Dataset Loaded: {total_rows:,} rows | {total_cols} columns")
    print(f"Memory Usage: {df.estimated_size() / (1024**2):.2f} MB\n")
    
    validation_passed = True
    issues = []

    # 1. Structural Checks
    print("[1/4] Structural Checks...")
    if df["Patient_ID"].n_unique() != total_rows:
        issues.append(f"CRITICAL: Duplicate Patient_IDs found! ({total_rows - df['Patient_ID'].n_unique():,} duplicates)")
        validation_passed = False
    
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    if total_nulls > 0:
        issues.append(f"WARNING: Dataset contains {total_nulls:,} null/missing values.")
    else:
        print(" -> No null values detected.")

    # 2. Boundary & Range Checks
    print("[2/4] Boundary & Range Checks...")
    
    # Age [15 - 80]
    out_of_bounds_age = df.filter((pl.col("Age") < 15) | (pl.col("Age") > 80)).height
    if out_of_bounds_age > 0:
        issues.append(f"FAIL: Age out of range [15-80]: {out_of_bounds_age:,} rows")
        validation_passed = False
        
    # Current CD4 Count [5 - 1600]
    out_of_bounds_cd4 = df.filter((pl.col("Current_CD4_Count") < 5) | (pl.col("Current_CD4_Count") > 1600)).height
    if out_of_bounds_cd4 > 0:
        issues.append(f"FAIL: Current CD4 Count out of range [5-1600]: {out_of_bounds_cd4:,} rows")
        validation_passed = False

    # Hemoglobin [5.0 - 18.0]
    out_of_bounds_hgb = df.filter((pl.col("Hemoglobin_g_dL") < 5.0) | (pl.col("Hemoglobin_g_dL") > 18.0)).height
    if out_of_bounds_hgb > 0:
        issues.append(f"FAIL: Hemoglobin out of range [5.0-18.0]: {out_of_bounds_hgb:,} rows")
        validation_passed = False

    # 3. Domain Logic & Relationship Checks
    print("[3/4] Domain Logic & Relationship Checks...")
    
    # Rule A: Patients NOT on ART must have ART_Adherence_Pct == 0 and ART_Regimen == "None"
    invalid_art_adherence = df.filter((pl.col("ART_Status") == 0) & (pl.col("ART_Adherence_Pct") > 0)).height
    invalid_art_regimen = df.filter((pl.col("ART_Status") == 0) & (pl.col("ART_Regimen") != "None")).height
    if invalid_art_adherence > 0 or invalid_art_regimen > 0:
        issues.append(f"FAIL: Untreated patients have active adherence/regimen entries.")
        validation_passed = False
        
    # Rule B: Viral Suppression Flag must accurately match Current_Viral_Load < 200
    flag_mismatch = df.filter(
        ((pl.col("Current_Viral_Load") < 200) & (pl.col("Viral_Suppression_Flag") != 1)) |
        ((pl.col("Current_Viral_Load") >= 200) & (pl.col("Viral_Suppression_Flag") != 0))
    ).height
    if flag_mismatch > 0:
        issues.append(f"FAIL: Viral_Suppression_Flag logic mismatch in {flag_mismatch:,} rows.")
        validation_passed = False

    # 4. Summary Statistics
    print("[4/4] Generating Summary Report...")
    summary_stats = df.select([
        pl.col("Age").mean().alias("Avg_Age"),
        pl.col("ART_Status").mean().alias("ART_Coverage_Rate"),
        pl.col("Viral_Suppression_Flag").mean().alias("Viral_Suppression_Rate"),
        pl.col("Current_CD4_Count").median().alias("Median_Current_CD4"),
        pl.col("TB_Coinfection").mean().alias("TB_Rate"),
        pl.col("Drug_Resistance_Mutation").mean().alias("Resistance_Mutation_Rate"),
        pl.col("Mortality_5Yr_Outcome").mean().alias("Mortality_Rate")
    ]).to_dicts()[0]

    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("VALIDATION REPORT")
    print("="*50)
    print(f"Status: {'PASSED' if validation_passed else 'FAILED WITH ISSUES'}")
    print(f"Execution Time: {elapsed:.2f} seconds\n")
    
    print("Statistical Summary:")
    for key, val in summary_stats.items():
        print(f"  - {key}: {val:.4f}")
        
    if issues:
        print("\nIssues Found:")
        for issue in issues:
            print(f"  - {issue}")
    print("="*50)

if __name__ == "__main__":
    validate_hiv_dataset_v2("global_hiv_synthetic_microdata_v2.csv")