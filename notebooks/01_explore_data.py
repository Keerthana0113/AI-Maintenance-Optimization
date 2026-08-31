import pandas as pd


# ============================================================
# AI MAINTENANCE & OPTIMIZATION
# Step 1: Dataset Exploration
# ============================================================


# ------------------------------------------------------------
# 1. Dataset path
# ------------------------------------------------------------

DATA_PATH = "data/raw/simulated_iiot_dataset.csv"


# ------------------------------------------------------------
# 2. Load the CSV
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ------------------------------------------------------------
# 3. Dataset size
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("1. DATASET SIZE")
print("=" * 60)

print("Number of rows   :", df.shape[0])
print("Number of columns:", df.shape[1])


# ------------------------------------------------------------
# 4. Column names
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. COLUMN NAMES")
print("=" * 60)

for column in df.columns:
    print("-", column)


# ------------------------------------------------------------
# 5. First 5 rows
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ------------------------------------------------------------
# 6. Data types
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ------------------------------------------------------------
# 7. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ------------------------------------------------------------
# 8. Duplicate rows
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. DUPLICATE ROWS")
print("=" * 60)

print("Number of duplicates:", df.duplicated().sum())


# ------------------------------------------------------------
# 9. Statistical summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ------------------------------------------------------------
# 10. Machine failure distribution
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. MACHINE FAILURE DISTRIBUTION")
print("=" * 60)

print(df["machine_failure"].value_counts())