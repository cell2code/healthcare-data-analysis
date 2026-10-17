import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Display project title
print("=" * 50)
print("HEALTHCARE DATA ANALYSIS")
print("=" * 50)

# Define the dataset path
file_path = "healthcare data analyst/Healthcare_Data_Analysis/data/raw/claims_data.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display the first five rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display data types
print("\nData types:")
print(df.dtypes)

print("\nDataset loaded successfully!")
# ==========================================
# STEP 2: DATA QUALITY ASSESSMENT
# ==========================================

print("\n" + "=" * 50)
print("DATA QUALITY ASSESSMENT")
print("=" * 50)

# 1. Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 2. Calculate total missing values
total_missing = df.isnull().sum().sum()
print("\nTotal missing values:", total_missing)

# 3. Check duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nTotal duplicate rows:", duplicate_rows)

# 4. Check duplicate Claim IDs
duplicate_claim_ids = df["Claim ID"].duplicated().sum()
print("\nDuplicate Claim IDs:", duplicate_claim_ids)

# 5. Check the number of unique patients
unique_patients = df["Patient ID"].nunique()
print("\nNumber of unique patients:", unique_patients)

# 6. Check the number of unique claims
unique_claims = df["Claim ID"].nunique()
print("\nNumber of unique claims:", unique_claims)
# ==========================================
# STEP 3: DATE AND FINANCIAL DATA CHECKS
# ==========================================

print("\n" + "=" * 50)
print("DATE AND FINANCIAL DATA CHECKS")
print("=" * 50)

# 1. Convert service dates into datetime format
df["Date of Service"] = pd.to_datetime(
    df["Date of Service"],
    errors="coerce"
)

# Check invalid dates
invalid_dates = df["Date of Service"].isnull().sum()

print("\nInvalid service dates:", invalid_dates)

# Display date range
print("\nEarliest service date:")
print(df["Date of Service"].min())

print("\nLatest service date:")
print(df["Date of Service"].max())

# 2. Examine financial columns
financial_columns = [
    "Billed Amount",
    "Allowed Amount",
    "Paid Amount"
]

print("\nFinancial data summary:")
print(df[financial_columns].describe())

# 3. Check for negative financial amounts
for column in financial_columns:
    negative_count = (df[column] < 0).sum()
    print(f"Negative values in {column}: {negative_count}")

# 4. Check whether paid amounts exceed allowed amounts
paid_exceeds_allowed = (
    df["Paid Amount"] > df["Allowed Amount"]
).sum()

print(
    "\nRecords where Paid Amount exceeds Allowed Amount:",
    paid_exceeds_allowed
)

# 5. Check whether allowed amounts exceed billed amounts
allowed_exceeds_billed = (
    df["Allowed Amount"] > df["Billed Amount"]
).sum()

print(
    "Records where Allowed Amount exceeds Billed Amount:",
    allowed_exceeds_billed
)
# ==========================================
# STEP 4: BASIC HEALTHCARE DATA ANALYSIS
# ==========================================

print("\n" + "=" * 50)
print("BASIC HEALTHCARE DATA ANALYSIS")
print("=" * 50)

# 1. Most common diagnosis codes
print("\nTop 10 Most Common Diagnosis Codes:")

top_diagnoses = df["Diagnosis Code"].value_counts().head(10)

print(top_diagnoses)

# 2. Distribution of claim statuses
print("\nClaim Status Distribution:")

claim_status = df["Claim Status"].value_counts()

print(claim_status)

# 3. Distribution of claim outcomes
print("\nClaim Outcome Distribution:")

claim_outcomes = df["Outcome"].value_counts()

print(claim_outcomes)

# 4. Insurance type distribution
print("\nInsurance Type Distribution:")

insurance_distribution = df["Insurance Type"].value_counts()

print(insurance_distribution)

# 5. Calculate financial totals
print("\nFinancial Summary:")

print("Total Billed Amount:", df["Billed Amount"].sum())

print("Total Allowed Amount:", df["Allowed Amount"].sum())

print("Total Paid Amount:", df["Paid Amount"].sum())

# 6. Calculate overall payment percentage
total_allowed = df["Allowed Amount"].sum()
total_paid = df["Paid Amount"].sum()

if total_allowed != 0:
    payment_percentage = (total_paid / total_allowed) * 100

    print(
        "Paid Amount as a percentage of Allowed Amount:",
        round(payment_percentage, 2),
        "%"
    )
else:
    print("Payment percentage cannot be calculated.")
    # ==========================================
# STEP 5: DATA VISUALISATION
# ==========================================

print("\nCreating visualisations...")

# Create output folder
os.makedirs(
    "healthcare data analyst/Healthcare_Data_Analysis/visualizations",
    exist_ok=True
)

output_folder = (
    "healthcare data analyst/Healthcare_Data_Analysis/visualizations"
)

# 1. Top 10 diagnosis codes
top_diagnoses = df["Diagnosis Code"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_diagnoses.sort_values().plot(kind="barh")

plt.title("Top 10 Most Common Diagnosis Codes")
plt.xlabel("Number of Claims")
plt.ylabel("Diagnosis Code")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "top_diagnoses.png")
)
plt.close()


# 2. Claim outcome distribution
claim_outcomes = df["Outcome"].value_counts()

plt.figure(figsize=(8, 6))
claim_outcomes.plot(kind="bar")

plt.title("Distribution of Claim Outcomes")
plt.xlabel("Claim Outcome")
plt.ylabel("Number of Claims")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "claim_outcomes.png")
)
plt.close()


# 3. Financial comparison
financial_totals = df[
    ["Billed Amount", "Allowed Amount", "Paid Amount"]
].sum()

plt.figure(figsize=(8, 6))
financial_totals.plot(kind="bar")

plt.title("Comparison of Healthcare Claim Amounts")
plt.xlabel("Financial Category")
plt.ylabel("Total Amount")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "financial_summary.png")
)
plt.close()

print("Visualisations saved successfully!")
# STEP 6: DATA CLEANING AND SAVING

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Create a working copy; preserve the original dataset
clean_df = df.copy()

# Remove unnecessary spaces from column names
clean_df.columns = clean_df.columns.str.strip()

# Remove leading and trailing spaces from text columns
text_columns = clean_df.select_dtypes(include=["object"]).columns

for column in text_columns:
    clean_df[column] = clean_df[column].str.strip()

# Convert dates into proper datetime format
clean_df["Date of Service"] = pd.to_datetime(
    clean_df["Date of Service"], errors="coerce"
)

# Check for missing values after cleaning
print("\nMissing values after cleaning:")
print(clean_df.isnull().sum().sum())

# Remove exact duplicate rows
before = len(clean_df)
clean_df = clean_df.drop_duplicates()
removed = before - len(clean_df)

print("Duplicate rows removed:", removed)
print("Final dataset shape:", clean_df.shape)

# Create cleaned-data folder
output_folder = (
    "healthcare data analyst/Healthcare_Data_Analysis/data/cleaned"
)
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned dataset
output_path = os.path.join(
    output_folder, "claims_data_cleaned.csv"
)

clean_df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)
#  7: CLAIM DENIAL ANALYSIS

print("\n" + "=" * 50)
print("CLAIM DENIAL ANALYSIS")
print("=" * 50)

# Analyse claim outcomes
print("\nClaim Outcome Counts:")
print(clean_df["Outcome"].value_counts())

# Calculate denial percentage
total_claims = len(clean_df)
denied_claims = (
clean_df["Outcome"].astype(str).str.strip().str.lower() == "denied"
).sum()

if total_claims > 0:
    denial_percentage = (denied_claims / total_claims) * 100
    print("\nTotal Claims:", total_claims)
    print("Denied Claims:", denied_claims)
    print("Denial Percentage:", round(denial_percentage, 2), "%")

# Analyse denial reasons
print("\nTop Reasons Associated with Claims:")
print(clean_df["Reason Code"].value_counts().head(10))

# Compare outcomes by insurance type
print("\nClaim Outcomes by Insurance Type:")
print(pd.crosstab(clean_df["Insurance Type"], clean_df["Outcome"]))

# Compare paid amounts by outcome
print("\nAverage Paid Amount by Claim Outcome:")
print(clean_df.groupby("Outcome")["Paid Amount"].mean().round(2))

print("\nClaim denial analysis completed!")
print("STEP 7 IS RUNNING!")
# STEP 8: INSURANCE TYPE VISUALISATION

print("\nCreating insurance analysis graph...")

output_folder = (
    "healthcare data analyst/Healthcare_Data_Analysis/visualizations"
)

os.makedirs(output_folder, exist_ok=True)

insurance_outcomes = pd.crosstab(
    clean_df["Insurance Type"],
    clean_df["Outcome"]
)

insurance_outcomes.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Claim Outcomes by Insurance Type")
plt.xlabel("Insurance Type")
plt.ylabel("Number of Claims")
plt.xticks(rotation=0)
plt.legend(title="Claim Outcome")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "insurance_claim_outcomes.png")
)

plt.close()

print("Insurance visualisation saved successfully!")