# Healthcare Data Analysis

## Project Overview
This project explores healthcare insurance claims using Python.
It focuses on data quality assessment, cleaning, exploratory
data analysis, financial summaries, and visualisation.

## Objectives
- Inspect the dataset structure and data types.
- Identify missing values and duplicate records.
- Clean and validate the data.
- Analyse claim outcomes and diagnosis codes.
- Examine claim outcomes across insurance types.
- Summarise billed, allowed, and paid amounts.
- Visualise important findings.

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Visual Studio Code

## Dataset
The dataset contains 1,000 healthcare claim records and
15 columns, including patient and provider identifiers,
service dates, diagnosis codes, insurance types, claim
outcomes, and financial amounts.

The dataset does not contain patient age, gender,
medication, or dosage fields.

## Data Cleaning
- Checked missing values.
- Checked duplicate rows and claim IDs.
- Removed leading and trailing spaces from text fields.
- Converted service dates to datetime format.
- Saved the cleaned dataset separately.

## Key Findings
- Total claims analysed: 1,000.
- Missing values detected in the initial dataset: 0.
- Exact duplicate rows detected: 0.
- Denied claims: 331 (33.1%).
- Paid claims: 357 (35.7%).
- Partially paid claims: 312 (31.2%).
- Total billed amount: 297,191.
- Total allowed amount: 223,112.
- Total paid amount: 200,754.
- Paid amount as a percentage of allowed amount: 89.98%.

## Visualisations
The project includes graphs showing:
- The most frequent diagnosis codes.
- Claim outcome distribution.
- Comparison of billed, allowed, and paid amounts.
- Claim outcomes by insurance type.

## Limitations
The analysis describes patterns in the available claims
dataset. It does not establish the causes of claim denials.
Diagnosis and reason codes require appropriate reference
documentation for clinical interpretation.

## How to Run
Install the required libraries:

pip install pandas numpy matplotlib seaborn

Run the analysis script from the project directory:

python healthcare_analysis.py
