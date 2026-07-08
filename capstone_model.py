import numpy as np
import pandas as pd

file_path = r"C:\Users\USER\Downloads\loan.csv"
df = pd.read_csv(file_path)

print("RAW DATASET")
print(f"intial dimensions: {df.shape[0]}rows, {df.shape[1]}columns")
df.columns = df.columns.str.strip().str.lower()
print(f"Normalized Column Headers: {list(df.columns)}\n")

print("HANDLING MISSING VALUES")

print(df.isnull().sum())
df['loanamount'] = df['loanamount'].fillna(df['loanamount'].median())
df['loan_amount_term'] = df['loan_amount_term'].fillna(df['loan_amount_term'].median())

categorical_cols = ['gender', 'married', 'dependents', 'self_employed', 'credit_history']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())
df['applicantincome'] = df['applicantincome'].astype(float)
df['coapplicantincome'] = df['coapplicantincome'].astype(float)
df['loanamount'] = df['loanamount'].astype(float)
df['loan_amount_term'] = df['loan_amount_term'].astype(int)

print("COMPREHENSIVE STATISTICAL ANALYSIS")
numeric_features = ['applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term']

for col in numeric_features:
    series = df[col]
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"Feature Column: [{col.upper()}]")
    print(f"  Mean : {mean_val:.2f}")
    print(f"  Median : {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")
    print(f"  First Quartile : {q1:.2f}")
    print(f"  Third Quartile : {q3:.2f}")
    print(f"  IQR : {iqr:.2f}")
    print(f"  Lower Outlier: {lower_bound:.2f}")
    print(f"  Upper Outlier: {upper_bound:.2f}")
    

print("PRODUCTION EXPORT")
filename = "capstone_clean.csv"
df.to_csv(filename, index=False)
print(f"Cleaned database stored as: '{filename}'")