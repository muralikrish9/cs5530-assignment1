"""
CS 5530 - Assignment 1, Question 1
Frailty & Grip Strength Analysis
Three-stage workflow: Ingest -> Process -> Analyze
"""

import pandas as pd
import os

# ============================================================
# Stage 1: Ingest
# ============================================================
df = pd.read_csv("data.csv")
print("=== Stage 1: Ingestion ===")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(df.head())
print()

# ============================================================
# Stage 2: Process
# ============================================================
print("=== Stage 2: Processing ===")

# Unit standardization
df["Height_m"] = df["Height"] * 0.0254
df["Weight_kg"] = df["Weight"] * 0.45359237

# Feature engineering
df["BMI"] = round(df["Weight_kg"] / (df["Height_m"] ** 2), 2)

# Age group binning
bins = [0, 30, 45, 60, 200]
labels = ["<30", "30-45", "46-60", ">60"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

# Binary encoding for Frailty
df["Frailty_binary"] = df["Frailty"].map({"Y": 1, "N": 0}).astype("int8")

# One-hot encode AgeGroup
age_dummies = pd.get_dummies(df["AgeGroup"], prefix="AgeGroup", dtype=int)
df = pd.concat([df, age_dummies], axis=1)

print("Processed dataframe:")
print(df.to_string())
print()

# ============================================================
# Stage 3: Analyze & Report
# ============================================================
print("=== Stage 3: Analysis ===")

# Summary statistics for numeric columns
numeric_cols = ["Height_m", "Weight_kg", "BMI", "Grip_strength", "Frailty_binary"]
summary = df[numeric_cols].agg(["mean", "median", "std"]).round(4)
print("Summary statistics:")
print(summary)
print()

# Correlation between grip strength and frailty
corr = df["Grip_strength"].corr(df["Frailty_binary"])
print(f"Correlation (Grip_strength vs Frailty_binary): {corr:.4f}")

# Save report
os.makedirs("reports", exist_ok=True)
with open("reports/findings.md", "w") as f:
    f.write("# Frailty & Grip Strength Analysis - Findings\n\n")
    f.write("## Summary Statistics\n\n")
    f.write("| Metric | Height_m | Weight_kg | BMI | Grip_strength | Frailty_binary |\n")
    f.write("|--------|----------|-----------|-----|---------------|----------------|\n")
    for stat in ["mean", "median", "std"]:
        row = [f"{summary.loc[stat, c]:.4f}" for c in numeric_cols]
        f.write(f"| {stat} | " + " | ".join(row) + " |\n")
    f.write("\n## Correlation Analysis\n\n")
    f.write(f"Pearson correlation between Grip Strength (kg) and Frailty (binary): **{corr:.4f}**\n\n")
    if corr < 0:
        f.write("The negative correlation indicates that lower grip strength is associated "
                "with higher likelihood of frailty, which aligns with clinical expectations. "
                "Participants classified as frail tend to have weaker grip strength.\n")
    else:
        f.write("The positive correlation suggests grip strength increases with frailty, "
                "which is unexpected and may warrant further investigation.\n")

print("\nReport saved to reports/findings.md")
