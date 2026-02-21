"""
CS 5530 - Assignment 1, Question 2
Student Performance Dataset - EDA & Visualizations
Stages: Ingest -> Preprocess -> Visualize
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ============================================================
# Stage 1: Ingest
# ============================================================
df = pd.read_csv("student_performance.csv")
print("=== Stage 1: Ingestion ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# ============================================================
# Stage 2: Preprocess
# ============================================================
print("=== Stage 2: Preprocessing ===")

# Check for missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)

# Fill numeric missing values with column mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"  Filled {col} missing values with mean")

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  Filled {col} missing values with mode")

# Check dtypes
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic stats:\n{df.describe()}")
print()

# ============================================================
# Stage 3: Visualizations
# ============================================================
print("=== Stage 3: Visualizations ===")
report_lines = ["# Student Performance Analysis - Report\n"]

# --- V1: Gender Boxplots (math vs reading) ---
fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=300)

sns.boxplot(data=df, x="gender", y="math score", hue="gender", ax=axes[0], palette="Set2", legend=False)
axes[0].set_title("Math Score by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Math Score")

sns.boxplot(data=df, x="gender", y="reading score", hue="gender", ax=axes[1], palette="Set2", legend=False)
axes[1].set_title("Reading Score by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Reading Score")

fig.suptitle("V1: Gender Differences in Math vs Reading Scores", fontsize=12)
plt.tight_layout()
plt.savefig("figures/v1_gender_boxplots.png")
plt.close()
print("Saved V1")

# V1 interpretation
male_math = df[df["gender"] == "male"]["math score"]
female_math = df[df["gender"] == "female"]["math score"]
male_read = df[df["gender"] == "male"]["reading score"]
female_read = df[df["gender"] == "female"]["reading score"]

report_lines.append("## V1 - Gender Boxplots (Math vs Reading)\n")
report_lines.append(
    f"Male students have a higher median math score ({male_math.median():.1f}) compared to "
    f"female students ({female_math.median():.1f}). "
    f"In contrast, female students score higher in reading with a median of {female_read.median():.1f} "
    f"versus {male_read.median():.1f} for males. "
    f"The spread of math scores is similar for both genders, with comparable IQR values. "
    f"Reading scores show a wider gap, with females consistently outperforming males. "
    f"Both distributions are roughly symmetric with few outliers. "
    f"This suggests a gender-based difference in subject strengths, with males leaning toward math "
    f"and females toward reading. "
    f"However, the overlap between distributions is substantial, meaning gender alone is not a strong predictor.\n"
)

# --- V2: Test Prep Impact on Math ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sns.violinplot(data=df, x="test preparation course", y="math score", hue="test preparation course", palette="muted", ax=ax, legend=False)
ax.set_title("V2: Test Preparation Impact on Math Score")
ax.set_xlabel("Test Preparation Course")
ax.set_ylabel("Math Score")
plt.tight_layout()
plt.savefig("figures/v2_test_prep_math.png")
plt.close()
print("Saved V2")

completed = df[df["test preparation course"] == "completed"]["math score"]
none_prep = df[df["test preparation course"] == "none"]["math score"]

report_lines.append("## V2 - Test Prep Impact on Math\n")
report_lines.append(
    f"Students who completed the test preparation course have a higher mean math score "
    f"({completed.mean():.1f}) compared to those who did not ({none_prep.mean():.1f}). "
    f"The violin plot shows the completed group has a denser concentration of scores in the upper range. "
    f"The none group has a wider tail toward lower scores. "
    f"The median for the completed group is also noticeably higher. "
    f"This indicates that test preparation has a measurable positive effect on math performance. "
    f"However, some students who did not prepare still scored very high, suggesting other factors "
    f"like prior knowledge or aptitude also play a role. "
    f"Overall, test prep appears to shift the distribution upward by roughly "
    f"{completed.mean() - none_prep.mean():.1f} points on average.\n"
)

# --- V3: Lunch Type and Average Performance ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
lunch_means = df.groupby("lunch")[["math score", "reading score", "writing score"]].mean()
lunch_means.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_title("V3: Mean Scores by Lunch Type")
ax.set_xlabel("Lunch Type")
ax.set_ylabel("Mean Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Subject")
plt.tight_layout()
plt.savefig("figures/v3_lunch_scores.png")
plt.close()
print("Saved V3")

std_lunch = df[df["lunch"] == "standard"]
free_lunch = df[df["lunch"] == "free/reduced"]
std_avg = std_lunch[["math score", "reading score", "writing score"]].mean().mean()
free_avg = free_lunch[["math score", "reading score", "writing score"]].mean().mean()

report_lines.append("## V3 - Lunch Type and Average Performance\n")
report_lines.append(
    f"Students with standard lunch have a higher overall average score ({std_avg:.1f}) "
    f"compared to those with free/reduced lunch ({free_avg:.1f}). "
    f"This difference is consistent across all three subjects. "
    f"Math shows the largest gap between the two groups. "
    f"Reading and writing scores follow a similar pattern but with slightly smaller differences. "
    f"Lunch type is often used as a proxy for socioeconomic status. "
    f"These results suggest that socioeconomic background has a noticeable impact on academic performance. "
    f"The gap of approximately {std_avg - free_avg:.1f} points is significant and consistent.\n"
)

# --- V4: Subject Correlations ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
corr_matrix = df[["math score", "reading score", "writing score"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
            square=True, ax=ax, linewidths=1)
ax.set_title("V4: Correlation Heatmap of Subject Scores")
plt.tight_layout()
plt.savefig("figures/v4_correlation_heatmap.png")
plt.close()
print("Saved V4")

report_lines.append("## V4 - Subject Correlations\n")
report_lines.append(
    f"Reading and writing scores have the highest correlation ({corr_matrix.loc['reading score', 'writing score']:.3f}), "
    f"indicating students who read well also tend to write well. "
    f"Math and reading show a moderate-to-strong correlation ({corr_matrix.loc['math score', 'reading score']:.3f}). "
    f"Math and writing also correlate well ({corr_matrix.loc['math score', 'writing score']:.3f}). "
    f"All three subjects move together positively, meaning strong students tend to perform well across the board. "
    f"The reading-writing pair being strongest makes intuitive sense as both are language-based skills. "
    f"Math has a somewhat lower correlation with the other two, suggesting it tests a more distinct skill set. "
    f"No negative correlations exist, confirming that scoring well in one subject does not come at the cost of another.\n"
)

# --- V5: Math vs Reading Scatter with Trend Lines by Test Prep ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

colors = {"completed": "#2ca02c", "none": "#d62728"}
for group, color in colors.items():
    subset = df[df["test preparation course"] == group]
    ax.scatter(subset["reading score"], subset["math score"], alpha=0.4,
               label=f"{group} (n={len(subset)})", color=color, s=20)
    # Best-fit line
    z = np.polyfit(subset["reading score"], subset["math score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset["reading score"].min(), subset["reading score"].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2)

ax.set_xlabel("Reading Score")
ax.set_ylabel("Math Score")
ax.set_title("V5: Math vs Reading Score by Test Preparation Course")
ax.legend()
plt.tight_layout()
plt.savefig("figures/v5_scatter_trendlines.png")
plt.close()
print("Saved V5")

# Compute slopes for report
comp_sub = df[df["test preparation course"] == "completed"]
none_sub = df[df["test preparation course"] == "none"]
slope_comp = np.polyfit(comp_sub["reading score"], comp_sub["math score"], 1)[0]
slope_none = np.polyfit(none_sub["reading score"], none_sub["math score"], 1)[0]

report_lines.append("## V5 - Math vs Reading Scatter with Trend Lines\n")
report_lines.append(
    f"There is a clear positive association between math and reading scores for both groups. "
    f"The completed group (n={len(comp_sub)}) has a best-fit slope of {slope_comp:.3f}, "
    f"while the none group (n={len(none_sub)}) has a slope of {slope_none:.3f}. "
    f"The completed group's trend line sits higher, reflecting their overall better performance. "
    f"Both slopes are similar, suggesting the rate at which math improves with reading is "
    f"roughly the same regardless of test prep. "
    f"The main effect of test prep appears to be an upward shift (intercept) rather than a change in slope. "
    f"Scatter shows considerable overlap between groups, meaning prep is not the sole factor. "
    f"Some students in the none group still achieve top scores, indicating natural aptitude or "
    f"other preparation methods. "
    f"Overall, the plot confirms that math and reading are moderately correlated and test prep "
    f"provides a consistent but not dramatic advantage.\n"
)

# Write report
with open("reports/findings.md", "w") as f:
    f.write("\n".join(report_lines))
print("\nReport saved to reports/findings.md")
