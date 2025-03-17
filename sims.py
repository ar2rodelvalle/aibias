# -------------------------------
# AI Bias Evaluation - Complete Automated Analysis
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- User-controlled parameter ---
AI_biased_dummy_data = True  # Set to False for unbiased scenario

# ------------------------------------
# Step 1: Generate Dummy Data
# ------------------------------------
print("\n=== Step 1: Generating Dummy Data ===")

np.random.seed(42)
n = 1000  

accounts = [f"ACC_{i:05d}" for i in range(n)]
gender_labels = np.random.binomial(1, 0.5, n)  

if AI_biased_dummy_data:
    response_prob = np.where(gender_labels == 1, 0.8, 0.3)
    scenario = "BIASED"
else:
    response_prob = np.full(n, 0.55)
    scenario = "UNBIASED"

responses = np.random.binomial(1, response_prob)

gender_pred_scores = np.clip(
    gender_labels * np.random.uniform(0.6, 1.0, n) +
    (1 - gender_labels) * np.random.uniform(0.0, 0.4, n), 0, 1)

dummy_df = pd.DataFrame({
    "Account": accounts,
    "Gender_prediction_Score": gender_pred_scores,
    "AI_Product_Class_Response": responses,
    "Gender_Label": gender_labels
})

print("Dataset preview:")
print(dummy_df.head())
print("Dataset size:", dummy_df.shape)

plt.figure(figsize=(8, 5))
plt.hist(dummy_df['Gender_prediction_Score'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Gender Prediction Scores')
plt.xlabel('Gender Prediction Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
dummy_df.groupby('Gender_Label')['AI_Product_Class_Response'].mean().plot(kind='bar', color=['lightblue', 'orange'])
plt.title('Mean AI Product Response by True Gender')
plt.xlabel('True Gender (0=Female, 1=Male)')
plt.ylabel('Mean Response Rate')
plt.grid(True)
plt.show()

print("\nNarrative:")
print(f"We generated a {scenario} dummy dataset with {n} accounts. Each account has a predicted gender probability, an actual binary gender label, and a binary response to an AI product. The visuals illustrate potential bias based on gender.")

# ------------------------------------
# Step 2: Monte Carlo Simulation
# ------------------------------------
print("\n=== Step 2: Monte Carlo Simulation for Bias Assessment ===")

concentration = 10
dummy_df['alpha'] = dummy_df['Gender_prediction_Score'] * concentration
dummy_df['beta'] = (1 - dummy_df['Gender_prediction_Score']) * concentration

def monte_carlo_bias(df, iterations=1000):
    biases = []
    for _ in range(iterations):
        sim_gender = np.random.beta(df['alpha'], df['beta']) > 0.5
        if len(np.unique(sim_gender)) < 2:
            continue
        group_means = df.groupby(sim_gender)['AI_Product_Class_Response'].mean()
        bias = group_means[1] - group_means[0]
        biases.append(bias)
    return np.array(biases)

mc_biases = monte_carlo_bias(dummy_df)
mc_biases = mc_biases[~np.isnan(mc_biases)]

plt.figure(figsize=(8, 5))
plt.hist(mc_biases, bins=30, color='orchid', edgecolor='black')
plt.title('Distribution of Monte Carlo Bias Metric')
plt.xlabel('Bias Metric (Difference by Gender)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print(f"Monte Carlo mean bias: {np.mean(mc_biases):.4f}")
print(f"Monte Carlo bias standard deviation: {np.std(mc_biases):.4f}")

print("\nNarrative:")
if np.mean(mc_biases) > 0.05:
    print("Our Monte Carlo simulations indicate a clear bias favoring one gender.")
elif np.mean(mc_biases) < -0.05:
    print("Our Monte Carlo simulations indicate a clear bias favoring the opposite gender.")
else:
    print("Our Monte Carlo simulations indicate no significant gender bias.")

# ------------------------------------
# Step 3: Bootstrapping for Confidence Intervals
# ------------------------------------
print("\n=== Step 3: Confidence Interval Estimation (Bootstrap) ===")

def bootstrap_bias(df, iterations=200):
    boot_means = []
    for _ in range(iterations):
        sample = df.sample(n=len(df), replace=True)
        boot_biases = monte_carlo_bias(sample, iterations=100)
        boot_means.append(np.nanmean(boot_biases))
    return np.array(boot_means)

bootstrap_results = bootstrap_bias(dummy_df)

ci_lower, ci_upper = np.nanpercentile(bootstrap_results, [2.5, 97.5])
print(f"95% Confidence Interval for Bias Metric: [{ci_lower:.4f}, {ci_upper:.4f}]")

plt.figure(figsize=(8, 5))
plt.hist(bootstrap_results, bins=30, color='lightcoral', edgecolor='black')
plt.axvline(ci_lower, color='blue', linestyle='--', label='Lower CI (2.5%)')
plt.axvline(ci_upper, color='red', linestyle='--', label='Upper CI (97.5%)')
plt.title('Bootstrap Distribution of Bias Metrics')
plt.xlabel('Bias Metric')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print("\nNarrative:")
if ci_lower > 0 or ci_upper < 0:
    print("Bootstrap confirms statistically significant bias. Immediate action recommended.")
else:
    print("Bootstrap shows no significant bias. Regular monitoring advised.")

# ------------------------------------
# Step 4: Sensitivity Analysis
# ------------------------------------
print("\n=== Step 4: Sensitivity Analysis ===")

concentrations = [5, 10, 15, 20, 25]
mean_biases = []

for c in concentrations:
    dummy_df['alpha'] = dummy_df['Gender_prediction_Score'] * c
    dummy_df['beta'] = (1 - dummy_df['Gender_prediction_Score']) * c
    biases_c = monte_carlo_bias(dummy_df, iterations=200)
    mean_biases.append(np.nanmean(biases_c))

plt.figure(figsize=(8, 5))
plt.plot(concentrations, mean_biases, marker='o', linestyle='-', color='purple')
plt.title('Sensitivity of Bias Metric to Concentration Parameter')
plt.xlabel('Concentration Parameter')
plt.ylabel('Mean Bias Metric')
plt.grid(True)
plt.show()

print("\nNarrative:")
print("Sensitivity analysis ensures robustness. Stable results indicate reliable bias assessment.")


# ===================================================================================================
