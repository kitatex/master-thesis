import pandas as pd
import networkx as nx
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config.paths import TOPICS_PATH

TOPIC_NAME = "#gaza"
MIN_RP = 2

PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_leidenmeta0.3.graphml"
)

DATA_LEVEL = "discussion"  # global vs discussion

CSV_FEATURES = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/metadata/features/{TOPIC_NAME}_features_{DATA_LEVEL}.csv"
)

# --- 1. Load the Graph and Extract the Target Variable (Side) ---
print("Loading GraphML...")
graph = nx.read_graphml(PARTIT_GRAPH)

graph_nodes = []
for node, data in graph.nodes(data=True):
    if "name" in data and "side" in data:
        # 'name' in your GraphML holds the user_id as a float
        graph_nodes.append({"user_id": float(data["name"]), "side": int(data["side"])})

df_graph = pd.DataFrame(graph_nodes)

# --- 2. Load the Features CSV ---
print("Loading Features CSV...")
df_features = pd.read_csv(CSV_FEATURES)
df_features["user_id"] = df_features["user_id"].astype(float)

# --- 3. Merge Data ---
df = pd.merge(df_features, df_graph, on="user_id", how="inner")
print(f"Merged data: {len(df)} users ready for analysis.")

# --- 4. Prepare and Clean Features (X) and Target (y) ---
cols_to_drop = ["user_id", "side", "bias_distribution", "pct_curse"]

# Safely drop columns if they exist
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df["side"]

# Step 4a: Drop Zero-Variance Features
# A feature with 0 variance will immediately cause a singular matrix error
variances = X.var()
zero_var_cols = variances[variances == 0].index
if len(zero_var_cols) > 0:
    print(f"Dropping zero-variance columns: {list(zero_var_cols)}")
    X = X.drop(columns=zero_var_cols)

# Step 4b: Standardize Features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Add a constant (intercept) for statsmodels
X_scaled = sm.add_constant(X_scaled)

# Step 4c: Iterative VIF Filtering to fix Multicollinearity
print("\nChecking for Multicollinearity (VIF)...")
VIF_THRESHOLD = 10.0  # A strict but standard threshold for academic research

while True:
    # Calculate VIF for each feature (excluding the constant for dropping logic)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_scaled.columns  # type: ignore
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled.values, i)  # type: ignore
        for i in range(X_scaled.shape[1])  # type: ignore
    ]

    # Filter out the constant so we don't drop the intercept
    vif_data_no_const = vif_data[vif_data["feature"] != "const"]

    max_vif = vif_data_no_const["VIF"].max()
    if max_vif > VIF_THRESHOLD:
        # Identify the feature with the highest VIF
        feature_to_drop = vif_data_no_const.loc[
            vif_data_no_const["VIF"].idxmax(), "feature"
        ]
        print(f"Dropping '{feature_to_drop}' (VIF = {max_vif:.2f})")
        X_scaled = X_scaled.drop(columns=[feature_to_drop])  # type: ignore
    else:
        break

print("\nFinal selected features based on VIF:")
print(X_scaled.columns.tolist())  # type: ignore

# --- 5. Run Logistic Regression ---
print("\nFitting Logistic Regression Model...")
try:
    model = sm.Logit(y, X_scaled)
    result = model.fit(disp=False)
    print(result.summary())

    # --- 6. Calculate the "Single Number" Score ---
    df["composite_probability"] = result.predict(X_scaled)
    df["composite_log_odds"] = np.dot(X_scaled, result.params)

    # Save the final dataset
    df.to_csv("user_composite_scores.csv", index=False)
    print("\nSaved composite scores to user_composite_scores.csv")

except np.linalg.LinAlgError as e:
    print(f"\nLinAlgError encountered: {e}")
    print(
        "This implies perfectly overlapping data remains. Check your feature definitions."
    )
