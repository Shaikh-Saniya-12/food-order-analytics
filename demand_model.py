import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

# ---------- 1. BUILD DAILY DEMAND TABLE (one row per item per day) ----------
df = pd.read_csv("food_orders.csv", parse_dates=["Date"])

daily = df.groupby(["Date", "Item"])["Qty"].sum()

# add rows with 0 for days when an item sold nothing
all_days = pd.date_range(df["Date"].min(), df["Date"].max())
full_index = pd.MultiIndex.from_product(
    [all_days, df["Item"].unique()], names=["Date", "Item"])
daily = daily.reindex(full_index, fill_value=0).reset_index()
daily = daily.sort_values(["Item", "Date"]).reset_index(drop=True)

# ---------- 2. FEATURES ----------
daily["DayOfWeek"] = daily["Date"].dt.dayofweek
daily["Month"] = daily["Date"].dt.month
daily["IsWeekend"] = (daily["DayOfWeek"] >= 4).astype(int)  # Fri-Sun

# past demand of the same item (shift so we never use today's value)
grp = daily.groupby("Item")["Qty"]
daily["Lag1"] = grp.shift(1)
daily["Lag7"] = grp.shift(7)
daily["Roll7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
daily["Roll30"] = grp.transform(lambda s: s.shift(1).rolling(30).mean())

data = daily.dropna().reset_index(drop=True)

num_features = ["DayOfWeek", "Month", "IsWeekend",
                "Lag1", "Lag7", "Roll7", "Roll30"]
item_dummies = pd.get_dummies(data["Item"], prefix="Item", dtype=int)
X = pd.concat([data[num_features], item_dummies], axis=1)
y = data["Qty"]

# ---------- 3. TIME-BASED TRAIN/TEST SPLIT ----------
# train on the past, test on the last 60 days (no shuffling!)
cutoff = data["Date"].max() - pd.Timedelta(days=60)
train = data["Date"] <= cutoff
test = ~train
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")


# ---------- 4. MODELS ----------
print("\n===== RESULTS (test set) =====")

# Baseline: just predict the 7-day average. Our model must beat this.
evaluate("Baseline (7-day avg)", y_test, X_test["Roll7"])

lr = LinearRegression().fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr.predict(X_test))

rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                           min_samples_leaf=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
evaluate("Random Forest", y_test, rf_pred)

# ---------- 5. CHARTS ----------
# Actual vs predicted for the best-selling item
top_item = df.groupby("Item")["Qty"].sum().idxmax()
mask = (data.loc[test, "Item"] == top_item).values
dates = data.loc[test, "Date"][mask]

plt.figure(figsize=(10, 5))
plt.plot(dates, y_test[mask], label="Actual", marker="o", markersize=3)
plt.plot(dates, rf_pred[mask], label="Predicted", marker="o", markersize=3)
plt.title(f"Actual vs Predicted Daily Demand: {top_item}")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("charts/actual_vs_predicted.png")
plt.close()

# Feature importance
imp = pd.Series(rf.feature_importances_, index=X.columns)
imp = imp.sort_values(ascending=False).head(10)
print("\nTop features:\n", imp)
plt.figure(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, color="#2a9d8f")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("charts/feature_importance.png")
plt.close()

# ---------- 6. SAVE MODEL ----------
joblib.dump({"model": rf, "columns": list(X.columns)}, "demand_model.pkl", compress=3)
print("\nDone! Model saved as demand_model.pkl, charts saved in 'charts'.")