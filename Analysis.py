import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

# ---------- 1. LOAD & CHECK ----------
df = pd.read_csv("food_orders.csv", parse_dates=["Date"])

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate orders:", df["OrderID"].duplicated().sum())
print("\nData types:\n", df.dtypes)

# ---------- 2. NEW COLUMNS ----------
df["Month"] = df["Date"].dt.to_period("M").astype(str)
df["DayName"] = df["Date"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# ---------- 3. KEY NUMBERS ----------
total_revenue = df["Total"].sum()
total_orders = df["OrderID"].nunique()
print("\n===== KEY NUMBERS =====")
print(f"Total revenue     : Rs {total_revenue:,.0f}")
print(f"Total orders      : {total_orders}")
print(f"Avg order value   : Rs {total_revenue / total_orders:,.0f}")
print(f"Unique customers  : {df['Customer'].nunique()}")

# ---------- 4. ANALYSIS + CHARTS ----------

# Top 10 items by quantity sold
top_items = df.groupby("Item")["Qty"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 items (qty sold):\n", top_items)
plt.figure(figsize=(9, 5))
sns.barplot(x=top_items.values, y=top_items.index, color="#e4572e")
plt.title("Top 10 Items by Quantity Sold")
plt.xlabel("Quantity")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/top_items.png")
plt.close()

# Revenue by category
cat_rev = df.groupby("Category")["Total"].sum().sort_values(ascending=False)
print("\nRevenue by category:\n", cat_rev)
plt.figure(figsize=(7, 5))
sns.barplot(x=cat_rev.index, y=cat_rev.values, color="#4c78a8")
plt.title("Revenue by Category")
plt.ylabel("Revenue (Rs)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("charts/revenue_by_category.png")
plt.close()

# Orders by hour (peak hours)
hourly = df.groupby("Hour")["OrderID"].count()
print("\nPeak hours:\n", hourly.sort_values(ascending=False).head(3))
plt.figure(figsize=(9, 5))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="#2a9d8f")
plt.title("Orders by Hour of Day")
plt.xlabel("Hour (24h)")
plt.ylabel("Orders")
plt.xticks(hourly.index)
plt.tight_layout()
plt.savefig("charts/orders_by_hour.png")
plt.close()

# Orders by day of week
daily = df.groupby("DayName")["OrderID"].count().reindex(day_order)
print("\nOrders by weekday:\n", daily)
plt.figure(figsize=(8, 5))
sns.barplot(x=daily.index, y=daily.values, color="#f4a261")
plt.title("Orders by Day of Week")
plt.xlabel("")
plt.ylabel("Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("charts/orders_by_weekday.png")
plt.close()

# Monthly revenue trend
monthly = df.groupby("Month")["Total"].sum()
print("\nMonthly revenue:\n", monthly)
plt.figure(figsize=(9, 5))
sns.lineplot(x=monthly.index, y=monthly.values, marker="o", color="#6a4c93")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/monthly_revenue.png")
plt.close()

# Payment method split
pay = df["Payment"].value_counts()
print("\nPayment split:\n", pay)
plt.figure(figsize=(6, 6))
plt.pie(pay.values, labels=pay.index, autopct="%1.0f%%", startangle=90)
plt.title("Payment Methods")
plt.tight_layout()
plt.savefig("charts/payment_split.png")
plt.close()

# Revenue by city
city_rev = df.groupby("City")["Total"].sum().sort_values(ascending=False)
print("\nRevenue by city:\n", city_rev)

# Top customers + repeat customers
cust = df.groupby("Customer").agg(orders=("OrderID", "count"),
                                  spend=("Total", "sum"))
print("\nTop 5 customers by spend:\n",
      cust.sort_values("spend", ascending=False).head(5))
repeat_pct = (cust["orders"] > 1).mean() * 100
print(f"\nRepeat customers: {repeat_pct:.0f}%")

print("\nDone! Charts saved in the 'charts' folder.")