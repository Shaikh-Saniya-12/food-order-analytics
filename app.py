import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Food Order Analytics", page_icon="🍔",
                   layout="wide")


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("food_orders.csv", parse_dates=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DayName"] = df["Date"].dt.day_name()
    return df


@st.cache_resource
def load_model():
    if not os.path.exists("demand_model.pkl"):
        return None
    return joblib.load("demand_model.pkl")


df = load_data()
bundle = load_model()

st.title("🍔 Food Order Analytics & Demand Prediction")
st.caption("Synthetic order data | Python, pandas, scikit-learn, Streamlit")

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")
min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d),
                                   min_value=min_d, max_value=max_d)
cats = st.sidebar.multiselect("Category", sorted(df["Category"].unique()),
                              default=sorted(df["Category"].unique()))
cities = st.sidebar.multiselect("City", sorted(df["City"].unique()),
                                default=sorted(df["City"].unique()))

if len(date_range) != 2:
    st.info("Select both a start and an end date.")
    st.stop()

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
f = df[(df["Date"] >= start) & (df["Date"] <= end)
       & df["Category"].isin(cats) & df["City"].isin(cities)]

if f.empty:
    st.warning("No data for these filters.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Sales Insights", "🔮 Demand Forecast"])

# ================= TAB 1: SALES INSIGHTS =================
with tab1:
    revenue = f["Total"].sum()
    orders = f["OrderID"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{revenue:,.0f}")
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Avg Order Value", f"₹{revenue / orders:,.0f}")
    c4.metric("Customers", f"{f['Customer'].nunique()}")

    left, right = st.columns(2)

    top_items = (f.groupby("Item")["Qty"].sum()
                 .sort_values().tail(10).reset_index())
    left.plotly_chart(
        px.bar(top_items, x="Qty", y="Item", orientation="h",
               title="Top 10 Items by Quantity Sold"),
        use_container_width=True)

    cat_rev = f.groupby("Category")["Total"].sum().reset_index()
    right.plotly_chart(
        px.bar(cat_rev, x="Category", y="Total",
               title="Revenue by Category"),
        use_container_width=True)

    hourly = f.groupby("Hour")["OrderID"].count().reset_index()
    left.plotly_chart(
        px.line(hourly, x="Hour", y="OrderID", markers=True,
                title="Orders by Hour of Day"),
        use_container_width=True)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    wk = (f.groupby("DayName")["OrderID"].count()
          .reindex(days).fillna(0).reset_index())
    right.plotly_chart(
        px.bar(wk, x="DayName", y="OrderID",
               title="Orders by Day of Week"),
        use_container_width=True)

    monthly = f.groupby("Month")["Total"].sum().reset_index()
    left.plotly_chart(
        px.line(monthly, x="Month", y="Total", markers=True,
                title="Monthly Revenue Trend"),
        use_container_width=True)

    pay = f["Payment"].value_counts().reset_index()
    pay.columns = ["Payment", "Orders"]
    right.plotly_chart(
        px.pie(pay, names="Payment", values="Orders",
               title="Payment Methods"),
        use_container_width=True)

# ================= TAB 2: DEMAND FORECAST =================
with tab2:
    st.subheader("Predict daily demand for an item")

    if bundle is None:
        st.error("demand_model.pkl not found. Run demand_model.py first.")
        st.stop()

    model, columns = bundle["model"], bundle["columns"]

    # daily quantity per item (0 on days with no sales), same as training
    all_days = pd.date_range(df["Date"].min(), df["Date"].max())
    pivot = (df.pivot_table(index="Date", columns="Item", values="Qty",
                            aggfunc="sum")
             .reindex(all_days).fillna(0))
    last_date = pivot.index.max()

    item = st.selectbox("Item", sorted(pivot.columns))
    s = pivot[item]
    roll7, roll30, lag1 = s.iloc[-7:].mean(), s.iloc[-30:].mean(), s.iloc[-1]

    def predict(day):
        lag7_day = day - pd.Timedelta(days=7)
        lag7 = s.get(lag7_day, roll7)
        row = dict.fromkeys(columns, 0)
        row.update({
            "DayOfWeek": day.dayofweek, "Month": day.month,
            "IsWeekend": int(day.dayofweek >= 4),
            "Lag1": lag1, "Lag7": lag7, "Roll7": roll7, "Roll30": roll30,
        })
        row[f"Item_{item}"] = 1
        X = pd.DataFrame([row])[columns]
        return max(float(model.predict(X)[0]), 0)

    next_days = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    fc = pd.DataFrame({
        "Date": [d.strftime("%a %d %b") for d in next_days],
        "Predicted Qty": [round(predict(d), 1) for d in next_days],
    })

    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(
        px.bar(fc, x="Date", y="Predicted Qty",
               title=f"Next 7 days forecast: {item}"),
        use_container_width=True)
    c2.metric("Avg daily demand (last 30 days)", f"{roll30:.1f}")
    c2.metric("Forecast total (7 days)", f"{fc['Predicted Qty'].sum():.0f}")
    c2.dataframe(fc, hide_index=True)

    st.caption("Forecast starts the day after the last date in the data. "
               "It uses recent averages as inputs, so treat it as an "
               "estimate (model R² ≈ 0.27 on noisy item-level data).")