# 🍔 Food Order Analytics & Demand Prediction

An end-to-end data project: generate order data, analyze sales patterns, predict daily item demand with machine learning, and explore everything in an interactive Streamlit dashboard.

> **Note:** The dataset is synthetic. It was generated with Python and designed to mimic realistic ordering patterns (lunch/dinner rush hours, busier weekends, best-selling items, business growth over time).

## 🔍 Problem
A food ordering business needs to know what sells, when it sells, and how much of each item to prepare. This project answers those questions with data.

## 🛠️ Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Plotly, Streamlit

## 📁 Project Structure
```
food_order_analytics/
├── generate_food_orders.py   # creates food_orders.csv (5,000 synthetic orders)
├── analysis.py               # EDA + charts (saved in /charts)
├── demand_model.py           # feature engineering + ML models
├── app.py                    # Streamlit dashboard
├── food_orders.csv
├── charts/
└── requirements.txt
```

## 📊 Key Insights
<!-- Replace these with what YOU see in your charts -->
- Orders peak at lunch (12-2 PM) and dinner (7-10 PM)
- Friday to Sunday get more orders than weekdays
- Monthly revenue grows steadily through the year
- UPI is the most used payment method
- A few best-selling items drive most of the quantity sold

## 🤖 Machine Learning
**Goal:** predict daily quantity sold for each menu item.

**Features:** day of week, month, weekend flag, previous-day demand, demand 7 days ago, 7-day and 30-day rolling averages, item.

**Validation:** time-based split (trained on the past, tested on the last 60 days) to avoid data leakage.

| Model | MAE | R² |
|---|---|---|
| Baseline (7-day average) | 1.60 | 0.15 |
| Linear Regression | 1.58 | 0.27 |
| Random Forest | 1.58 | 0.26 |

**Takeaways:**
- Both models clearly beat the baseline.
- The 30-day rolling average was the most important feature.
- Item-level daily demand is noisy (about 1-2 units per day), so a modest R² is expected. A simple linear model matched the Random Forest.

## 📸 Screenshots
<!-- Add 2-3 screenshots of your dashboard in a /screenshots folder, then: -->
<!-- ![Dashboard](screenshots/dashboard.png) -->

## ▶️ How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the data
python generate_food_orders.py

# 3. (Optional) Run the analysis
python analysis.py

# 4. Train the model (creates demand_model.pkl)
python demand_model.py

# 5. Launch the dashboard
streamlit run app.py
```

## 🚀 Future Improvements
- Predict demand per category or total daily orders (less noisy)
- Add holiday and weather features
- Customer segmentation with KMeans
- Deploy the dashboard on Streamlit Community Cloud

## 👩‍💻 Author
**Saniya Shaikh**
3rd-year AI & Data Science student
[LinkedIn](https://linkedin.com/in/saniya-shaikh-2536a034a) | [GitHub](https://github.com/Shaikh-Saniya-12)