import pandas as pd
import random
from datetime import date, timedelta

random.seed(42)

menu = {
    "Pizza":   [("Margherita", 199), ("Paneer Tikka", 249), ("Farmhouse", 299)],
    "Burger":  [("Veg Burger", 99), ("Cheese Burger", 129), ("Chicken Burger", 149)],
    "Indian":  [("Paneer Butter Masala", 220), ("Dal Tadka", 150), ("Veg Biryani", 180)],
    "Chinese": [("Hakka Noodles", 140), ("Manchurian", 160), ("Fried Rice", 150)],
    "Drinks":  [("Cold Coffee", 90), ("Lassi", 70), ("Soft Drink", 50)],
}
cities = ["Pune", "Mumbai", "Nashik", "Nagpur", "Thane"]
payments = ["UPI", "Cash", "Credit Card", "Debit Card", "Wallet"]
customers = [f"Customer_{i}" for i in range(1, 151)]

# Some customers order much more often than others (repeat customers)
cust_weights = [random.choice([1, 1, 2, 3, 6]) for _ in customers]

# Lunch (12-14) and dinner (19-22) peaks
hours = list(range(10, 24))
hour_w = [3, 8, 12, 14, 8, 3, 3, 4, 8, 12, 14, 10, 5, 2]

# Business grows through the year; Fri-Sun are busier
start = date(2025, 1, 1)
days = [start + timedelta(days=d) for d in range(365)]
day_w = [(1 + d / 365) * (1.5 if days[d].weekday() >= 4 else 1)
         for d in range(365)]

rows = []
for order_id in range(1001, 6001):  # 5000 orders
    cat = random.choices(list(menu), weights=[30, 25, 20, 15, 10])[0]
    # first item in each category is the best-seller
    item, price = random.choices(menu[cat], weights=[5, 3, 1])[0]
    qty = random.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]

    rows.append({
        "OrderID": order_id,
        "Date": random.choices(days, day_w)[0],
        "Hour": random.choices(hours, hour_w)[0],
        "Customer": random.choices(customers, cust_weights)[0],
        "Category": cat,
        "Item": item,
        "Qty": qty,
        "Price": price,
        "Total": qty * price,
        "City": random.choice(cities),
        "Payment": random.choices(payments, weights=[45, 20, 15, 10, 10])[0],
    })

df = pd.DataFrame(rows)
df.to_csv("food_orders.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} orders to food_orders.csv")