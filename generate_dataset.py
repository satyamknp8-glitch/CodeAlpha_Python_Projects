"""
Generates a realistic, representative unemployment dataset modeled on the
publicly known 'Unemployment in India' dataset structure (Region, Date,
Estimated Unemployment Rate %, Estimated Employed, Estimated Labour
Participation Rate %, Area: Rural/Urban), spanning Jan 2019 - Oct 2020,
including the well documented COVID-19 lockdown spike (Apr-Jun 2020).

NOTE: The original file could not be downloaded directly in this environment,
so this script builds a statistically realistic stand-in with the same
columns and the same real-world pattern (sharp unemployment spike during the
Apr-Jun 2020 lockdown). If you have the actual CSV from the link provided in
your task sheet, just place it as 'unemployment_data.csv' in this folder and
skip this generator - the analysis script will use it as-is.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

states = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Gujarat",
    "Haryana", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu",
    "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

areas = ["Rural", "Urban"]

dates = pd.date_range("2019-01-31", "2020-10-31", freq="ME")

rows = []
for state in states:
    # each state has its own baseline unemployment level
    base_rate = np.random.uniform(4.0, 10.0)
    for area in areas:
        area_adj = 1.3 if area == "Urban" else 0.85  # urban tends higher
        labour_base = np.random.uniform(38, 46)
        employed_base = np.random.uniform(8_000_000, 25_000_000)
        for date in dates:
            month = date.month
            year = date.year

            # mild seasonal wave (higher unemployment in summer months, lower post-harvest)
            seasonal = 0.8 * np.sin((month - 3) / 12 * 2 * np.pi)

            # COVID lockdown shock: huge spike Apr-Jun 2020, easing after
            covid_shock = 0
            if year == 2020 and month in (4, 5):
                covid_shock = np.random.uniform(18, 30)
            elif year == 2020 and month == 6:
                covid_shock = np.random.uniform(8, 14)
            elif year == 2020 and month in (7, 8):
                covid_shock = np.random.uniform(3, 6)
            elif year == 2020 and month in (9, 10):
                covid_shock = np.random.uniform(1, 2.5)

            noise = np.random.normal(0, 0.6)
            rate = max(0.5, base_rate * area_adj + seasonal + covid_shock + noise)

            # employment falls when unemployment rises
            employment_drop_factor = 1 - min(rate / 100 * 2.5, 0.35)
            employed = employed_base * employment_drop_factor * np.random.uniform(0.97, 1.03)

            labour_participation = max(30, labour_base - covid_shock * 0.3 + np.random.normal(0, 0.4))

            rows.append({
                "Region": state,
                "Date": date.strftime("%d-%m-%Y"),
                "Frequency": "Monthly",
                "Estimated Unemployment Rate (%)": round(rate, 2),
                "Estimated Employed": int(employed),
                "Estimated Labour Participation Rate (%)": round(labour_participation, 2),
                "Area": area,
            })

df = pd.DataFrame(rows)
df.to_csv("unemployment_data.csv", index=False)
print("Generated unemployment_data.csv with shape:", df.shape)
print(df.head())
