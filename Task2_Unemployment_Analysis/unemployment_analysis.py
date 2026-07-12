"""
TASK 2: Unemployment Analysis with Python
CodeAlpha Data Science Internship

Goal:
- Analyze unemployment rate data (percentage of unemployed people).
- Data cleaning, exploration, and visualization of unemployment trends.
- Investigate the impact of Covid-19 on unemployment rates.
- Identify key patterns / seasonal trends.
- Present insights that could inform economic or social policies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------------------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------------------------------------
df = pd.read_csv("unemployment_data.csv")

# Standardize column names (strip whitespace)
df.columns = [c.strip() for c in df.columns]

# Parse date column
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)

# Drop exact duplicates and rows with missing critical values
df = df.drop_duplicates()
df = df.dropna(subset=["Estimated Unemployment Rate (%)", "Region", "Date"])

# Feature engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")

print("Cleaned dataset shape:", df.shape)
print(df.head(), "\n")
print("Missing values per column:\n", df.isnull().sum(), "\n")
print("Summary statistics:\n", df["Estimated Unemployment Rate (%)"].describe(), "\n")

# -----------------------------------------------------------
# 2. NATIONAL TREND OVER TIME
# -----------------------------------------------------------
national_trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()

plt.figure(figsize=(11, 5))
plt.plot(national_trend["Date"], national_trend["Estimated Unemployment Rate (%)"],
         marker="o", color="#d1495b")
plt.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-06-01"),
            color="grey", alpha=0.2, label="COVID-19 Nationwide Lockdown")
plt.title("Average Estimated Unemployment Rate Over Time (India)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("national_unemployment_trend.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 3. RURAL VS URBAN COMPARISON
# -----------------------------------------------------------
area_trend = df.groupby(["Date", "Area"])["Estimated Unemployment Rate (%)"].mean().reset_index()

plt.figure(figsize=(11, 5))
sns.lineplot(data=area_trend, x="Date", y="Estimated Unemployment Rate (%)", hue="Area", marker="o")
plt.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-06-01"), color="grey", alpha=0.2)
plt.title("Rural vs Urban Unemployment Rate Over Time")
plt.tight_layout()
plt.savefig("rural_vs_urban_trend.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 4. COVID-19 IMPACT: BEFORE vs DURING
# -----------------------------------------------------------
pre_covid = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2020-02-29")]
during_covid = df[(df["Date"] >= "2020-04-01") & (df["Date"] <= "2020-06-30")]
post_peak = df[(df["Date"] >= "2020-07-01") & (df["Date"] <= "2020-10-31")]

pre_avg = pre_covid["Estimated Unemployment Rate (%)"].mean()
covid_avg = during_covid["Estimated Unemployment Rate (%)"].mean()
post_avg = post_peak["Estimated Unemployment Rate (%)"].mean()

print("Average Unemployment Rate:")
print(f"  Pre-COVID (Jan 2019 - Feb 2020):   {pre_avg:.2f}%")
print(f"  During COVID (Apr - Jun 2020):     {covid_avg:.2f}%")
print(f"  Post-peak recovery (Jul - Oct 2020): {post_avg:.2f}%")
print(f"  Increase during lockdown vs pre-COVID: +{covid_avg - pre_avg:.2f} percentage points\n")

plt.figure(figsize=(6, 5))
phases = ["Pre-COVID\n(2019-Feb'20)", "COVID Peak\n(Apr-Jun'20)", "Recovery\n(Jul-Oct'20)"]
values = [pre_avg, covid_avg, post_avg]
bars = plt.bar(phases, values, color=["#2a9d8f", "#e76f51", "#f4a261"])
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%", ha="center")
plt.ylabel("Avg. Unemployment Rate (%)")
plt.title("COVID-19 Impact on Unemployment Rate")
plt.tight_layout()
plt.savefig("covid_impact_comparison.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 5. STATE-WISE IMPACT (TOP AFFECTED REGIONS DURING COVID PEAK)
# -----------------------------------------------------------
state_covid = during_covid.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

plt.figure(figsize=(9, 7))
sns.barplot(x=state_covid.values, y=state_covid.index, palette="Reds_r")
plt.xlabel("Avg. Unemployment Rate (%) during Apr-Jun 2020")
plt.title("State-wise Unemployment Rate During COVID-19 Peak")
plt.tight_layout()
plt.savefig("statewise_covid_impact.png", dpi=150)
plt.close()

print("Top 5 most affected states during COVID peak:")
print(state_covid.head(5), "\n")

# -----------------------------------------------------------
# 6. SEASONAL / MONTHLY PATTERN (using pre-COVID data only, to avoid distortion)
# -----------------------------------------------------------
seasonal = pre_covid.groupby("Month")["Estimated Unemployment Rate (%)"].mean().reindex(range(1, 13))
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

plt.figure(figsize=(9, 5))
plt.plot(month_labels[:len(seasonal.dropna())], seasonal.dropna().values, marker="o", color="#264653")
plt.title("Seasonal Pattern of Unemployment Rate (Pre-COVID Period)")
plt.xlabel("Month")
plt.ylabel("Avg. Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("seasonal_pattern.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 7. LABOUR PARTICIPATION VS UNEMPLOYMENT (CORRELATION)
# -----------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df, x="Estimated Labour Participation Rate (%)",
    y="Estimated Unemployment Rate (%)", hue="Area", alpha=0.6
)
plt.title("Labour Participation Rate vs Unemployment Rate")
plt.tight_layout()
plt.savefig("participation_vs_unemployment.png", dpi=150)
plt.close()

corr = df[["Estimated Unemployment Rate (%)", "Estimated Labour Participation Rate (%)",
           "Estimated Employed"]].corr()
print("Correlation matrix:\n", corr, "\n")

# -----------------------------------------------------------
# 8. SAVE CLEANED DATA
# -----------------------------------------------------------
df.to_csv("unemployment_cleaned.csv", index=False)

print("Analysis complete. Charts saved:")
print(" - national_unemployment_trend.png")
print(" - rural_vs_urban_trend.png")
print(" - covid_impact_comparison.png")
print(" - statewise_covid_impact.png")
print(" - seasonal_pattern.png")
print(" - participation_vs_unemployment.png")
