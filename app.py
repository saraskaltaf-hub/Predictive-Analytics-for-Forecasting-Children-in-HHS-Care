import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="HHS UAC Predictive Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HHS UAC Predictive Analytics Dashboard")

st.write(
    "Predictive analytics project for analyzing and forecasting "
    "children in HHS Care using historical UAC Program data."
)

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "HHS_Unaccompanied_Alien_Children_Program.csv"
    )

    # Remove completely blank rows
    df = df.dropna(how="all")

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Numeric columns
    numeric_columns = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    for column in numeric_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    df = df.sort_values("Date").reset_index(drop=True)

    return df


df = load_data()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Project Information")

st.sidebar.write(
    "Dataset: HHS Unaccompanied Alien Children Program"
)

st.sidebar.write(
    "Target: Children in HHS Care"
)

# -----------------------------
# Dataset Overview
# -----------------------------

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Observations", len(df))

col2.metric(
    "Start Date",
    df["Date"].min().strftime("%d-%m-%Y")
)

col3.metric(
    "End Date",
    df["Date"].max().strftime("%d-%m-%Y")
)

col4.metric(
    "Average HHS Care",
    f"{df['Children in HHS Care'].mean():,.0f}"
)

# -----------------------------
# HHS Care Trend
# -----------------------------

st.header("2. HHS Care Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df["Children in HHS Care"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Children in HHS Care Over Time")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Transfers vs Discharges
# -----------------------------

st.header("3. Transfers vs Discharges")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df["Children transferred out of CBP custody"],
    label="Transfers"
)

ax.plot(
    df["Date"],
    df["Children discharged from HHS Care"],
    label="Discharges"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("Transfers vs Discharges")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Monthly Average
# -----------------------------

st.header("4. Monthly Average HHS Care")

monthly = (
    df.set_index("Date")
    .resample("ME")["Children in HHS Care"]
    .mean()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(monthly.index, monthly.values)

ax.set_xlabel("Month")
ax.set_ylabel("Average Children")
ax.set_title("Monthly Average HHS Care")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Net Pressure
# -----------------------------

st.header("5. Net Pressure")

df["Net_Pressure"] = (
    df["Children transferred out of CBP custody"]
    -
    df["Children discharged from HHS Care"]
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Date"],
    df["Net_Pressure"]
)

ax.axhline(
    0,
    linestyle="--"
)

ax.set_xlabel("Date")
ax.set_ylabel("Transfers - Discharges")
ax.set_title("Net Pressure Trend")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Machine Learning Features
# -----------------------------

st.header("6. Random Forest Forecasting")

ml_df = df.copy()

target = "Children in HHS Care"

ml_df["lag_1"] = ml_df[target].shift(1)
ml_df["lag_7"] = ml_df[target].shift(7)
ml_df["lag_14"] = ml_df[target].shift(14)

ml_df["rolling_7_mean"] = (
    ml_df[target]
    .shift(1)
    .rolling(7)
    .mean()
)

ml_df["rolling_14_mean"] = (
    ml_df[target]
    .shift(1)
    .rolling(14)
    .mean()
)

ml_df["rolling_7_std"] = (
    ml_df[target]
    .shift(1)
    .rolling(7)
    .std()
)

ml_df["rolling_14_std"] = (
    ml_df[target]
    .shift(1)
    .rolling(14)
    .std()
)

ml_df["day_of_week"] = ml_df["Date"].dt.dayofweek
ml_df["month"] = ml_df["Date"].dt.month
ml_df["year"] = ml_df["Date"].dt.year

ml_df = ml_df.dropna().reset_index(drop=True)

features = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_14_mean",
    "rolling_7_std",
    "rolling_14_std",
    "day_of_week",
    "month",
    "year"
]

X = ml_df[features]
y = ml_df[target]

split_point = int(len(ml_df) * 0.80)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_forecast = rf_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    rf_forecast
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_forecast
    )
)

col1, col2 = st.columns(2)

col1.metric(
    "Random Forest MAE",
    f"{mae:,.2f}"
)

col2.metric(
    "Random Forest RMSE",
    f"{rmse:,.2f}"
)

# -----------------------------
# Actual vs Prediction
# -----------------------------

st.subheader("Actual vs Random Forest Prediction")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    ml_df["Date"].iloc[split_point:],
    y_test.values,
    label="Actual"
)

ax.plot(
    ml_df["Date"].iloc[split_point:],
    rf_forecast,
    linestyle="--",
    label="Random Forest Prediction"
)

ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Actual vs Random Forest Prediction")

ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Random Forest Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title("Random Forest Feature Importance")

ax.invert_yaxis()

plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Model Results
# -----------------------------

st.header("7. Model Comparison")

results = pd.DataFrame({
    "Model": [
        "Naive Forecast",
        "Moving Average",
        "Exponential Smoothing",
        "ARIMA",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        4063.36,
        2849.78,
        2758.92,
        2772.31,
        354.14,
        397.07
    ],
    "RMSE": [
        4933.29,
        3373.27,
        3362.16,
        3365.99,
        448.62,
        504.37
    ],
    "MAPE": [
        140.29,
        80.65,
        74.76,
        72.88,
        6.14,
        7.12
    ]
})

st.dataframe(
    results,
    use_container_width=True
)

st.success(
    "Random Forest achieved the lowest test-set MAE, RMSE, "
    "and MAPE among the evaluated models."
)

st.info(
    "Note: The source data has irregular reporting dates. "
    "Therefore, observation-based lag features do not necessarily "
    "represent exact calendar-day intervals."
)
