import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ✅ Load Dataset with Error Handling
@st.cache_data
def load_data():
    file_path = "GCB2022v27_MtCO2_flat.csv"
    if not os.path.exists(file_path):
        st.error(f"Dataset file '{file_path}' not found! Please check the file location.")
        return None
    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ✅ Define Sustainability Tips
def get_sustainability_tips(total_emission):
    if total_emission > 5000:
        return [
            "💡 **Invest in renewable energy** like solar and wind.",
            "🚆 **Improve public transport** to reduce vehicle emissions.",
            "🌳 **Plant more trees** for carbon offset.",
            "🏭 **Adopt carbon capture** technologies in industries.",
            "🔋 **Use energy-efficient appliances** and smart grids."
        ]
    elif 1000 <= total_emission <= 5000:
        return [
            "🚲 **Encourage cycling** and public transport.",
            "🏡 **Upgrade buildings** for better energy efficiency.",
            "🌱 **Support local & organic food** to reduce footprints.",
            "⚡ **Adopt electric vehicles** for eco-friendly transport."
        ]
    else:
        return [
            "🌍 **Continue using sustainable energy** sources.",
            "🔄 **Improve waste management** and recycling.",
            "🚀 **Innovate in green technology** and solutions.",
            "🏡 **Promote local green initiatives** and awareness."
        ]


# ✅ Train Model & Predict Emissions
def train_and_predict(df, country, model_type):
    country_df = df[df["Country"] == country]
    if country_df.empty:
        return None, None, None

    # Prepare Data
    X = country_df[["Year"]].values
    y = country_df["Total"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select Model
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        return None, None, None

    # Train & Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    future_year = np.array([[country_df["Year"].max() + 1]])
    future_emission = model.predict(future_year)[0]

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return future_emission, mae, r2


# ✅ Main Streamlit UI
def main():
    st.set_page_config(page_title="Carbon Emissions & Sustainability", layout="wide")

    # 🎨 Custom CSS Styling
    st.markdown("""
        <style>
            .main { background-color: #F5F5F5; }
            h1 { color: #4CAF50; text-align: center; font-size: 40px; }
            .css-1d391kg { background-color: #E8F5E9 !important; padding: 20px; border-radius: 10px; }
            .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; }
            .stSelectbox div[data-baseweb="select"] { background-color: #FFFFFF; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

    # 🏆 **Header**
    st.markdown("<h1>🌍 Carbon Emission Insights & Sustainability Tips</h1>", unsafe_allow_html=True)

    # 🔄 Load Data
    df = load_data()
    if df is None:
        return

    # 📌 Sidebar with Country & Model Selection
    with st.sidebar:
        st.markdown("## 🌎 Select a Country")
        countries = sorted(df["Country"].unique())
        country_selected = st.selectbox("Choose a Country", countries)

        st.markdown("## 🔍 Select Model")
        model_selected = st.selectbox("Choose a Model", ["Linear Regression", "Random Forest", "XGBoost"])

        st.markdown("---")
        st.markdown("**🌱 Reduce Your Carbon Footprint & Save the Planet!**")

    # 📊 Filter dataset for selected country
    country_df = df[df["Country"] == country_selected]
    if country_df.empty:
        st.error(f"No data available for {country_selected}")
        return

    # ✅ Show Carbon Emission Trend
    st.markdown(f"### 📊 Carbon Emission Trends for **{country_selected}**")

    # 🎨 Seaborn Styled Graph
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=country_df, x="Year", y="Total", marker="o", color="red", linewidth=2.5)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Emissions (MtCO2)", fontsize=12)
    plt.title(f"Carbon Emission Trends for {country_selected}", fontsize=14, fontweight="bold")
    plt.grid(True)
    st.pyplot(plt)

    # ✅ Get Latest Emission Data
    latest_year = country_df["Year"].max()
    latest_emission = country_df[country_df["Year"] == latest_year]["Total"].values[0]

    # 🚀 Predict Future Emissions
    future_emission, mae, r2 = train_and_predict(df, country_selected, model_selected)

    # 🏆 Display Results in Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🌎 **Latest Year: {latest_year}**")
    with col2:
        st.info(f"📉 **Current Emissions: {latest_emission:.2f} MtCO2**")
    with col3:
        if future_emission:
            st.warning(f"🔮 **Predicted Emissions ({latest_year + 1}): {future_emission:.2f} MtCO2**")

    # ✅ Display Accuracy Metrics
    if future_emission:
        st.markdown("### 🔍 Model Performance")
        st.write(f"📌 **Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"📌 **R² Score:** {r2:.2f}")

    # ✅ Display Sustainability Tips
    st.markdown("### 🌱 Sustainability Tips to Reduce Emissions")
    tips = get_sustainability_tips(future_emission if future_emission else latest_emission)

    for tip in tips:
        st.markdown(f"- {tip}")

    # 🚀 Call to Action
    st.markdown(
        "<h3 style='text-align: center; color: #2E7D32;'>🌎 Act Now! Every Action Counts for a Greener Planet! 🌱</h3>",
        unsafe_allow_html=True)

#python -m streamlit run app.py

if __name__ == "__main__":
    main()
#
