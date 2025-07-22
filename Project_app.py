import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import io

# Streamlit settings
st.set_page_config(page_title="CLV Predictor Pro", layout="wide")

st.title("🧮 Customer Lifetime Value (CLV) Predictor Pro")
st.markdown("Predict the future value of your customers using data science techniques.")

# Upload
uploaded_file = st.file_uploader("📤 Upload Online Retail II Excel File", type=["xlsx"])

@st.cache_data
def load_data(file):
    xls = pd.ExcelFile(file)
    df_2009 = xls.parse("Year 2009-2010", dtype={'Customer ID': str})
    df_2010 = xls.parse("Year 2010-2011", dtype={'Customer ID': str})
    return pd.concat([df_2009, df_2010], ignore_index=True)

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully")

        # --- Preprocessing ---
        df.dropna(subset=["Customer ID"], inplace=True)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalAmount"] = df["Quantity"] * df["Price"]
        df["Customer ID"] = df["Customer ID"].astype(str)

        clv = df.groupby("Customer ID").agg({
            "InvoiceDate": [np.min, np.max],
            "Invoice": "nunique",
            "TotalAmount": np.sum
        })
        clv.columns = ["FirstPurchase", "LastPurchase", "Frequency", "Monetary"]
        clv["Duration"] = (clv["LastPurchase"] - clv["FirstPurchase"]).dt.days
        clv["CLV"] = clv["Frequency"] * clv["Monetary"]
        clv.reset_index(inplace=True)

        features = clv[["Frequency", "Monetary", "Duration"]]
        target = clv["CLV"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Tabs layout
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Data", "📈 Model Insights", "🔮 Predict", "🧠 Explainability"])

        with tab1:
            st.subheader("Preview of Raw & Processed Data")
            st.write("🧾 Raw Data Sample", df.head())
            st.write("📊 Aggregated CLV Data", clv.head())

        with tab2:
            st.subheader("Model Performance")
            st.write(f"✅ R2 Score: `{r2_score(y_test, y_pred):.2f}`")
            st.write(f"📉 RMSE: `{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}`")

            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual CLV")
            ax.set_ylabel("Predicted CLV")
            ax.set_title("Actual vs Predicted CLV")
            st.pyplot(fig)

            st.markdown("### 📌 Top Customers by Predicted CLV")
            clv["Predicted_CLV"] = model.predict(features)
            top_customers = clv.sort_values(by="Predicted_CLV", ascending=False).head(5)
            st.dataframe(top_customers[["Customer ID", "Frequency", "Monetary", "Duration", "Predicted_CLV"]])

        with tab3:
            st.subheader("Predict CLV for New Customer")
            freq_input = st.number_input("🧾 Frequency (Invoices)", min_value=1, value=10)
            monetary_input = st.number_input("💰 Monetary Value", min_value=1.0, value=1000.0)
            duration_input = st.number_input("📅 Duration (days)", min_value=1, value=365)

            input_data = pd.DataFrame([[freq_input, monetary_input, duration_input]], columns=["Frequency", "Monetary", "Duration"])
            predicted_clv = model.predict(input_data)[0]
            st.success(f"🎯 Predicted CLV: ₹{predicted_clv:.2f}")

        with tab4:
            st.subheader("SHAP-based Explainability")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            st.markdown("#### 🔍 Feature Impact on Prediction")

            # Convert shap plot to image buffer
            buf = io.BytesIO()
            shap.summary_plot(shap_values, features, show=False)
            plt.savefig(buf, format="png")
            st.image(buf)

    except Exception as e:
        st.error(f"❌ Error: {e}")
