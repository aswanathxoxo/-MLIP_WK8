import streamlit as st
import numpy as np
import pandas as pd
import joblib
import mlflow
import os

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

st.title("🏡 California House Price Predictor")
st.markdown(
    """
    This app predicts **median house prices** using a model trained and tracked with **MLflow**.  
    The model shown below is the *best-performing* one based on R² score across multiple experiments.
    """
)

# Load best model
MODEL_PATH = "/workspaces/-MLIP_WK8/models/best_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found. Please train models first using `src/train.py`.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Best model loaded successfully!")

# Display MLflow runs summary (optional)
if os.path.exists("../mlruns"):
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp = client.get_experiment_by_name("house_price_prediction")
        if exp:
            runs = client.search_runs([exp.experiment_id], order_by=["metrics.R2 DESC"])
            st.subheader("📈 Top Experiment Runs (from MLflow)")
            df_runs = pd.DataFrame(
                [
                    {
                        "Model": r.data.params.get("model_name", ""),
                        "R2": r.data.metrics.get("R2", 0),
                        "RMSE": r.data.metrics.get("RMSE", 0),
                    }
                    for r in runs[:5]
                ]
            )
            st.dataframe(df_runs)
    except Exception as e:
        st.warning(f"Could not fetch MLflow experiments: {e}")

# Sidebar input
st.sidebar.header("Input Features")

MedInc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
HouseAge = st.sidebar.slider("House Age (years)", 1, 50, 20)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100, 5000, 800)
AveOccup = st.sidebar.slider("Average Occupants", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.5)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Prediction
if st.button("🔮 Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.subheader("🏠 Predicted Median House Value")
    st.metric(label="Estimated Price", value=f"${prediction * 100000:,.2f}")