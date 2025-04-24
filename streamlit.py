import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 🚀 Set Streamlit Page Config
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="wide")

# 🎨 Apply Premium Styling Enhancements
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #141e30, #243b55); 
    color: white;
    font-weight: bold;
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
}
h1 {
    color: #ffcc00;
    text-align: center;
    font-weight: bold;
    text-shadow: 2px 2px 7px rgba(0,0,0,0.6);
}
p, label {
    color: #e0e0e0;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 🚀 Ensure Title Visibility
st.title("📱 Mobile Price Predictor")

# Cache Data Loading
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

# Load Dataset
df = load_data()
x = df.iloc[:, :20]  # Ensure all 20 features
y = df["price_range"]

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize Features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Model
model = RandomForestClassifier(max_depth=15, n_estimators=100, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(x_train_scaled, y_train)

# Save Model & Scaler
@st.cache_resource
def load_model():
    with open("mobile_app.pkl", "wb") as file:
        pickle.dump(model, file)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    return scaler

model = load_model()
scaler = load_scaler()

# 🔹 Price Mapping & Images (Replace URLs)
price_map = {
    0: "💰 Low Cost (₹10K - ₹20K)",
    1: "💵 Medium Cost (₹20.5K - ₹35K)",
    2: "💎 High Cost (₹35.5K - ₹70K)",
    3: "👑 Very High Cost (₹70K+)"
}

price_images = {
    0: r"C:\Users\ankit\OneDrive\Desktop\Mobile Price Range Prediction Project\nokia.jpg",
    1: r"C:\Users\ankit\OneDrive\Desktop\Mobile Price Range Prediction Project\redmi.jpg",
    2: r"C:\Users\ankit\OneDrive\Desktop\Mobile Price Range Prediction Project\iphone.jpg",
    3: r"C:\Users\ankit\OneDrive\Desktop\Mobile Price Range Prediction Project\samsung.jpg",
}

# Sidebar - Numerical Inputs
st.sidebar.title("🔧 Mobile Features")
battery_power = st.sidebar.slider("🔋 Battery Power (mAh)", 500, 5000, 2500)
clock_speed = st.sidebar.slider("⏳ Processor Clock Speed (GHz)", 0.5, 3.5, 2.0)
fc = st.sidebar.slider("📸 Front Camera (MP)", 0, 20, 5)
int_memory = st.sidebar.slider("💾 Internal Memory (GB)", 2, 256, 64)
mobile_wt = st.sidebar.slider("⚖️ Mobile Weight (grams)", 80, 250, 150)
ram = st.sidebar.slider("🚀 RAM (MB)", 512, 8000, 4000)
sc_h = st.sidebar.slider("📏 Screen Height (cm)", 5, 20, 15)
sc_w = st.sidebar.slider("📏 Screen Width (cm)", 5, 10, 7)
talk_time = st.sidebar.slider("📞 Talk Time (hours)", 5, 30, 15)

# Sidebar - Categorical Inputs
blue = 1 if st.sidebar.selectbox("📶 Bluetooth?", ["No", "Yes"]) == "Yes" else 0
dual_sim = 1 if st.sidebar.selectbox("📲 Dual SIM?", ["No", "Yes"]) == "Yes" else 0
four_g = 1 if st.sidebar.selectbox("🌐 4G Supported?", ["No", "Yes"]) == "Yes" else 0
three_g = 1 if st.sidebar.selectbox("🌍 3G Supported?", ["No", "Yes"]) == "Yes" else 0
touch_screen = 1 if st.sidebar.selectbox("👆 Touchscreen?", ["No", "Yes"]) == "Yes" else 0
wifi = 1 if st.sidebar.selectbox("📡 WiFi Available?", ["No", "Yes"]) == "Yes" else 0

# 🔹 Fix Missing Features
additional_features = [0, 1, 2, 3, 4]  # Default values for missing features

# Prepare Input Data - Ensure ALL 20 features are included
input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
                        mobile_wt, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi, *additional_features]], dtype=float)

# 🔹 Fix Input Reshaping
input_data_scaled = scaler.transform(input_data.reshape(1, -1))

# Prediction & Image Display
if st.sidebar.button("🔮 Predict Price"):
    try:
        result = model.predict(input_data_scaled)[0]
        st.success(f"### Predicted Price Range: {price_map[result]}")
        st.image(price_images[result], caption=price_map[result], width=400)  # 🔥 Updated Image Size
        st.metric("⚡ Model Accuracy", f"{accuracy_score(y_test, model.predict(x_test_scaled)) * 100:.2f}%")
    except ValueError as e:
        st.error(f"⚠️ Error: {str(e)} - Ensure all features are correctly provided.")