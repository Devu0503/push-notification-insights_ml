
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Push Notification Optimizer", layout="wide")

# Load Data & Model
df = pd.read_csv("enhanced_push_notification_data.csv")
with open("models/response_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Add ResponseNumeric column: map 'Yes' to 1, 'No' to 0
if "Response" in df.columns and "ResponseNumeric" not in df.columns:
    df["ResponseNumeric"] = df["Response"].map({"Yes": 1, "No": 0})

# ✅ Create TextLengthBin for binned barplots
if "TextLengthBin" not in df.columns:
    def bin_text_length(length):
        if length < 60:
            return "Short (<60)"
        elif 60 <= length <= 100:
            return "Medium (60-100)"
        else:
            return "Long (>100)"
    df["TextLengthBin"] = df["NotificationTextLength"].apply(bin_text_length)

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
category = st.sidebar.multiselect("Category", df["Category"].unique(), default=list(df["Category"].unique()))
device_type = st.sidebar.multiselect("Device Type", df["DeviceType"].unique(), default=list(df["DeviceType"].unique()))
os = st.sidebar.multiselect("OS", df["OS"].unique(), default=list(df["OS"].unique()))
user_type = st.sidebar.multiselect("User Type", df["UserType"].unique(), default=list(df["UserType"].unique()))
notification_type = st.sidebar.multiselect("Notification Type", df["NotificationType"].unique(), default=list(df["NotificationType"].unique()))

# Apply Filters
filtered_df = df[
    (df["Category"].isin(category)) &
    (df["DeviceType"].isin(device_type)) &
    (df["OS"].isin(os)) &
    (df["UserType"].isin(user_type)) &
    (df["NotificationType"].isin(notification_type))
]

st.title("📊 Push Notification Performance Dashboard")

# EDA Section
with st.expander("📈 EDA Visualizations", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Response Rate by Category")
        fig, ax = plt.subplots()
        sns.barplot(x="Category", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

        st.subheader("Response Rate by Device")
        fig, ax = plt.subplots()
        sns.barplot(x="DeviceType", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

        st.subheader("Response by Notification Type")
        fig, ax = plt.subplots()
        sns.barplot(x="NotificationType", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Response by OS")
        fig, ax = plt.subplots()
        sns.barplot(x="OS", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

        st.subheader("Response by User Type")
        fig, ax = plt.subplots()
        sns.barplot(x="UserType", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

        st.subheader("Response by Text Length Bin")
        fig, ax = plt.subplots()
        sns.barplot(x="TextLengthBin", y="ResponseNumeric", data=filtered_df, ax=ax)
        st.pyplot(fig)

# ML Prediction Panel
st.markdown("---")
st.header("🧠 Predict Notification Response")
with st.form("predict_form"):
    st.subheader("Input Notification Details for Prediction")

    category = st.selectbox("Category", df["Category"].unique())
    user_type = st.selectbox("User Type", df["UserType"].unique())
    device = st.selectbox("Device Type", df["DeviceType"].unique())
    os = st.selectbox("Operating System", df["OS"].unique())
    notif_type = st.selectbox("Notification Type", df["NotificationType"].unique())
    day = st.selectbox("Day of Week", df["DayOfWeek"].unique())
    discount = st.slider("Discount Offered (%)", 0, 100, 20)
    hour = st.slider("Hour of Notification", 0, 23, 12)
    delay = st.slider("Delay before user opens app (min)", 0, 120, 30)
    text_len = st.slider("Notification Text Length", 10, 200, 80)

    submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_data = pd.DataFrame([{
            "Category": category,
            "UserType": user_type,
            "DeviceType": device,
            "OS": os,
            "NotificationType": notif_type,
            "DayOfWeek": day,
            "DiscountOffered": discount,
            "HourOfDay": hour,
            "TimeDelayMin": delay,
            "NotificationTextLength": text_len
        }])

        pred_proba = model.predict_proba(input_data)[0][1]
        label = "✅ Will Respond" if pred_proba > 0.5 else "❌ Won't Respond"
        st.success(f"Prediction: {label} (Confidence: {pred_proba:.2%})")
