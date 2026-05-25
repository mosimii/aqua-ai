
# 📦 Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

# 🧠 Defining weighted risk scoring logic
def calculate_weighted_risk(row):
    score = 0
    if 20 <= row['temperature_c'] <= 45:
        score += 3
    if row['flow_rate_lpm'] == 0:
        score += 2
    if row['free_chlorine_mgL'] < 0.2:
        score += 1
    return score

def assign_weighted_risk_band(score):
    if score >= 5:
        return 'High'
    elif score >= 3:
        return 'Medium'
    else:
        return 'Low'

# 📊 Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("aquadataa.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['weighted_risk_score'] = df.apply(calculate_weighted_risk, axis=1)
    df['weighted_risk_band'] = df['weighted_risk_score'].apply(assign_weighted_risk_band)
    return df

df = load_data()

# 🧠 Load pre-trained model (optional)
try:
    model = joblib.load("model.joblib")
except:
    model = None

# 🌐 Streamlit layout
st.set_page_config(page_title="Aqua-AI Legionella Risk Portal", layout="wide")

# st.title("💧 Aqua-AI Legionella Risk Monitoring Portal")
st.markdown(
    "<h1 style='text-align: center;'>💧 Aqua-AI Legionella Risk Monitoring Portal</h1>",
    unsafe_allow_html=True
)

# === OVERVIEW METRICS ===
st.header("📊 Sensor Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total stations", df['station_id'].nunique())

with col2:
    high_risk_count = (df['weighted_risk_band'] == 'High').sum()
    st.metric("High Risk Samples", high_risk_count)

with col3:
    latest = df['timestamp'].max()
    st.metric("Last Data Timestamp", str(latest.date()))



# === Station SELECTION ===
st.header("📍 Location/Station Explorer")

station_id = st.selectbox("Select station", sorted(df['station_id'].unique()))

station_df = df[df['station_id'] == station_id].sort_values("timestamp")

st.write(f"### Risk Summary for {station_id}")
st.dataframe(station_df[['timestamp', 'temperature_c', 'ph', 'free_chlorine_mgL', 'flow_rate_lpm', 'weighted_risk_score', 'weighted_risk_band']].tail(20))



# === AT-RISK STATUS ===
st.subheader("❗️ Station Risk Status")

latest_row = station_df.sort_values("timestamp").iloc[-1]
current_band = latest_row['weighted_risk_band']

if current_band == "High":
    st.error(f"🚨 {station_id} is currently AT HIGH RISK")
elif current_band == "Medium":
    st.warning(f"⚠️ {station_id} is currently at MEDIUM risk")
else:
    st.success(f"✅ {station_id} is currently doing well (LOW risk)")




# 📊 Analyzing last N readings for trend
N = 10
recent = station_df.sort_values('timestamp').tail(N)

high_count = (recent['weighted_risk_band'] == "High").sum()
medium_count = (recent['weighted_risk_band'] == "Medium").sum()
risk_ratio = (high_count + medium_count) / N

# 🕒 Latest reading
latest_row = recent.iloc[-1]
latest_band = latest_row['weighted_risk_band']
latest_time = latest_row['timestamp']

# 🧠 Combine trend and current status
st.subheader("⚠️ Station Risk Trend/Summary")

if risk_ratio >= 0.6:
    if latest_band == "High" or latest_band == "Medium":
        st.error(f"🚨 This station shows consistent risk over the last {N} readings and is still in a risky state. Immediate investigation recommended.")
    else:
        st.warning(f"⚠️ Trend indicates elevated risk ({risk_ratio*100:.0f}%), but the latest reading on {latest_time.strftime('%Y-%m-%d %H:%M')} is safe. Please investigate.")
elif risk_ratio >= 0.3:
    if latest_band == "High":
        st.warning(f"⚠️ Risk levels have been moderate ({risk_ratio*100:.0f}%), but the latest reading is HIGH. Monitor closely.")
    else:
        st.info(f"🔍 Moderate recent risk detected, but latest reading is OK. Consider preventive checks.")
else:
    if latest_band == "Low":
        st.success("✅ This station is consistently safe.")
    else:
        st.warning("🟡 Station was mostly safe recently, but latest reading shows elevated risk. Monitor it.")


# latest readings gauge visual
temp = latest_row['temperature_c']
chlorine = latest_row['free_chlorine_mgL']
flow = latest_row['flow_rate_lpm']
ph = latest_row['ph']

# Get latest station reading and metadata
latest_row = station_df.sort_values("timestamp").iloc[-1]
latest_time = latest_row['timestamp']
station_name = station_id 

# Header with context
st.markdown(f"""
### 🧾 Latest Sensor Readings for **{station_name}**
📅 Timestamp: **{latest_time.strftime('%Y-%m-%d %H:%M')}**
""")


# === Soft, clean color scheme ===
# === Modern color palette ===
SAFE = "#2CA02C"
WARN = "#FF7F0E"
DANGER = "#D62728"
NEUTRAL = "#F5F5F5"
NEEDLE = "#636EFA"

# === Display gauges with improved visuals ===
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['temperature_c'],
        title={'text': "Temperature (°C)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 70], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 20], 'color': NEUTRAL},
                {'range': [20, 45], 'color': WARN},
                {'range': [45, 70], 'color': SAFE}
            ],
            'threshold': {'line': {'color': DANGER, 'width': 4}, 'value': 45}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['free_chlorine_mgL'],
        title={'text': "Free Chlorine (mg/L)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.2], 'color': DANGER},
                {'range': [0.2, 1], 'color': SAFE}
            ],
            'threshold': {'line': {'color': DANGER, 'width': 4}, 'value': 0.2}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['flow_rate_lpm'],
        title={'text': "Flow Rate (LPM)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.01], 'color': DANGER},
                {'range': [0.01, 5], 'color': SAFE}
            ],
            'threshold': {'line': {'color': DANGER, 'width': 4}, 'value': 0.01}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['ph'],
        title={'text': "pH Level", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 14], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [6.5, 7.5], 'color': SAFE},
                {'range': [0, 6.5], 'color': WARN},
                {'range': [7.5, 14], 'color': WARN}
            ],
            'threshold': {'line': {'color': WARN, 'width': 4}, 'value': 7}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)





st.subheader("📅 Risk Band Timeline Per Station By Date")

# Creating a copy with shortened timestamps
timeline_df = station_df.copy()
timeline_df['Time'] = timeline_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

# Plotting with Plotly bar color
fig = px.bar(
    timeline_df,
    x='Time',
    y=[1]*len(timeline_df),  # dummy y-axis just to display blocks
    color='weighted_risk_band',
    color_discrete_map={
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    },
    labels={'weighted_risk_band': 'Risk Band'},
    height=250
)

fig.update_layout(
    showlegend=True,
    yaxis=dict(showticklabels=False),  # hide Y-axis
    xaxis_title='Time',
    title=f"{station_id} - Risk Level Timeline",
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)



# === RISK SEVERITY TIMELINE ===
st.subheader("🧯 Risk Level Over Time")

risk_color_map = {"Low": "green", "Medium": "orange", "High": "red"}
colors = station_df['weighted_risk_band'].map(risk_color_map)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=station_df['timestamp'],
    y=station_df['weighted_risk_score'],
    mode='lines+markers',
    marker=dict(color=colors),
    line=dict(color='gray'),
    name='Risk Score'
))

fig2.update_layout(
    title=f"{station_id} - Risk Score with Band Colors",
    xaxis_title='Time',
    yaxis_title='Risk Score',
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig2, use_container_width=True)



# Risk Distribution
st.subheader("📊 Overall Risk Distribution")

fig = px.pie(
    station_df,
    names='weighted_risk_band',
    title=f"{station_id} - Risk Distribution Summary",
    color='weighted_risk_band',
    color_discrete_map={
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
)

st.plotly_chart(fig, use_container_width=True)


latest_band = station_df.sort_values("timestamp").iloc[-1]['weighted_risk_band']

if latest_band == "High":
    st.markdown("🚨 **This station is currently in a critical condition according to the latest readings. Immediate action is recommended.**")
elif latest_band == "Medium":
    st.markdown("⚠️ **This station shows moderate risk according to the latest readings. Please monitor or take preventive action based on the trend/summary above.**")
else:
    st.markdown("✅ **This station is operating safely according to the latest readings. Check the trends section above for recommented action.**")


# === OPTIONAL SHAP EXPLANATION ===
if model:
    st.subheader("🧠 Explain Model Prediction (SHAP)")
    shap.initjs()

    # Selecting the latest row for explanation
    sample = station_df[["temperature_c", "ph", "free_chlorine_mgL", "flow_rate_lpm"]].iloc[[-1]]

    # Predicting
    prediction = model.predict(sample)[0]
    st.write("**Predicted Risk Band**:", prediction)

    # Explaining with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(
        explainer.expected_value[prediction],
        shap_values[prediction],
        sample,
        matplotlib=True,
        show=True
    )
    st.pyplot(bbox_inches='tight')
