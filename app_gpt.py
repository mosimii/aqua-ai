# Importing necessary libraries
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Defining weighted risk scoring logic
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

# Loading data
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

# === CUSTOM STYLING ===
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        .main {
            background-color: #161B22;
            padding: 1rem;
            border-radius: 8px;
        }
        h1, h2, h3, h4 {
            color: #00B2A9;
        }
        .stMetric {
            background-color: #1B2A41;
            padding: 0.5rem;
            border-radius: 0.5rem;
        }
        .stMarkdown, .stMarkdown p, .stText, .stApp p, .stApp span, .stApp li, .stApp label {
            color: #E0E0E0 !important;
        }
        .stSelectbox label, .stTextInput label, .stMultiSelect label {
            color: #E0E0E0 !important;
        }
        .stAlert p {
            color: inherit !important;
        }
        a[name] {
            display: block;
            position: relative;
            top: -80px;
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


# === SIDEBAR ===
with st.sidebar:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #1B2A41;
                color: white;
            }
            .sidebar-title {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .sidebar-link a {
                color: #00B2A9;
                text-decoration: none;
                font-size: 16px;
            }
            .sidebar-link a:hover {
                text-decoration: underline;
                color: #00FFF7;
            }
            .external-link a {
                color: white !important;
            }
            .external-link a:hover {
                color: #00B2A9 !important;
            }
        </style>
        <div class='sidebar-title'>💧 Aqua-AI</div>
        <div class='sidebar-link'><a href="#overview">📊 Overview</a></div>
        <div class='sidebar-link'><a href="#explorer">📍 Station Explorer</a></div>
        <div class='sidebar-link'><a href="#trends">📈 Sensor Trends</a></div>
        <div class='sidebar-link'><a href="#rag">🧠 Risk Explanation</a></div>
        <div class='sidebar-link'><a href="#animated">🎬 Animated Trends</a></div>
        <div class='sidebar-link'><a href="#correlation">🫧 Correlation Explorer</a></div>
        <div class='sidebar-link'><a href="#riskscore">📈 Risk Score Over Time</a></div>
        <div class='sidebar-link'><a href="#compare">🔀 Stations Comparison</a></div>
        <div class='sidebar-link'><a href="#download-report">📥 Download Report</a></div>
    """, unsafe_allow_html=True)







# === OVERVIEW METRICS ===
st.markdown('<a name="overview"></a>', unsafe_allow_html=True)
st.header("📊 Sensor Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total stations", df['station_id'].nunique())

with col2:
    high_risk_count = (df['weighted_risk_band'] == 'High').sum()
    st.metric("High Risk Readings", high_risk_count)

with col3:
    latest = df['timestamp'].max()
    st.metric("Last Data Timestamp", str(latest.date()))

# === Station SELECTION ===
st.markdown('<a name="explorer"></a>', unsafe_allow_html=True)
st.header("📍 Location/Station Explorer")

station_id = st.selectbox("Select station", sorted(df['station_id'].unique()))

station_df = df[df['station_id'] == station_id].sort_values("timestamp")

# st.write(f"### Risk Summary for {station_id}")
# st.dataframe(station_df[['timestamp', 'temperature_c', 'ph', 'free_chlorine_mgL', 'flow_rate_lpm', 'weighted_risk_score', 'weighted_risk_band']].tail(20))

styled_df = station_df[['timestamp', 'temperature_c', 'ph', 'free_chlorine_mgL',
                        'flow_rate_lpm', 'weighted_risk_score', 'weighted_risk_band']].tail(20)

# ✅ Rename columns for cleaner display
styled_df = styled_df.rename(columns={
    'timestamp': 'Timestamp',
    'temperature_c': 'Temperature (°C)',
    'ph': 'pH',
    'free_chlorine_mgL': 'Free Chlorine (mg/L)',
    'flow_rate_lpm': 'Flow Rate (LPM)',
    'weighted_risk_score': 'Risk Score',
    'weighted_risk_band': 'Risk Band'
})

# ✅ Style: Dark blue background, white text
def dark_theme_style(row):
    return ['background-color: #1B2A41; color: white;'] * len(row)

# ✅ Apply styling + optional formatting
styled_df = styled_df.style \
    .apply(dark_theme_style, axis=1) \
    .format({
        'Temperature (°C)': '{:.2f}',
        'pH': '{:.2f}',
        'Free Chlorine (mg/L)': '{:.3f}',
        'Flow Rate (LPM)': '{:.3f}',
        'Risk Score': '{:.1f}'
    })

# ✅ Render as scrollable table
st.subheader(f"📋 Risk Summary for {station_id}")
st.dataframe(styled_df, use_container_width=True, height=400)

# Prepare CSV for download
csv = station_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Station Data (CSV)",
    data=csv,
    file_name=f"{station_id}_sensor_data.csv",
    mime='text/csv'
)

# TABLE STYLING
# def hover_style(row):
#     default_bg = 'background-color: #2CA4C2; color: white;'  # base dashboard color
#     hover_styles = []

#     risk_band = row['weighted_risk_band']
#     if risk_band == 'High':
#         hover_color = '#D62728'
#     elif risk_band == 'Medium':
#         hover_color = '#FF7F0E'
#     elif risk_band == 'Low':
#         hover_color = '#2CA02C'
#     else:
#         hover_color = '#2CA4C2'

#     for _ in row:
#         hover_styles.append(f"{default_bg} transition: background-color 0.3s ease-in-out;")

#     return hover_styles

# styled_df = (
#     station_df[['timestamp', 'temperature_c', 'ph', 'free_chlorine_mgL',
#                 'flow_rate_lpm', 'weighted_risk_score', 'weighted_risk_band']]
#     .tail(20)
#     .style
#     .apply(hover_style, axis=1)
#     .set_table_styles([
#         {
#             'selector': 'tbody tr:hover',
#             'props': [('background-color', '#FFD700'), ('color', 'black')]
#         },
#         {
#             'selector': 'thead th',
#             'props': [('background-color', '#1B2A41'), ('color', 'white')]
#         },
#         {
#             'selector': '',
#             'props': [('border-radius', '10px')]
#         }
#     ])
# )

# st.dataframe(styled_df, use_container_width=True)




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
st.markdown('<a name="trends"></a>', unsafe_allow_html=True)
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
NEEDLE = "#02030B"
GAUGE_BG = "#1B2A41"  # or use "#1B2A41" for darker contrast

# === Display gauges with themed visuals ===
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['temperature_c'],
        title={'text': "Temperature (°C)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 70], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': GAUGE_BG,
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
    fig.update_layout(paper_bgcolor=GAUGE_BG, plot_bgcolor=GAUGE_BG)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['free_chlorine_mgL'],
        title={'text': "Free Chlorine (mg/L)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': GAUGE_BG,
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.2], 'color': DANGER},
                {'range': [0.2, 1], 'color': SAFE}
            ],
            'threshold': {'line': {'color': DANGER, 'width': 4}, 'value': 0.2}
        }
    ))
    fig.update_layout(paper_bgcolor=GAUGE_BG, plot_bgcolor=GAUGE_BG)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['flow_rate_lpm'],
        title={'text': "Flow Rate (LPM)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': GAUGE_BG,
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.01], 'color': DANGER},
                {'range': [0.01, 5], 'color': SAFE}
            ],
            'threshold': {'line': {'color': DANGER, 'width': 4}, 'value': 0.01}
        }
    ))
    fig.update_layout(paper_bgcolor=GAUGE_BG, plot_bgcolor=GAUGE_BG)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_row['ph'],
        title={'text': "pH Level", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 14], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': NEEDLE, 'thickness': 0.3},
            'bgcolor': GAUGE_BG,
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
    fig.update_layout(paper_bgcolor=GAUGE_BG, plot_bgcolor=GAUGE_BG)
    st.plotly_chart(fig, use_container_width=True)




# Inject station + sensor context into RAG prompt
latest_row = station_df.iloc[-1]  # gets the most recent reading

@st.cache_data(show_spinner="Loading HSE...")
def fetch_hse_text():
    url = "https://www.hse.gov.uk/healthservices/legionella.htm"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])

@st.cache_data(show_spinner="Loading HSE...")
def fetch_hse_text2():
    url = "https://www.hse.gov.uk/legionnaires/legionella-landlords-responsibilities.htm"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])

@st.cache_data(show_spinner="Loading HSE L8 Approved Code of Practice...")
def fetch_hse_l8_text():
    url = "https://www.hse.gov.uk/legionnaires/what-you-must-do/index.htm"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])

@st.cache_data(show_spinner="Loading HSE HSG274 Guidance...")
def fetch_hse_hsg274_text():
    url = "https://www.hse.gov.uk/legionnaires/what-you-must-do/control-scheme.htm"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])

@st.cache_data(show_spinner="Loading CDC Legionella Toolkit...")
def fetch_cdc_text():
    url = "https://www.cdc.gov/legionella/wmp/toolkit/index.html"
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])

@st.cache_data(show_spinner="Loading GOV.UK...")
def fetch_gov_text():
    url = "https://www.gov.uk/government/publications/guidance-for-organisations-on-supplying-safe-water-supplies/a-safe-water-supply-information-for-all-organisations"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])


@st.cache_data(show_spinner="Loading MN Health Dept PDF...")
def fetch_us_health_text():
    pdf_path = "legionella_mn.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

@st.cache_data(show_spinner="Loading Italian Research Paper...")
def fetch_paper_text():
    pdf_path = "De_Giglio_2025.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

VECTORSTORE_DIR = "chroma_db"

@st.cache_resource(show_spinner="Building vector database...")
def build_vectorstore():
    embeddings = OpenAIEmbeddings()

    # Reuse persisted vectorstore if it exists
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        return vectordb

    sources = [
        ("HSE", fetch_hse_text()),
        ("HSE", fetch_hse_text2()),
        ("HSE L8", fetch_hse_l8_text()),
        ("HSE HSG274", fetch_hse_hsg274_text()),
        ("CDC", fetch_cdc_text()),
        ("GOV.UK", fetch_gov_text()),
        ("MN Dept Health", fetch_us_health_text()),
        ("De Giglio 2025", fetch_paper_text())
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_docs, metadatas = [], []

    for name, content in sources:
        chunks = splitter.split_text(content)
        all_docs.extend(chunks)
        metadatas.extend([{"source": name}] * len(chunks))

    vectordb = Chroma.from_texts(
        texts=all_docs,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=VECTORSTORE_DIR
    )
    return vectordb


# === GPT-4 Risk Explanation (RAG) ===
st.markdown('<a name="rag"></a>', unsafe_allow_html=True)
st.markdown("---")
st.subheader("🧠 Why Is This Site Risky (or NOT)? (GPT-4 + Expert Guidance)")

try:
    # ✅ Pull latest risk band
    risk_band = latest_row['weighted_risk_band']

    # ✅ Generate a meaningful default question
    if risk_band == "High":
        default_query = f"Why is {station_id} at high risk of Legionella formation?"
    elif risk_band == "Medium":
        default_query = f"What risk factors are currently present at {station_id}?"
    else:  # Low
        default_query = f"Is {station_id} operating safely, or are there any warning signs?"

    st.markdown(f"### Current Risk Band: **{risk_band}**")

    # ✅ Build sensor context
    sensor_context = f"""
Latest readings for {station_id}:
- Temperature: {latest_row['temperature_c']} °C
- Free Chlorine: {latest_row['free_chlorine_mgL']} mg/L
- Flow Rate: {latest_row['flow_rate_lpm']} LPM
- pH: {latest_row['ph']}
"""

    # ✅ Let user enter or refine question
    query = st.text_input("Ask a risk-related question:", value=default_query)

    if query and len(query.strip()) > 0:
        vectordb = build_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})

        llm = ChatOpenAI(temperature=0, model="gpt-4")

        from langchain.prompts import PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Legionella risk assessment expert. You have access to official regulatory guidance from the HSE, CDC, GOV.UK, and published research.

When answering, you MUST:
1. Directly assess the sensor readings provided against regulatory thresholds (e.g. HSE L8 states temperatures between 20-45°C promote Legionella growth, free chlorine below 0.2 mg/L is insufficient, stagnant water with zero flow is a risk factor).
2. Give a clear verdict on whether the station is safe, at risk, or in danger, do not hedge or say "it depends" without first evaluating the data.
3. Reference specific guidance documents and thresholds from the context provided.
4. Recommend concrete next steps based on the risk level.
5. Keep the response concise and actionable.

Context from regulatory documents:
{context}

Question with sensor data:
{question}

Answer:"""
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        with st.spinner("🔎 Reasoning with GPT-4..."):
            full_query = f"{sensor_context}\n\n{query}"
            result = qa({"query": full_query})

        # ✅ Post-process the result to enrich it with helpful links if needed
        response_text = result["result"]

        if "chlorine" in response_text.lower() and "not provided" in response_text.lower():
            response_text += "\n\n🔗 **Free chlorine is critical for Legionella prevention.** Learn more here: [HSE Chlorine Guidance](https://www.hse.gov.uk/legionnaires/control.htm#monitoring)"

        if "consult with a water safety expert" in response_text.lower() or "recommend consulting" in response_text.lower():
            response_text += "\n\n📞 **Need expert advice?** Consider contacting a qualified water safety specialist for a professional risk assessment."

        # ✅ Display the final answer with enhancements
        st.markdown("### 💡 Answer:")
        st.write(response_text)

        # ✅ Show sources
        st.markdown("### 📚 Sources Used:")
        used_sources = set(doc.metadata['source'] for doc in result["source_documents"])
        for src in used_sources:
            st.markdown(f"- **{src}** ✅")

        with st.expander("📄 View Extracted Source Chunks"):
            for doc in result["source_documents"]:
                st.markdown(f"**{doc.metadata['source']}**")
                st.code(doc.page_content.strip()[:800])

        # ✅ Always show helpful resources
        st.markdown("---")
        st.markdown("🔗 **Helpful Resources**")
        st.markdown("- [HSE Legionella Risk Guide](https://www.hse.gov.uk/legionnaires/what-you-must-do/identify-assess-sources-risk.htm)")
        st.markdown("- [WHO Legionella Guidelines](https://iris.who.int/bitstream/handle/10665/43233/9241562978_eng.pdf?sequence=1)")

except NameError:
    st.warning("Please select a station first to view risk explanation.")







# visual 
st.subheader("📅 Risk Band Timeline Per Station By Date")
st.caption("Each bar is a reading, coloured by risk level. A wall of green is good news. Red bars popping up means it is time to investigate.")

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
    yaxis=dict(showticklabels=False),
    xaxis_title='Time',
    title=f"{station_id} - Risk Level Timeline",
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor='#161B22',
    plot_bgcolor='#1B2A41',
    font=dict(color='#E0E0E0')
)

st.plotly_chart(fig, use_container_width=True)



# === RISK SEVERITY TIMELINE ===
# st.subheader("🧯 Risk Level Over Time")

# risk_color_map = {"Low": "green", "Medium": "orange", "High": "red"}
# colors = station_df['weighted_risk_band'].map(risk_color_map)

# fig2 = go.Figure()

# fig2.add_trace(go.Scatter(
#     x=station_df['timestamp'],
#     y=station_df['weighted_risk_score'],
#     mode='lines+markers',
#     marker=dict(color=colors),
#     line=dict(color='gray'),
#     name='Risk Score'
# ))

# fig2.update_layout(
#     title=f"{station_id} - Risk Score with Band Colors",
#     xaxis_title='Time',
#     yaxis_title='Risk Score',
#     height=300,
#     margin=dict(l=20, r=20, t=40, b=20)
# )

# st.plotly_chart(fig2, use_container_width=True)



# Risk Distribution
st.subheader("📊 Overall Risk Distribution")
st.caption("How much of the time has this station been safe, moderate, or high risk? The bigger the red slice, the bigger the problem.")

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
fig.update_layout(
    paper_bgcolor='#161B22',
    font=dict(color='#E0E0E0')
)

st.plotly_chart(fig, use_container_width=True)


# === ANIMATED SENSOR TRENDS ===
st.markdown('<a name="animated"></a>', unsafe_allow_html=True)
st.subheader("🎬 Animated Sensor Trends Over Time")
st.caption("Pick a sensor and hit play to watch it change over time. If the line drifts into the red shaded area, that sensor is outside safe limits.")

sensor_anim_df = station_df[['timestamp', 'temperature_c', 'free_chlorine_mgL', 'flow_rate_lpm', 'ph']].copy()
sensor_anim_df = sensor_anim_df.sort_values('timestamp').reset_index(drop=True)

sensor_choice = st.selectbox("Select sensor to animate:", [
    "Temperature (°C)", "Free Chlorine (mg/L)", "Flow Rate (LPM)", "pH Level"
])

sensor_map = {
    "Temperature (°C)": ("temperature_c", '#FF7F0E', 20, 45, "Legionella Growth Zone (20-45°C)"),
    "Free Chlorine (mg/L)": ("free_chlorine_mgL", '#2CA02C', 0, 0.2, "Insufficient Chlorine (<0.2 mg/L)"),
    "Flow Rate (LPM)": ("flow_rate_lpm", '#0072CE', None, None, None),
    "pH Level": ("ph", '#9467BD', None, None, None)
}

col_name, color, danger_lo, danger_hi, danger_label = sensor_map[sensor_choice]

fig_anim = go.Figure()

# Build frames - each frame shows data up to that point
frames = []
step = max(1, len(sensor_anim_df) // 50)  # cap at ~50 frames for performance
frame_indices = list(range(step, len(sensor_anim_df) + 1, step))
if frame_indices[-1] != len(sensor_anim_df):
    frame_indices.append(len(sensor_anim_df))

for i in frame_indices:
    frame_data = sensor_anim_df.iloc[:i]
    frames.append(go.Frame(
        data=[go.Scatter(
            x=frame_data['timestamp'],
            y=frame_data[col_name],
            mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)'
        )],
        name=str(i)
    ))

# Initial trace (full data as background in light gray)
fig_anim.add_trace(go.Scatter(
    x=sensor_anim_df['timestamp'],
    y=sensor_anim_df[col_name],
    mode='lines',
    line=dict(color='rgba(200,200,200,0.3)', width=1, dash='dot'),
    showlegend=False,
    hoverinfo='skip'
))

# Animated trace (starts empty)
fig_anim.add_trace(go.Scatter(
    x=[], y=[],
    mode='lines+markers',
    line=dict(color=color, width=2.5),
    marker=dict(size=5, color=color),
    fill='tozeroy',
    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)',
    name=sensor_choice
))

# Add danger zone if applicable
if danger_lo is not None:
    fig_anim.add_hrect(
        y0=danger_lo, y1=danger_hi,
        fillcolor="red", opacity=0.08, line_width=0,
        annotation_text=danger_label,
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="red"
    )

# Override frames to update only the second trace
for frame in frames:
    frame.traces = [1]

fig_anim.frames = frames

fig_anim.update_layout(
    height=420,
    title=f"{station_id} - {sensor_choice} Building Over Time",
    xaxis_title="Time",
    yaxis_title=sensor_choice,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.12,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True, "transition": {"duration": 20}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
        ]
    )],
    margin=dict(l=40, r=40, t=80, b=40),
    paper_bgcolor='#161B22',
    font=dict(color='#E0E0E0'),
    plot_bgcolor='#1B2A41',
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.05),
    )
)

st.plotly_chart(fig_anim, use_container_width=True)


# === SENSOR CORRELATION BUBBLE CHART ===
st.markdown('<a name="correlation"></a>', unsafe_allow_html=True)
st.subheader("🫧 Sensor Correlation Explorer")
st.caption("This plots temperature against chlorine for every reading. Bubbles that land in the red box mean both are in a bad place at the same time, which is when Legionella risk is highest.")

bubble_df = station_df.copy()
bubble_df['risk_color'] = bubble_df['weighted_risk_band'].map({'Low': '#2CA02C', 'Medium': '#FF7F0E', 'High': '#D62728'})

fig_bubble = px.scatter(
    bubble_df,
    x='temperature_c',
    y='free_chlorine_mgL',
    size='weighted_risk_score',
    size_max=20,
    color='weighted_risk_band',
    color_discrete_map={'Low': '#2CA02C', 'Medium': '#FF7F0E', 'High': '#D62728'},
    hover_data={
        'temperature_c': ':.2f',
        'free_chlorine_mgL': ':.3f',
        'flow_rate_lpm': ':.2f',
        'ph': ':.2f',
        'weighted_risk_score': True,
        'weighted_risk_band': True
    },
    labels={
        'temperature_c': 'Temperature (°C)',
        'free_chlorine_mgL': 'Free Chlorine (mg/L)',
        'weighted_risk_band': 'Risk Band'
    },
    title=f"{station_id} - Temperature vs Chlorine (bubble size = risk score)",
    animation_frame=bubble_df['timestamp'].dt.strftime('%Y-%m-%d') if len(bubble_df['timestamp'].dt.date.unique()) > 1 else None,
    height=450
)

# Add danger zone rectangle
fig_bubble.add_shape(
    type="rect", x0=20, x1=45, y0=0, y1=0.2,
    fillcolor="red", opacity=0.06, line=dict(color="red", dash="dot", width=1),
    layer="below"
)
fig_bubble.add_annotation(
    x=32.5, y=0.1, text="⚠️ High Risk Zone",
    showarrow=False, font=dict(color="red", size=11)
)

fig_bubble.update_layout(
    paper_bgcolor='#161B22',
    font=dict(color='#E0E0E0'),
    plot_bgcolor='#1B2A41',
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_bubble, use_container_width=True)


# === RISK SCORE OVER TIME ===
st.markdown('<a name="riskscore"></a>', unsafe_allow_html=True)
st.subheader("📈 Risk Score Over Time")
st.caption("One line, one story. When it sits in the green zone, the station is fine. When it climbs into orange or red, something needs fixing.")

risk_timeline_df = station_df[['timestamp', 'weighted_risk_score']].copy().sort_values('timestamp')

fig_risk = go.Figure()

# Background colour bands
fig_risk.add_hrect(y0=0, y1=2.9, fillcolor="green", opacity=0.12, line_width=0, annotation_text="Safe", annotation_position="top left", annotation_font_color="green")
fig_risk.add_hrect(y0=3, y1=4.9, fillcolor="orange", opacity=0.12, line_width=0, annotation_text="Caution", annotation_position="top left", annotation_font_color="orange")
fig_risk.add_hrect(y0=5, y1=6, fillcolor="red", opacity=0.12, line_width=0, annotation_text="Danger", annotation_position="top left", annotation_font_color="red")

# Risk score line
fig_risk.add_trace(go.Scatter(
    x=risk_timeline_df['timestamp'],
    y=risk_timeline_df['weighted_risk_score'],
    mode='lines+markers',
    line=dict(color='#00B2A9', width=2.5),
    marker=dict(size=6, color='#00B2A9'),
    name='Risk Score',
    hovertemplate='Time: %{x}<br>Risk Score: %{y}<extra></extra>'
))

fig_risk.update_layout(
    title=f"{station_id} - Risk Score Trend",
    xaxis_title="Time",
    yaxis_title="Risk Score",
    yaxis=dict(range=[0, 6.5]),
    height=400,
    margin=dict(l=40, r=40, t=60, b=40),
    paper_bgcolor='#161B22',
    plot_bgcolor='#1B2A41',
    font=dict(color='#E0E0E0')
)

st.plotly_chart(fig_risk, use_container_width=True)

# === ROLLING RISK HEATMAP ===
# st.subheader("🌡️ Risk Heatmap Over Time")

# heatmap_df = station_df[['timestamp', 'temperature_c', 'free_chlorine_mgL', 'flow_rate_lpm', 'ph', 'weighted_risk_score']].copy()
# heatmap_df = heatmap_df.sort_values('timestamp')
# heatmap_df['time_label'] = heatmap_df['timestamp'].dt.strftime('%m-%d %H:%M')

# # Normalise each sensor to 0-1 for comparable heatmap
# for col in ['temperature_c', 'free_chlorine_mgL', 'flow_rate_lpm', 'ph']:
#     col_min = heatmap_df[col].min()
#     col_max = heatmap_df[col].max()
#     if col_max > col_min:
#         heatmap_df[f'{col}_norm'] = (heatmap_df[col] - col_min) / (col_max - col_min)
#     else:
#         heatmap_df[f'{col}_norm'] = 0.5

# z_data = heatmap_df[['temperature_c_norm', 'free_chlorine_mgL_norm', 'flow_rate_lpm_norm', 'ph_norm']].T.values
# y_labels = ['Temperature', 'Free Chlorine', 'Flow Rate', 'pH']

# fig_heat = go.Figure(data=go.Heatmap(
#     z=z_data,
#     x=heatmap_df['time_label'].values,
#     y=y_labels,
#     colorscale=[
#         [0, '#2CA02C'],
#         [0.5, '#FFD700'],
#         [1, '#D62728']
#     ],
#     hovertemplate='Sensor: %{y}<br>Time: %{x}<br>Normalised Value: %{z:.2f}<extra></extra>',
#     colorbar=dict(title="Intensity")
# ))

# fig_heat.update_layout(
#     title=f"{station_id} - Sensor Intensity Heatmap",
#     xaxis_title="Time",
#     yaxis_title="Sensor",
#     height=350,
#     margin=dict(l=40, r=40, t=60, b=40),
#     paper_bgcolor='#161B22',
#     font=dict(color='#E0E0E0'),
#     plot_bgcolor='#1B2A41'
# )

# st.plotly_chart(fig_heat, use_container_width=True)


# latest_band = station_df.sort_values("timestamp").iloc[-1]['weighted_risk_band']

# if latest_band == "High":
#     st.markdown("🚨 **This station is currently in a critical condition according to the latest readings. Immediate action is recommended.**")
# elif latest_band == "Medium":
#     st.markdown("⚠️ **This station shows moderate risk according to the latest readings. Please monitor or take preventive action based on the trend/summary above.**")
# else:
#     st.markdown("✅ **This station is operating safely according to the latest readings. Check the trends section above for recommented action.**")




# Compare stations
def show_station_comparison(df):
    # st.title("📊 Station Comparison")

    station_ids = st.multiselect("Select stations to compare:", options=sorted(df['station_id'].unique()), default=None)

    if not station_ids or len(station_ids) < 2:
        st.info("Please select at least **two** stations to compare.")
        return

    st.markdown("### 📈 Latest Readings Comparison")

    latest_data = df[df['station_id'].isin(station_ids)].sort_values("timestamp").groupby('station_id').tail(1)

    # Display latest metrics in columns
    for _, row in latest_data.iterrows():
        st.subheader(f"🚰 {row['station_id']} (Risk: {row['weighted_risk_band']})")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Temperature (°C)", f"{row['temperature_c']:.2f}")
        with col2:
            st.metric("Free Chlorine (mg/L)", f"{row['free_chlorine_mgL']:.3f}")
        with col3:
            st.metric("Flow Rate (LPM)", f"{row['flow_rate_lpm']:.2f}")
        with col4:
            st.metric("pH", f"{row['ph']:.2f}")

    # === RADIAL GAUGE PLOTS ===
    st.markdown("### 🎯 Sensor Gauges")
    for _, row in latest_data.iterrows():
        st.markdown(f"#### {row['station_id']} - Risk: **{row['weighted_risk_band']}**")
        col1, col2, col3, col4 = st.columns(4)

        def radial(title, value, vmin, vmax, steps):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title, 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [vmin, vmax]},
                    'bar': {'color': "black"},
                    'steps': steps
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col1:
            radial("Temperature (°C)", row['temperature_c'], 0, 70, [
                {'range': [0, 20], 'color': "lightblue"},
                {'range': [20, 45], 'color': "orange"},
                {'range': [45, 70], 'color': "green"}
            ])
        with col2:
            radial("Free Chlorine", row['free_chlorine_mgL'], 0, 1, [
                {'range': [0, 0.2], 'color': "red"},
                {'range': [0.2, 1], 'color': "green"}
            ])
        with col3:
            radial("Flow Rate", row['flow_rate_lpm'], 0, 5, [
                {'range': [0, 0.01], 'color': "red"},
                {'range': [0.01, 5], 'color': "green"}
            ])
        with col4:
            radial("pH", row['ph'], 0, 14, [
                {'range': [0, 6.5], 'color': "orange"},
                {'range': [6.5, 7.5], 'color': "green"},
                {'range': [7.5, 14], 'color': "orange"}
            ])

    # === Temperature Comparison ===
    st.markdown("### 📈 Temperature Trends Over Time")

    plt.style.use('dark_background')
    fig_mpl, ax = plt.subplots(figsize=(12, 5))
    fig_mpl.patch.set_facecolor('#161B22')
    ax.set_facecolor('#1B2A41')

    # Filter for selected stations
    for sid in station_ids:
        sid_df = df[df['station_id'] == sid].sort_values('timestamp')
        ax.plot(sid_df['timestamp'], sid_df['temperature_c'], label=sid, linewidth=2)

    ax.set_title("Temperature Trends Over Time", color='#E0E0E0')
    ax.set_xlabel("Timestamp", color='#E0E0E0')
    ax.set_ylabel("Temperature (°C)", color='#E0E0E0')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Station")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig_mpl)


st.markdown('<a name="compare"></a>', unsafe_allow_html=True)
st.markdown("<h2 id='compare'>🔀 Stations Comparison</h2>", unsafe_allow_html=True)
show_station_comparison(df)



st.markdown("<h2 id='download-report'>📥 Download Report</h2>", unsafe_allow_html=True)

st.markdown("""
To save the full report, including sensor data, trends, risk summary and guidance:

- Press **Ctrl+P / Cmd+P** or
- Click the button below
""")

# Inject JavaScript-based download button (outside the sidebar)
st.markdown("""
<style>
@media print {
    .no-print {
        display: none;
    }
}
.download-btn {
    background-color: #00B2A9;
    color: white;
    padding: 0.6em 1.2em;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
}
.download-btn:hover {
    background-color: #009195;
}
</style>

<button class="download-btn" onclick="window.print()">📥 Download Report (PDF)</button>
""", unsafe_allow_html=True)



# # === OPTIONAL SHAP EXPLANATION ===
# if model:
#     st.subheader("🧠 Explain Model Prediction (SHAP)")
#     shap.initjs()

#     # Selecting the latest row for explanation
#     sample = station_df[["temperature_c", "ph", "free_chlorine_mgL", "flow_rate_lpm"]].iloc[[-1]]

#     # Predicting
#     prediction = model.predict(sample)[0]
#     st.write("**Predicted Risk Band**:", prediction)

#     # Explaining with SHAP
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(sample)

#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     shap.force_plot(
#         explainer.expected_value[prediction],
#         shap_values[prediction],
#         sample,
#         matplotlib=True,
#         show=True
#     )
#     st.pyplot(bbox_inches='tight')
