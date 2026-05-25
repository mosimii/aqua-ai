# 💧 Aqua-AI: Legionella Risk Monitoring Portal

Aqua-AI is a real-time AI-powered dashboard that helps facilities monitor conditions contributing to Legionella risk. Built with Streamlit, it integrates advanced analytics, visualizations, and LLM-based explanations to support proactive risk management.

---

## 🚀 Features

- 📊 **Sensor Data Overview** - Displays total stations, latest readings, and high-risk counts.
- 📍 **Station Explorer** - View historical data per station including temperature, chlorine, pH, and flow.
- 📈 **Trends & Risk Bands** - Time-series plots with risk color bands.
- 🎯 **Radial Gauges** – Interactive gauges for latest readings.
- 🧠 **Questions and Risk Explanation (RAG)** – GPT-4 explains site risk using contextual data + expert documentation with citations.
- 🔀 **Stations Comparison** – Compare two or more stations across key metrics and trends.
- 📥 **Download Reports** – Export dashboard visuals as PDF.
- 🌐 **Sidebar Navigation** – Quick access to all sections with links to Aquatrust resources and training.

---

## 🧠 Powered By

- **Streamlit** – Rapid UI for data apps
- **Plotly** – Interactive charts and gauges
- **LangChain, OpenAIEmbeddings + OpenAI GPT-4** – Risk insight via Retrieval-Augmented Generation (RAG)
- **ChromaDB** – Vector store for expert guidance documents, from Aquatrust, HSE, WHO
- **Pandas & Joblib** – Data processing and ML model loading
