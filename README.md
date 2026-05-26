# 💧 Aqua-AI: Legionella Risk Monitoring Portal

Aqua-AI is a dashboard I built to make Legionella risk monitoring something anyone can actually use, not just engineers or consultants. It takes live sensor data, scores the risk, and tells you in plain English whether a station needs attention. No spreadsheets, no guesswork.

---

## What it does

- 📊 **Overview** - See how many stations you have, how many are flagged, and when the last reading came in.
- 📍 **Station Explorer** - Pick a station and dig into its history across temperature, chlorine, pH, and flow rate.
- 📈 **Risk Score Over Time** - One line, one story. Green zone means safe, orange means watch it, red means act now.
- 🎬 **Animated Sensor Trends** - Hit play and watch how readings have changed over time. If the line drifts into the red shaded area, that sensor is outside safe limits.
- 🫧 **Sensor Correlation Explorer** - Temperature and chlorine plotted together. Bubbles in the red box mean both are in a bad place at the same time, which is when risk is highest.
- 🎯 **Radial Gauges** - At-a-glance gauges showing the latest readings against safe thresholds.
- 🧠 **GPT-4 Risk Q&A** - Ask a question about your station in plain English and get an answer grounded in official guidance from the HSE, CDC, and GOV.UK. Not generic AI waffle, actual regulatory references.
- 🔀 **Compare Stations** - Put two or more stations side by side across every metric.
- 📥 **Download Reports** - Export everything as a PDF.

---

## Built with

Python, Streamlit, Plotly, LangChain, OpenAI GPT-4, ChromaDB, Pandas, Scikit-learn

