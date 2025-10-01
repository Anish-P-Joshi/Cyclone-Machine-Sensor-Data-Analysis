# Cyclone-Machine-Sensor-Data-Analysis

This repository contains the full implementation of a data science case study focused on **industrial sensor data analysis**.  
The objective was to explore, detect, and interpret patterns in time-series process data, including system shutdowns, operating states, anomalies, and forecasting.

---

## Project Overview
The assignment simulates a real-world problem of monitoring an industrial process with multiple sensor inputs. The main goals were:

1. **Exploratory Analysis**  
   - Preprocessing raw sensor data  
   - Visualizing representative time slices (weekly, yearly) to understand variance and trends  

2. **Shutdown Detection**  
   - Identifying and marking system shutdown periods using threshold-based rules  
   - Visualizing shutdowns across a full year  

3. **State/Cluster Identification**  
   - Clustering normal operations into distinct regimes using KMeans  
   - Summarizing each state with descriptive statistics (means, standard deviations, percentiles)  
   - Computing frequency and duration of each state  

4. **Contextual Anomaly Detection**  
   - Detecting anomalies within states using Isolation Forest  
   - Highlighting anomalies in time series context, along with implicated variables  

5. **Forecasting**  
   - Applying ARIMA and Prophet models for key sensor variables  
   - Comparing performance and highlighting forecasting challenges (e.g., regime shifts, shutdowns, non-stationarity)  

6. **Insights & Recommendations**  
   - Deriving exploratory, predictive, and prescriptive insights  
   - Connecting anomalies, shutdowns, and state behavior to actionable monitoring strategies  
   - Recommending alert rules and future data collection needs  

---

## Repository Structure
├── task1_analysis.ipynb # Main notebook with step-by-step analysis
├── data.csv # Input dataset (or sample dataset)
├── Task1/
│ ├── state_summary.csv # Summary of states/clusters
│ ├── anomalous_periods.csv # Detected anomalies
│ └── plots/ # Visualization outputs
└── README.md # Project documentation


---

## Tools & Libraries
- Python 3.10+  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- scikit-learn (KMeans, Isolation Forest)  
- statsmodels (ARIMA)  
- Prophet (forecasting)  

---

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/Anish-P-Joshi/Cyclone-Machine-Sensor-Data-Analysis.git
   cd Cyclone-Machine-Sensor-Data-Analysis
Key Insights

Shutdown periods are strongly associated with specific low sensor thresholds.

Certain operating states persist longer and exhibit higher anomaly rates.

Many anomalies precede shutdowns by 10–30 minutes, suggesting early-warning potential.

Forecasting is challenged by non-stationarity and operating regime shifts, but short-term predictions remain useful for monitoring.

Author
Prepared by Anish Joshi
