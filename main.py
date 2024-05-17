import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from adtk.data import validate_series
from adtk.visualization import plot
# remove the wildcard import later 
# with only used the imports 
from adtk.detector import * 

# st title
st.title("Anomaly Detection For Time Series Data in Python using ADTK library")
st.write("Dataset : monthly temperature data in CSV format.")
# Author and GitHub link
st.markdown("**Created By**: Kulraj Chavda,  [GitHub Source Code](https://github.com/kulraj710)")

# st heading
st.markdown("## Select & click on Analyse to detect anamolies")
# st user selection field
options = st.selectbox(
    'Select Technique',
    ('ThresholdAD', 'Quantile', 'InterQuartileRangeAD (IQR)')
)

# load data
data = pd.read_csv('monthly_temperature.csv')
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date")
data = data['Mean']

# Buttom to trigger the analysis
if st.button('Analyse'):
    st.write(f"Anomalies detected using : {options}")
    
    if options == 'ThresholdAD':
    # Use the function 'ThresholdAD' when you want to manually provide
    # high and low but as this is not very intelligent we will use QuantaileAD
        threshold_detector = ThresholdAD(low=-0.5, high=0.75)
        anomalies = threshold_detector.detect(data)
        
    if options == 'Quantile':
        # Quantile Detector
        quantile_detector = QuantileAD(low=0.01, high=0.99)
        anomalies = quantile_detector.fit_detect(data)

    if options == 'InterQuartileRangeAD (IQR)':
        # IQR detector
        iqr_detector = InterQuartileRangeAD(c=1.5)
        anomalies = iqr_detector.fit_detect(data)

    # plot params
    plot(data, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
    
    # st plot (equivalent to saying "plt.plot()" )
    st.pyplot(plt)



# END of Project Logic
# Bullet points describing learnings
st.markdown("### My Learnings from this small project:")
st.markdown("""
    1. **Anomaly Detection Techniques**: Gained insights into different anomaly detection techniques such as ThresholdAD, QuantileAD, and InterQuartileRangeAD (IQR) using the ADTK library.
    2. **Data Preprocessing**: Learned how to preprocess and validate time series data for anomaly detection, including handling datetime formats and setting the index appropriately.
    3. **ADTK Library Usage**: Acquired knowledge on how to use the ADTK library for anomaly detection, including how to fit detectors and interpret the results.
    4. **Parameter Tuning**: Experienced the importance of parameter tuning in anomaly detection models to improve the accuracy and relevance of detected anomalies.
    """)
