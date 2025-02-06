import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb

st.title("Generative AI application")

st.write("This is my first simple text")
dataframe=pd.DataFrame({
    'first coloumn': [23,34,23,12,23,3],
    'second column': [23,34,23,12,23,3]
})

st.write(dataframe)
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)
st.write(chart_data)