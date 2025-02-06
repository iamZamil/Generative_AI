import streamlit as st 

st.title("Get Input from User")
name=st.text_input("Enter your name :")
if name:
    st.write(f"Hello {name}")
    
age=st.slider("Select your age:",0,50,100)
st.write(f"Your age is {age}")
if name and age:
    st.write(f"Hello {name}" f" Your age is : {age}" )