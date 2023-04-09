import pickle
import streamlit as st
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
# App Header
st.header('Loan Approvel Prediction  ')
model = pickle.load(open('Model\knn.pkl', 'rb'))
standarscalr = pickle.load(open('Model\Sc.pkl', 'rb'))

Gender = st.selectbox(
    "Enter Your Gender => Male : 2 | Female : 1 | Other : 0", (0, 1, 2))
Married = st.selectbox(
    "Enter Your Married Status => Yes : 1 | No : 0 ", (0, 1))
Dependents = st.selectbox(
    "Enter Your Dependents Status => 0 | 1 | 2 | 3", (0, 1, 2, 3))
Education = st.selectbox(
    "Enter Your Education Name => 'Graduate' : 1, 'Not Graduate' : 0", (0, 1))
Self_Employed = st.selectbox(
    "Enter Your Self_Employed Status => Yes : 1 | No : 0 ", (0, 1))

ApplicantIncome = st.number_input('Enter a ApplicantIncome')

CoapplicantIncome = st.number_input('Enter a CoapplicantIncome')

LoanAmount = st.number_input('Enter a LoanAmount')

Loan_Amount_Term = st.selectbox(
    "Enter Your Loan_Amount_Term Status?", (480, 360, 300, 240, 180, 120, 60, 36, 84, 12))

Credit_History = st.selectbox(
    "Enter Your Credit_History Status => Yes : 1 | No : 0 ", (0, 1))
Property_Area = st.selectbox(
    "Enter Your Married Status\n => Urban : 1 | Rural: 2 | Semiurban :3 ", (1, 2, 3))

x_test = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                      CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])
x_test_sc = standarscalr.transform(x_test)

result = model.predict(x_test_sc)[0]
if st.button('Predict'):
    if result == 1:
        st.write("Approved")
    else:
        st.write("Not Approved")
else:
    st.write(".......")
