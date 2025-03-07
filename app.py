import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Apekshaj04/LoanEligibility-/refs/heads/main/loan-train.csv")
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df = df.drop(columns=['Loan_ID'])

    if 'Loan_Status' not in df.columns:
        raise KeyError("Target column 'Loan_Status' not found in the DataFrame.")

    df = pd.get_dummies(df, drop_first=True)
    
    
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    print(df.columns)
    df = df.drop(columns = ['Education_Not Graduate'])

    print(df.columns)
    X = df.drop(columns=['Loan_Status_Y'])
    y = df['Loan_Status_Y']
    X = X.fillna(X.mean())  
    return X, y, df


if 'df' not in st.session_state:
    X, y, df = load_data()
    st.session_state.df = df
    st.session_state.X = X
    st.session_state.y = y

def user_input():
    st.subheader("Loan Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=2895)
    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    with col3:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=95.0)
    with col4:
        loan_amount_term = st.number_input("Loan Term (months)", min_value=0, value=360)

    # Radio Buttons grouped in Rows of 3
    st.subheader("Personal Details")
    col5, col6, col7 = st.columns(3)
    with col5:
        credit_history = st.radio(
            "Credit History", 
            options=[1.0, 0.0], 
            index=0, 
            format_func=lambda x: "Good" if x == 1.0 else "Bad"
        )
        gender = st.radio(
            "Gender", 
            options=[1, 0], 
            index=0, 
            format_func=lambda x: "Male" if x == 1 else "Female"
        )
        marital_status = st.radio(
            "Marital Status", 
            options=[1, 0], 
            index=0, 
            format_func=lambda x: "Married" if x == 1 else "Single"
        )
    with col6:
        dependents = st.radio(
            "Dependents", 
            options=[0, 1, 2, '3+'], 
            index=0
        )
        education = st.radio(
            "Education", 
            options=[1, 0], 
            index=0, 
            format_func=lambda x: "Graduate" if x == 1 else "Non-Graduate"
        )
        self_employed = st.radio(
            "Self Employed", 
            options=[1, 0], 
            index=0, 
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    with col7:
        property_area = st.radio(
            "Property Area", 
            options=["Urban", "Semiurban", "Rural"], 
            index=0
        )
    return {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Gender_Male': gender,
        'Married_Yes': marital_status,
        'Dependents': dependents,
        'Education_Graduate': education,
        'Self_Employed_Yes': self_employed,
        'Property_Area_Rural': 1 if property_area == "Rural" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }


def decision_tree_logic(user_inputs):
    dependents_dict = {0: 'Dependents_0', 1: 'Dependents_1', 2: 'Dependents_2', '3+': 'Dependents_3+'}
    dependents_values = {key: False for key in dependents_dict.values()}
    selected_dependent = dependents_dict[user_inputs['Dependents']]
    dependents_values[selected_dependent] = True

    new_data = pd.DataFrame({
        'ApplicantIncome': [user_inputs['ApplicantIncome']],
        'CoapplicantIncome': [user_inputs['CoapplicantIncome']],
        'LoanAmount': [user_inputs['LoanAmount']],
        'Loan_Amount_Term': [user_inputs['Loan_Amount_Term']],
        'Credit_History': [user_inputs['Credit_History']],
        'Gender_Male': [user_inputs['Gender_Male']],
        'Married_Yes': [user_inputs['Married_Yes']],
        'Dependents_0': [dependents_values['Dependents_0']],
        'Dependents_1': [dependents_values['Dependents_1']],
        'Dependents_2': [dependents_values['Dependents_2']],
        'Dependents_3+': [dependents_values['Dependents_3+']],
        'Education_Graduate': [user_inputs['Education_Graduate']],
        'Self_Employed_Yes': [user_inputs['Self_Employed_Yes']],
        'Property_Area_Rural': [user_inputs['Property_Area_Rural']],
        'Property_Area_Semiurban': [user_inputs['Property_Area_Semiurban']],
        'Property_Area_Urban': [user_inputs['Property_Area_Urban']]
    })

    dt = DecisionTreeClassifier()
    dt.fit(st.session_state.X, st.session_state.y)
    
    new_data = new_data[st.session_state.X.columns]
    prediction = dt.predict(new_data)

    return prediction[0]

def main():
    st.title("Loan Status Prediction App")

    user_inputs = user_input()

    if st.button("Submit"):
        prediction = decision_tree_logic(user_inputs)
        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

if __name__ == "__main__":
    main()
