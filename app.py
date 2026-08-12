import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧"
)

st.title("📧 Email Spam Detector")
st.write("Enter the email details below.")

model = joblib.load("best_spam_model.pkl")

df = pd.read_csv("email_spam_detection.csv")

df["Sender_Email"] = df["Sender_Email"].replace(
    r"^\s*$", np.nan, regex=True
)

df["Sender_Email"] = df["Sender_Email"].fillna(
    "unknown@example.com"
)

df["Email_Length"] = df["Email_Length"].fillna(
    df["Email_Length"].median()
)

df["Sender_Domain"] = (
    df["Sender_Email"]
    .astype(str)
    .str.split("@")
    .str[-1]
    .str.lower()
)
for col in ["Num_Links", "Num_Special_Chars", "Capital_Words"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(
        df[col] > upper,
        upper,
        np.where(df[col] < lower, lower, df[col])
    )

df = pd.get_dummies(
    df,
    columns=["Subject", "Sender_Domain"],
    drop_first=True,
    dtype=int
)

X = df.drop(["Spam", "Sender_Email"], axis=1)
y = df["Spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numerical_cols = [
    "Email_Length",
    "Num_Links",
    "Num_Special_Chars",
    "Capital_Words"
]

scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])

sender_email = st.text_input(
    "Sender Email",
    placeholder="example@gmail.com"
)

subject = st.text_input(
    "Subject",
    placeholder="Enter subject"
)

email_length = st.number_input(
    "Email Length",
    min_value=0,
    value=100
)

num_links = st.number_input(
    "Number of Links",
    min_value=0,
    value=0
)

num_special_chars = st.number_input(
    "Number of Special Characters",
    min_value=0,
    value=0
)

capital_words = st.number_input(
    "Capital Words",
    min_value=0,
    value=0
)

has_attachment = st.selectbox(
    "Has Attachment",
    ["No", "Yes"]
)
if st.button("Predict Spam", type="primary"):

    if sender_email.strip() == "":
        st.warning("Please enter sender email.")

    elif subject.strip() == "":
        st.warning("Please enter subject.")

    else:

        if "@" in sender_email:
            domain = sender_email.split("@")[-1].lower()
        else:
            domain = "unknown"

        input_data = pd.DataFrame([{
            "Email_Length": email_length,
            "Num_Links": num_links,
            "Num_Special_Chars": num_special_chars,
            "Capital_Words": capital_words,
            "Has_Attachment": 1 if has_attachment == "Yes" else 0,
            "Subject": subject,
            "Sender_Domain": domain
        }])

        input_data = pd.get_dummies(
            input_data,
            columns=["Subject", "Sender_Domain"],
            drop_first=True,
            dtype=int
        )

        if "Email_ID" in input_data.columns:
            input_data = input_data.drop(
                "Email_ID",
                axis=1
            )
        model_features = model.feature_names_in_

        input_data = input_data.reindex(
            columns=model_features,
            fill_value=0
        )

        input_data[numerical_cols] = scaler.transform(
            input_data[numerical_cols]
        )
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")