import streamlit as st
from train_model import predict_spam


st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detector")
st.write("Enter the email details below to check whether it is spam.")

st.divider()

sender_email = st.text_input(
    "Sender Email",
    placeholder="example@gmail.com"
)

subject = st.text_input(
    "Subject",
    placeholder="Enter email subject"
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

st.divider()

if st.button(
    "🔍 Predict Spam",
    type="primary",
    use_container_width=True
):

    if sender_email.strip() == "":
        st.warning("Please enter sender email.")

    elif subject.strip() == "":
        st.warning("Please enter email subject.")

    else:

        prediction = predict_spam(
            sender_email,
            subject,
            email_length,
            num_links,
            num_special_chars,
            capital_words,
            has_attachment
        )

        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")