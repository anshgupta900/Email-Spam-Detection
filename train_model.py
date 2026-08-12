import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("email_spam_detection.csv")

df["Sender_Email"] = df["Sender_Email"].replace(
    r"^\s*$",
    np.nan,
    regex=True
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

outlier_cols = [
    "Num_Links",
    "Num_Special_Chars",
    "Capital_Words"
]

for col in outlier_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(
        df[col] > upper_bound,
        upper_bound,
        np.where(
            df[col] < lower_bound,
            lower_bound,
            df[col]
        )
    )


df = pd.get_dummies(
    df,
    columns=[
        "Subject",
        "Sender_Domain"
    ],
    drop_first=True,
    dtype=int
)


X = df.drop(
    ["Spam", "Sender_Email"],
    axis=1
)

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

X_train[numerical_cols] = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test[numerical_cols] = scaler.transform(
    X_test[numerical_cols]
)


log = LogisticRegression(
    max_iter=1000
)

log.fit(
    X_train,
    y_train
)

log_pred = log.predict(X_test)

log_accuracy = accuracy_score(
    y_test,
    log_pred
)


dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)


rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)


print("Logistic Regression:", log_accuracy)
print("Decision Tree:", dt_accuracy)
print("Random Forest:", rf_accuracy)


models = {
    "Logistic Regression": (
        log,
        log_accuracy
    ),
    "Decision Tree": (
        dt,
        dt_accuracy
    ),
    "Random Forest": (
        rf,
        rf_accuracy
    )
}


best_model_name = max(
    models,
    key=lambda name: models[name][1]
)

best_model = models[
    best_model_name
][0]


print("Best Model:", best_model_name)


joblib.dump(
    best_model,
    "best_spam_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    list(X.columns),
    "model_features.pkl"
)


def predict_spam(
    sender_email,
    subject,
    email_length,
    num_links,
    num_special_chars,
    capital_words,
    has_attachment
):

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
        columns=[
            "Subject",
            "Sender_Domain"
        ],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )

    input_data[numerical_cols] = scaler.transform(
        input_data[numerical_cols]
    )

    prediction = best_model.predict(
        input_data
    )[0]

    return prediction