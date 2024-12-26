import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Bank stability Insights ",
    layout="wide",
    page_icon="🏦",
)


@st.cache_data
def load_data(uploaded_file):
    """Load the uploaded CSV file into a DataFrame."""
    return pd.read_csv(uploaded_file)


def preprocess_data(data, target_column):
    """Preprocess the data for modeling."""
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X, y, label_encoder


@st.cache_data
def train_model(X, y):
    """Train an XGBoost model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)
    return model, X_train.columns


def main():
    """Main function for the Streamlit app."""
    st.title("🏦 Enhanced Bank Stability Predictor")
    st.markdown(
        """
        Welcome to the next-generation Bank Stability Predictor.  
        Upload your dataset, explore key insights, and predict stability using advanced machine learning models.
        """
    )
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio(
        "Choose an Option:",
        ["🏠 Home", "📂 Upload & Explore", "📈 Predict Stability", "ℹ️ About"]
    )

    if menu == "🏠 Home":
        st.header("🏠 Welcome to the Predictor")
        st.image(r"D:\Pictures\Bank.jpg", caption="Analyzing Financial Health")
        st.markdown(
            """
            - **Objective**: Evaluate financial indicators and predict bank stability.  
            - **Features**: Upload your dataset, explore its structure, and make predictions dynamically.
            """
        )
        st.markdown("---")
        st.info("Navigate through the app using the menu on the left!")

    elif menu == "📂 Upload & Explore":
        st.header("📂 Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.success("Dataset uploaded successfully!")
            st.markdown("### Dataset Overview")
            st.write(data.head())

            st.markdown("### Data Summary")
            st.write(data.describe())

            st.markdown("### Missing Values")
            st.write(data.isnull().sum())

            st.markdown("### Correlation Heatmap (Numeric Data Only)")
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=["number"])

            if not numeric_data.empty:
                # Generate the heatmap only if numeric data exists
                plt.figure(figsize=(10, 6))
                sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
                st.pyplot(plt)
            else:
                st.warning("No numeric columns available for correlation heatmap.")

    elif menu == "📈 Predict Stability":
        st.header("📈 Make Predictions")
        uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.success("Dataset uploaded successfully!")

            # Choose Target Column
            target_column = st.selectbox("Select Target Column:", options=data.columns)

            if target_column:
                X, y, label_encoder = preprocess_data(data, target_column)
                model, feature_columns = train_model(X, y)

                st.markdown("### Enter Bank Details")
                input_data = {}

                with st.form("prediction_form"):
                    for feature in feature_columns:
                        input_data[feature] = st.text_input(f"{feature}:", value="0")

                    submitted = st.form_submit_button("Predict")

                    if submitted:
                        try:
                            input_df = pd.DataFrame([input_data], columns=feature_columns)
                            prediction = model.predict(input_df.apply(pd.to_numeric, errors="coerce").fillna(0))
                            if label_encoder:
                                prediction = label_encoder.inverse_transform(prediction)
                            st.success(f"The bank is predicted to be: **{prediction[0]}**")
                        except Exception as e:
                            st.error(f"Error: {e}")

    elif menu == "ℹ️ About":
        st.header("ℹ️ About the App")
        st.markdown(
            """
            - **Purpose**: Predict bank stability dynamically using financial metrics.  
            - **Technologies**: Python, Streamlit, Scikit-learn, XGBoost.  
            - **Developer**: Priyanka Mohapatra!  
            """
        )
        st.markdown("---")
        st.info("Explore, Predict, and Make Data-Driven Decisions.")


if __name__ == "__main__":
    main()
