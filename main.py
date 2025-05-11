import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Title and Description
# ----------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("📊 Student Performance Predictor")
st.write(
    """
    This app predicts a student's **Final Grade** based on their **Study Hours** and **Attendance** percentage using a simple linear regression model.
    """
)

# ----------------------------
# Sample Data
# ----------------------------
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 92, 95, 98],
    'FinalGrade': [50, 55, 60, 63, 67, 72, 78, 85, 88, 94]
}
df = pd.DataFrame(data)

# ----------------------------
# Model Training
# ----------------------------
X = df[['StudyHours', 'Attendance']]
y = df['FinalGrade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# User Input
# ----------------------------
st.sidebar.header("Enter Student Info")
study_hours = st.sidebar.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)

# ----------------------------
# Prediction
# ----------------------------
def predict_grade(study_hours, attendance):
    prediction = model.predict([[study_hours, attendance]])
    return prediction[0]

if st.sidebar.button("Predict Grade"):
    result = predict_grade(study_hours, attendance)
    st.success(f"🎯 Predicted Final Grade: **{result:.2f}**")

# ----------------------------
# Data and Visualization
# ----------------------------
st.subheader("📋 Sample Dataset")
st.dataframe(df)

st.subheader("📉 Data Visualization")
fig, ax = plt.subplots()
ax.scatter(df['StudyHours'], df['FinalGrade'], color='blue', label='Study Hours vs Final Grade')
ax.scatter(df['Attendance'], df['FinalGrade'], color='green', label='Attendance vs Final Grade')
ax.set_xlabel('Study Hours / Attendance')
ax.set_ylabel('Final Grade')
ax.set_title('Effect of Study Hours & Attendance on Final Grade')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Linear Regression Model Example")
