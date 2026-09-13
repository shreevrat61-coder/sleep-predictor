import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Screen Time & Sleep Predictor", layout="wide")
st.title("📱 Daily Screen Time & Sleep Quality Predictor")
st.write("An AI model analyzing how daily digital habits affect sleep quality.")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 150
    screen_time = np.random.uniform(2.0, 9.0, n)
    late_night_use = screen_time * np.random.uniform(0.2, 0.6, n)
    caffeine = np.random.randint(0, 4, n)
    sleep_score = 10 - (screen_time * 0.4) - (late_night_use * 0.8) - (caffeine * 0.5) + np.random.normal(0, 0.4, n)
    sleep_score = np.clip(sleep_score, 1, 10)
    
    return pd.DataFrame({
        'Total_Screen_Time': np.round(screen_time, 1),
        'Late_Night_Use_Hrs': np.round(late_night_use, 1),
        'Caffeine_Cups': caffeine,
        'Sleep_Quality_Score': np.round(sleep_score, 1)
    })

df = load_data()

X = df[['Total_Screen_Time', 'Late_Night_Use_Hrs', 'Caffeine_Cups']]
y = df['Sleep_Quality_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.sidebar.header("Enter Your Daily Log")
input_st = st.sidebar.slider("Total Screen Time (Hours)", 1.0, 12.0, 5.0)
input_ln = st.sidebar.slider("Screen Time After 10 PM (Hours)", 0.0, 5.0, 1.5)
input_caf = st.sidebar.slider("Caffeine Cups Consumed", 0, 5, 1)

user_inputs = pd.DataFrame({
    'Total_Screen_Time': [input_st],
    'Late_Night_Use_Hrs': [input_ln],
    'Caffeine_Cups': [input_caf]
})

predicted_score = model.predict(user_inputs)[0]
predicted_score = max(1.0, min(10.0, predicted_score))

st.subheader("📊 Your Sleep Prediction")
st.metric(label="Predicted Sleep Quality Score (out of 10)", value=f"{predicted_score:.1f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.regplot(x='Late_Night_Use_Hrs', y='Sleep_Quality_Score', data=df, ax=ax, color='purple')
st.pyplot(fig)
