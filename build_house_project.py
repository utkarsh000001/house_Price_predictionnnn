import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

project_name = "house_price_project"

# 1. Define the directory structure
directories = [
    os.path.join(project_name, "Dataset"),
    os.path.join(project_name, "Documentation"),
    os.path.join(project_name, "Model"),
    os.path.join(project_name, "Notebook"),
    os.path.join(project_name, "Streamlit_App")
]

print("📁 Creating directories...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# 2. Write non-code files
files = {
    "requirements.txt": "streamlit\npandas\nscikit-learn\nnumpy\n",
    "README.md": "# House Price Prediction\n\nA Streamlit application to predict house prices.\n",
    "Dataset/house_data.csv": "SquareFeet,Bedrooms,Bathrooms,YearBuilt,Neighborhood_Score,Price\n1500,3,2,2010,7,250000\n2000,4,2.5,2015,8,350000\n1200,2,1,1995,5,180000\n2400,4,3,2020,9,450000\n900,1,1,1980,4,120000\n",
    "Documentation/Project_Report.txt": "House Price Prediction Project Metrics.\n"
}

for relative_path, content in files.items():
    with open(os.path.join(project_name, relative_path), "w", encoding="utf-8") as f:
        f.write(content)

# 3. Write train_model.py with bulletproof paths
train_model_code = '''import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'Dataset', 'house_data.csv')
    
    df = pd.read_csv(data_path)
    X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Neighborhood_Score']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    model_path = os.path.join(BASE_DIR, 'Model', 'house_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"🎉 Success! Model saved directly to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
'''
with open(os.path.join(project_name, "Model", "train_model.py"), "w", encoding="utf-8") as f:
    f.write(train_model_code)

# 4. Write app.py with bulletproof paths
app_code = '''import os
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction Tool")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'Model', 'house_model.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file not found! Expected it at: {model_path}")
    st.stop()

st.header("Property Details")
col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Total Square Footage", min_value=300, max_value=10000, value=1500)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
with col2:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010)
    location_score = st.slider("Neighborhood Rating (1-10)", min_value=1.0, max_value=10.0, value=5.0)

if st.button("Calculate Estimated Value", type="primary"):
    input_features = np.array([[sqft, bedrooms, bathrooms, year_built, location_score]])
    predicted_price = model.predict(input_features)[0]
    st.success(f"💰 Estimated Market Value: **${predicted_price:,.2f}**")
'''
with open(os.path.join(project_name, "Streamlit_App", "app.py"), "w", encoding="utf-8") as f:
    f.write(app_code)

# 5. Automatically execute the training script to generate the .pkl file right now
print("🤖 Training the initial model baseline...")
df_sample = pd.read_csv(os.path.join(project_name, "Dataset", "house_data.csv"))
X = df_sample[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Neighborhood_Score']]
y = df_sample['Price']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open(os.path.join(project_name, "Model", "house_model.pkl"), "wb") as file:
    pickle.dump(model, file)

print("\n🚀 All done! The folder 'house_price_project' is ready to compress and deploy.")