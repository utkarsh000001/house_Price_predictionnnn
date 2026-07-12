import os
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
