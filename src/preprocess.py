import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "shuttle-landing-control.data")

def load_data():
    try:
        column_names = [
            "Class", "Stability", "Error", "Sign", 
            "Wind", "Magnitude", "Visibility"
        ]

        data = pd.read_csv(DATA_PATH, names=column_names, delimiter=",", skipinitialspace=True)
        
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {DATA_PATH},")
        return None

def preprocess_data(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
