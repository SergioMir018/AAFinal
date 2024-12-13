import os
import pandas as pd

DATA_PATH = os.path.join("data", "shuttle-landing-control.data")

def load_data(file_path):
    try:
        
        column_names = [
            "Class", "Stability", "Error", "Sign", 
            "Wind", "Magnitude", "Visibility"
        ]

        data = pd.read_csv(file_path, names=column_names, delimiter=",", skipinitialspace=True)
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
        return None

def main():
    print("=== Shuttle Landing Control Project ===")
    
    data = load_data(DATA_PATH)
    if data is not None:
        print(f"Dataset cargado correctamente con {len(data)} filas:")
        print(data.head())
    else:
        print("Por favor, verifica la ruta del archivo o si el archivo existe.")

if __name__ == "__main__":
    main()
