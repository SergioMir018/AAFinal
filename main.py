from src import preprocess 
from src import knn

def main():
    print("=== Shuttle Landing Control Project ===")
    
    data = preprocess.load_data()

    if data is not None:
        print(f"Dataset cargado correctamente con {len(data)} filas:")
        print(data.head())

        data_scaled = preprocess.preprocess_data(data)

        test_point = [1, 0, 1, 1, 2, 1]
        k = 5
        print(f"=== Data after KNN, with K={k}  ===")
        predicted_class = knn.knn(data_scaled.values.tolist(), test_point, k)
        print(f"Clase predicha para el punto {test_point}: {predicted_class}")
    else:
        print("Por favor, verifica la ruta del archivo o si el archivo existe.")

if __name__ == "__main__":
    main()
