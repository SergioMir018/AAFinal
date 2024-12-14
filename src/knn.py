from collections import Counter

def euclidian_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def get_k_nearest_neighbors(data_scaled, test_point, k):
    distances = []

    for train_point in data_scaled:
        distance = euclidian_distance(train_point[:-1], test_point)
        distances.append((distance, train_point[-1]))

    distances.sort(key = lambda x: x[0])

    return distances[:k]

def predict_class(neighbors):
    classes = [neighbor[1] for neighbor in neighbors]
    return Counter(classes).most_common(1)[0][0]

def knn(data_scaled, test_point, k):
    neighbors = get_k_nearest_neighbors(data_scaled, test_point, k)
    predicted_class = predict_class(neighbors)

    return predicted_class
    
