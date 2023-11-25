import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sample data and labels
data = np.array([
    [1.70, 65, 20],
    [1.90, 85, 33],
    [1.78, 76, 31],
    [1.73, 74, 24],
    [1.81, 75, 35],
    [1.73, 70, 75],
    [1.75, 69, 25]
])
data_class = np.array(['programmer', 'builder', 'builder', 'programmer', 'builder', 'scientist', 'programmer'])

# Input data point for prediction
input_point = np.array([1.69, 79, 37])

# Number of neighbors to consider
k = 5

def knn_predict(input_point, data, data_class, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, data_class)
    predicted_class = knn.predict([input_point])
    return predicted_class[0]

predicted_class = knn_predict(input_point, data, data_class, k)
print(f"The predicted class for the data point {input_point} is: {predicted_class}")
