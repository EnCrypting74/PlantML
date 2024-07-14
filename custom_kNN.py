from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy import stats

# Utilizzeremo ClassifierMixin e BaseEstimator per rendere 
# il nostro K-NN compatibile con scikit-learn 
class Custom_kNN(ClassifierMixin, BaseEstimator):
    
    def __init__(self, k = 16, distance = 'Manhattan'):
        self.k = k
        self.distance = distance
    
    def fit(self, train_X, train_y):
        #controlliamo che gli array siano di dimensioni compatibili
        if train_X.shape[0] != train_y.shape[0]:
            raise ValueError("Dimensioni errate")
        self.X = train_X
        self.y = train_y
    
    def CalculateChebyshev(self, row):
        # Calcoliamo la distanza di chebyshev (non supportata da sklearn.metrics.pairwise)
        distances = []
        for i in range(len(self.X)):
            row_train = self.X.iloc[i, :]
            distances.append(np.abs(row_train - row).max())
        return distances
    
    def NearNeighbors(self, distances):
        label = []
        # Cerchiamo i k Neighbours più vicini e prendiamo le loro label
        for i in range(0, self.k):
            dist_min = np.argmin(distances)
            label.append(self.y[dist_min])
            distances[dist_min] = float('inf')
        # Determiniamo l'etichetta da predire in base a quella più frequente tra i k Neighbours
        label_classe = stats.mode(label).mode
        return label_classe

    def predict(self, test_x):
        # Predizione delle etichette di classe per le istanze di test
        predict = []
        for i in range(len(test_x)):
            row = test_x.iloc[i, :]
            # Calcolo delle distanze in base al tipo specificato
            if self.distance == "Euclidean":
                distances = euclidean_distances(self.X, [row])
            elif self.distance == "Manhattan":
                distances = manhattan_distances(self.X, [row])
            elif self.distance == "Chebyshev":
                distances = self.CalculateChebyshev(row)
            else:
                raise TypeError("Tipo di distanza non supportato")
            # Determinazione delle predizioni in base al valore di k
            if self.k > 1:
                predict.append(self.NearNeighbors(distances))
            elif self.k == 1:
                dist_min = np.argmin(distances)
                predict.append(self.y.iloc[dist_min])
            else:
                raise ValueError("Valore di k errato")
        return predict
    
    def fit_predict(self, train_x, train_y, test_x):
        self.fit(train_x,train_y)
        return self.predict(test_x)