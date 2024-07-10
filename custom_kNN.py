from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy import stats

#utilizzeremo ClassifierMixin e BaseEstimator per rendere 
# il nostro K-NN compatibile con scikit-learn 
class Custom_kNN(ClassifierMixin, BaseEstimator):
    def __init__(self, k = 4, distance = 'Manhattan'):
        self.k = k
        self.distance = distance
    
    def fit(self, X, y):
        #controlliamo che gli array siano di dimensioni compatibili
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensioni errate")
        self.X = X
        self.y = y
    
    def CalculateChebyshev(self, row):
        # Calcoliamo la distanza di chebyshev (non supportata da sklearn.metrics.pairwise)
        distances = []
        for i in range(len(self.X)):
            row_train = self.X.iloc[i, :]
            distances.append(np.abs(row_train - row).max())
        return distances
    
    def NearNeighbors(self, distances):
        label = []
        # Cerchiamo i k Neighbors più vicini e prendiamo le loro label
        for i in range(0, self.k):
            dist_min = np.argmin(distances)
            label.append(self.y[dist_min])
            distances[dist_min] = float('inf')
        # Determiniamo l'etichetta da predire in base a quella più frequente tra i k Neighbors
        label_classe = stats.mode(label).mode
        return label_classe

    def predict(self, test_x):
        # Predizione delle etichette di classe per le istanze di test
        predict = []
        for i in range(len(test_x)):
            row = test_x.iloc[i, :]
            # Calcolo delle distanze in base al tipo specificato
            if self.distance == "Euclidea":
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
