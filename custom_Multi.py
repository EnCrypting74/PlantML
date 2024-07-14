from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from DataSetSplitter import DS_Splitter
import numpy as np
import random
from scipy import stats

class CustomRandomForest():

    def __init__ (self, num_trees = 95, depth = 35, random_seed = 0):
        self.num_trees = num_trees # Numero di alberi 
        self.tree_array = []       # Array degli alberi creati
        self.column_array = []     # Array delle colonne degli alberi
        self.depth = depth         # Profondità alberi
        self.random_seed = random_seed

    def fit(self, x, y):
        for i in range(0, self.num_trees):

            random.seed(i + self.random_seed)
            np.random.seed(i + self.random_seed)

            # Numero di colonne da estrarre
            col_num = random.randint(1, len(x.columns))

            # Estrarre un numero casuale di colonne dal DataFrame
            columns = np.random.choice(x.columns, size = col_num, replace=False)

            # Salvo le colonne selezionate per fare poi il test
            self.column_array.append(columns)

            # Creare un nuovo DataFrame con le colonne estratte
            extracted = x[columns]

            # Esegui lo split del dataset per selezionare record random dal nuovo dataframe campione
            train_x, _, train_y, _ = train_test_split(extracted, y, test_size=0.35, random_state=i + self.random_seed)

            # Istanza di un albero di classificazione con una profondità di default
            dTree_clf = DecisionTreeClassifier(max_depth = self.depth, random_state = i + self.random_seed)

            # Salvo l'albero generato
            self.tree_array.append(dTree_clf)

            # Addestramento
            dTree_clf.fit(train_x, train_y)

    def predict(self, x):
        labels = []

        for i in range(len(x)):
            #seleziono la riga del test set
            row = x.iloc[i, :]

            #calcola etichetta
            labels.append(self.hard_voting(row))

        return labels
    
    def hard_voting(self, record):
        # Hard voting per ottenere la label di classe più comune tra gli alberi
        labels = []

        for i in range(len(self.tree_array)):
            current_col = self.column_array[i]
            row_selected = record[current_col].to_frame().transpose()
            labels.append(self.tree_array[i].predict(row_selected)[0])

        # Trova l'elemento con l'occorrenza massima
        label_classe = stats.mode(labels).mode
        return label_classe
    
    def fit_predict(self, train_x, train_y, test_x):
        # Addestramento del modello e predizione sul test set
        self.fit(train_x, train_y)
        return self.predict(test_x)
