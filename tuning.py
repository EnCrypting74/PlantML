import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from DataSetSplitter import DS_Splitter
from custom_kNN import Custom_kNN
from custom_Multi import CustomRandomForest
from metrics import tuningMetrics

class tuning():

    def svmTuning(train_x, train_y):
        
        def gridSearchSVM(train_x, train_y, param_grid):
            svm_clf = SVC()
            n_folds = 5
            grid_search_cv = GridSearchCV(svm_clf, param_grid, cv=n_folds)
            grid_search_cv.fit(train_x, train_y)
            #print('Accuratezza media per combinazione:\n', grid_search_cv.cv_results_['mean_test_score'])
            #print('Combinazione migliore:\n', grid_search_cv.best_params_)
            #print('Accuratezza media della combinazione migliore: %.3f' % grid_search_cv.best_score_)
            #print()
        
        param_grid_linear = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]
        param_grid_rbf = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.025, 0.05]}]
        
        scaler = StandardScaler()
        scld_train_x = scaler.fit_transform(train_x)
        #print('SVM lineare')
        gridSearchSVM(scld_train_x, train_y, param_grid_linear)
        #print('SVM con kernel RBF')
        gridSearchSVM(scld_train_x, train_y, param_grid_rbf)

    
    def decisionTreeTuning(train_x, test_x, train_y):
        
        # Definizione della griglia di iperparametri
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Ricerca degli iperparametri ottimali
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, refit=True, verbose=2, cv=5)
        grid_search.fit(train_x, train_y)

        # Stampa dei migliori parametri
        print("Best parameters found: ", grid_search.best_params_)

        # Predizione e valutazione del modello
        y_pred = grid_search.predict(test_x)

    
    #def clusteringTuning():
def tuning_KNN_custom():
    # Suddividi il dataset in training set e test set
    data = 'Total'
    train_x, test_x, train_y, test_y = DS_Splitter(data)

    # Definisci i range di valori per k e tipi di distanza
    k_range = list(range(1, 32))
    dist_range = ["Euclidean", "Manhattan", "Chebyshev"]
    model_stats = {}

    # Inizializza array per memorizzare le accuratezze durante il tunin

    # Loop attraverso le diverse distanze e valori di k
    for dist in dist_range:
        for k in k_range:
            # Crea un classificatore KNN personalizzato con la configurazione corrente
            clf = Custom_kNN(k = k, distance = dist)
            print("Distanza: ",dist, "  K: ", k)

            # Calcola l'accuratezza media sul set di test
            predictions = clf.fit_predict(train_x,train_y,test_x)

            acc, pre , rec , f1 = tuningMetrics(predictions, test_y)

            # Salva i risultati in un dizionario
            model_stats[f'{dist} with {k} neighbours'] = [acc, pre, rec, f1]
    
    # Estrai il modello con migliori performance
    best_model = max(model_stats, key=model_stats.get)
    max_values = model_stats[best_model]

    return ("metriche migliori = ",best_model, " con ", max_values)

def tuning_RF_custom():
    data = 'Total'
    train_x, test_x, train_y, test_y = DS_Splitter(data)

    # Definisci i range di valori per k e tipi di distanza
    trees_range = list(range(5, 100,10))
    depth_range = list(range(10,50,5))
    model_stats = {}
    # Addestra con tutte le possibili combinazioni
    for trees in trees_range:
        for depth in depth_range:
            RF = CustomRandomForest(trees,depth)
            print("Alberi :",trees, " Profondità : ",depth)
            predictions = RF.fit_predict(train_x,train_y,test_x)

            acc, pre , rec , f1 = tuningMetrics(predictions, test_y)
            # Salva i risultati in un dizionario
            model_stats[f'{trees} trees with depth : {depth}'] = [acc, pre, rec, f1]

    # Estrai il modello con migliori performance
    best_model = max(model_stats, key=model_stats.get)
    max_values = model_stats[best_model]

    return ("metriche migliori = ",best_model, " con ", max_values)

#print(tuning_KNN_custom())
#print(tuning_RF_custom())