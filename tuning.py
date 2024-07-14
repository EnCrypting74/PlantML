import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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