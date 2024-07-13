from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



class svm():

    @staticmethod
    def svm_fit(train_x, test_x, train_y, test_y):
        svm_clf = SVC()
        svm_clf.fit(train_x, train_y)
        
        pred_y = svm_clf.predict(test_x)
        print("Accuracy: ", svm_clf.score(test_x, test_y))
        print()
        return pred_y

    @staticmethod
    def svm_tuning(train_x, train_y):
        
        def gridSearchSVM(train_x, train_y, param_grid):
            svm_clf = SVC()
            n_folds = 5
            grid_search_cv = GridSearchCV(svm_clf, param_grid, cv=n_folds)
            grid_search_cv.fit(train_x, train_y)
            print('Accuratezza media per combinazione:\n', grid_search_cv.cv_results_['mean_test_score'])
            print('Combinazione migliore:\n', grid_search_cv.best_params_)
            print('Accuratezza media della combinazione migliore: %.3f' % grid_search_cv.best_score_)
            print()
        
        param_grid_linear = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]
        param_grid_rbf = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.025, 0.05]}]
        
        print('SVM lineare')
        gridSearchSVM(train_x, train_y, param_grid_linear)
        print('SVM con kernel RBF')
        gridSearchSVM(train_x, train_y, param_grid_rbf)

