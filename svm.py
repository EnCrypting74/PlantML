from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



class svm():

    
    def svm_fit(train_x, test_x, train_y, test_y):

        #Addestra il classificatore SVM
        svm_clf = SVC()
        svm_clf.fit(train_x, train_y)
        
        #Predici le etichette per il set di test
        pred_y = svm_clf.predict(test_x)

        return pred_y
