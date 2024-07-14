import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class decisionTree():

    def decisionTree_fit(train_x, test_x, train_y, test_y):
        
        # Addestra il classificatore decision tree
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(train_x, train_y)
        
        # Predici le etichette per il set di test
        y_pred = clf.predict(test_x)

        return y_pred