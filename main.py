# Import per il menù GUI
from tkinter import *
from tkinter import ttk

# Import di librerie standard
import numpy as np
from sklearn.datasets import load_files
from PIL import Image
import os
import matplotlib.pyplot as plt

# Import di funzioni create
from custom_kNN import Custom_kNN as cNN
from DataSetSplitter import DS_Splitter
from metrics import calculateMetrics
from custom_Multi import CustomRandomForest as CRF
from MatriciDiConfusione import MatriciDiConfusione

class Menu():
        
    def __init__(self,root):
        # Creiamo la finestra del menù
        self.root = root
        root.title("Progetto Machine Learning")
        root.geometry("1024x768")
        mainframe = ttk.Frame(root,padding = "20 5 20 5", relief='raised')


        mainframe.grid(column = 0, row = 0, sticky = (N,W,E,S))
        root.columnconfigure(0, weight = 1)
        root.rowconfigure(0,weight = 1)
        
        # Inseriamo i pulsanti e le scritte necessarie 
        ttk.Label(mainframe, text = 'Scegliere un opzione ').grid(column = 3, row = 1, sticky = (N,W,E,S))

        ttk.Button(mainframe, text="Explore DataSet", command = self.selectDataSet(mainframe)).grid(column=1, row=2, sticky=W)
        ttk.Button(mainframe, text="Train Model", command = self.selectTrain(mainframe)).grid(column=1, row=3, sticky=W)

        ttk.Button(mainframe, text="Exit", command = exit).grid(column=3, row=4, sticky=S)

    def selectDataSet(self,mainframe):
        # Creiamo la schermata per l'analisi del dataset
        dataset_frame = ttk.Frame(mainframe, padding="20 5 20 5")
        dataset_frame.grid(column = 0, row = 0, sticky = (N,W,E,S))

        ttk.Label(dataset_frame, text="Select Option").grid(column = 3, row = 1, sticky = (N,W,E,S))

        ttk.Button(dataset_frame, text="Back", command = None).grid(column=3, row=4, sticky=S)

    def selectTrain(self,mainframe):
        # Creiamo la schermata per la scelta del modello
        train_frame = ttk.Frame(mainframe ,padding="20 5 20 5")
        train_frame.grid(column = 0, row = 0, sticky = (N,W,E,S))

        # Seleziona il classificatore
        ttk.Label(train_frame, text = "Select Classifier : ").grid(column = 3, row = 1, sticky = (N,W,E,S))
        ttk.Button(train_frame, text = "SVM", command = self.SVM_clas).grid(column = 1, row = 3, sticky = W)
        ttk.Button(train_frame, text = "Standard 2", command = None).grid(column = 2, row = 3, sticky = W)
        ttk.Button(train_frame, text = "Standard 3", command = None).grid(column = 3, row = 3, sticky = (N,S))
        ttk.Button(train_frame, text = "Custom kNN", command = self.CustomKNN).grid(column = 4, row = 3, sticky = E)
        ttk.Button(train_frame, text = "Custom RForest", command = self.CustomRF).grid(column = 5, row = 3, sticky = E)

        ttk.Button(train_frame, text="Back", command = None).grid(column=3, row=4, sticky=S)
    
    def SVM_clas(self):
        return

    def CustomKNN(self):

        window = Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom kNN results ")

        data = 'Mixed'
        distance = 'Manhattan'

        ttk.Label(window, text = ("Dataset = ",data," distance type =",distance)).grid(column = 1, row = 2, sticky = W)

        train_x,test_x, train_y, test_y = DS_Splitter(data)
        kNN_clas = cNN(distance = distance)
        
        kNN_clas.fit(train_x,train_y)
        pred_y = kNN_clas.predict(test_x)

        output = calculateMetrics(pred_y,test_y)
        ttk.Label(window, text = output).grid(column = 1, row = 3, sticky = W)
        #MatriciDiConfusione(pred_y, test_y)

        return

    def CustomRF(self):

        window = Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom Random Forest results ")

        data = 'Mixed'

        Multi_clas = CRF()

        train_x,test_x, train_y, test_y = DS_Splitter(data)

        Multi_clas.fit(train_x,train_y)

        pred_y = Multi_clas.predict(test_x)

        ttk.Label(window, text = (Multi_clas.num_trees," alberi di profondità ",Multi_clas.depth)).grid(column = 1, row = 1, sticky = W)
        output = calculateMetrics(pred_y,test_y)
        ttk.Label(window, text = output).grid(column = 1, row = 2, sticky = W)
        return


root = Tk()
menu = Menu(root)
root.mainloop()