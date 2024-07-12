# Import per il menù GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


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
        # Creiamo la finestra principale del menù
        self.root = root
        root.title("Progetto Machine Learning")
        root.geometry("500x400")
        
        root.columnconfigure(0, weight = 1)
        root.rowconfigure(0,weight = 1)

        self.createScreen()

    def createScreen(self):
        #creazione della schermata principale
        mainscreen = ttk.Frame(root,padding = "20 5 20 5", relief='raised')

        mainscreen.pack(pady=10)
        
        # Inseriamo i pulsanti e le scritte necessarie 
        ttk.Label(mainscreen, text = 'Select an option').pack(pady=10)

        tk.Button(mainscreen, text="Explore Dataset", command=self.selectDataSet).pack(pady=10)

        tk.Button(mainscreen, text="Train Model", command=self.selectTrain).pack(pady=10)

        ttk.Button(mainscreen, text = "Exit", command = exit).pack(pady=10)

    def selectDataSet(self):
        # Creiamo la schermata per l'analisi del dataset
        dataset_screen = tk.Toplevel(self.root)
        dataset_screen.title("Dataset")
        dataset_frame = ttk.Frame(dataset_screen)
        dataset_frame.pack(pady=10)

        ttk.Label(dataset_frame, text="Select Option").pack(pady=10)

        ttk.Button(dataset_frame, text="Back", command = dataset_screen.destroy).pack(pady=10)

    def selectTrain(self):
        # Creiamo la schermata per la scelta del modello
        train_screen = tk.Toplevel(self.root)
        train_screen.title("Training")
        model_frame = tk.Frame(train_screen)
        model_frame.pack(side = tk.LEFT, padx = 10, pady = 10)
        preproc_frame = tk.Frame(train_screen)
        preproc_frame.pack(side = tk.RIGHT, padx = 10, pady = 10)
        bottom_frame = tk.Frame(train_screen)
        bottom_frame.pack(side = tk.BOTTOM, padx = 20)

        ModelType = ['SVM','DecisionTree','Standard 3','Custom_kNN','Custom_RForest']

        # Seleziona il classificatore
        ttk.Label(model_frame, text = "Select Classifier : ").pack(pady = 10)

        # Creazione scelta del modello
        self.model = tk.StringVar()
        SVM = ttk.Radiobutton(model_frame, text = 'SVM', variable = self.model, value = ModelType[0]).pack()
        DecisionTree = ttk.Radiobutton(model_frame, text = 'Decision Tree', variable = self.model, value = ModelType[1]).pack()
        Standard = ttk.Radiobutton(model_frame, text = 'Mobile', variable = self.model, value = ModelType[2]).pack()
        Custom_kNN = ttk.Radiobutton(model_frame, text = 'Custom kNN', variable = self.model, value = ModelType[3]).pack()
        Custom_RForest = ttk.Radiobutton(model_frame, text = 'Custom Random Forest', variable = self.model, value = ModelType[4]).pack()
 
        # Creazione scelta preprocessing
        ttk.Label(preproc_frame, text = "Select Preprocessing : ").pack()
        self.checkbox_values = {
                "Opzione 1": tk.IntVar(),
                "Opzione 2": tk.IntVar(),
                "Opzione 3": tk.IntVar(),
                "Opzione 4": tk.IntVar()
            }
            
        # Creazione delle checkbox
        for option, var in self.checkbox_values.items():
            checkbox = tk.Checkbutton(preproc_frame, text = option, variable=var).pack()

        preproc_options=[option for option, var in self.checkbox_values.items() if var.get() == 1]

        ttk.Button(bottom_frame, text="Train", command= lambda: self.trainModel(self.model.get(), preproc_options)).pack(side = tk.RIGHT, pady = 10)
        ttk.Button(bottom_frame, text="Back", command = train_screen.destroy).pack(side = tk.LEFT, pady = 10)


    def trainModel(self, model, preprocessing):
        if model == 'SVM':
            return self.SVM_clas()
        elif model == 'DecisionTree':
            return
        elif model == 'Standard 3':
            return
        elif model == 'Custom_kNN':
            return self.CustomKNN()
        elif model == 'Custom_RForest':
            return self.CustomRF()
        else:
            messagebox.showinfo("Selezione", model)
            raise TypeError("modello o preprocessing non corretti")

    def SVM_clas(self):
        return
    
    def CustomKNN(self):

        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom kNN results ")

        data = 'Total'

        train_x,test_x, train_y, test_y = DS_Splitter(data)
        kNN_clas = cNN()
        
        ttk.Label(window, text = ("K-Nearest Neighbors on total dataset with distance type =",kNN_clas.distance,"and k = ",kNN_clas.k)).pack(pady = 5)
        
        ttk.Label(window, text = calculateMetrics(kNN_clas.fit_predict(train_x,train_y,test_x),test_y)).pack()
        #MatriciDiConfusione(pred_y, test_y)

        return

    def CustomRF(self):

        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom Random Forest results ")

        data = 'Total'

        Multi_clas = CRF()

        train_x,test_x, train_y, test_y = DS_Splitter(data)

        Multi_clas.fit_predict(train_x, train_y, test_x)

        ttk.Label(window, text = (Multi_clas.num_trees," alberi di profondità ",Multi_clas.depth)).pack(pady=10)
       
        ttk.Label(window, text =calculateMetrics(Multi_clas.fit_predict(train_x, train_y, test_x),test_y)).pack(pady=10)
        return


root = tk.Tk()
menu = Menu(root)
root.mainloop()