# Import per il menù GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import di librerie standard
import numpy as np
from sklearn.datasets import load_files
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import di funzioni create
from custom_kNN import Custom_kNN as cNN
from DataSetSplitter import DS_Splitter
from metrics import calculateMetrics, histo, calc_zeros
from custom_Multi import CustomRandomForest as CRF
from Preprocessing import normalizeDataset, aggregateFeatures
from svm import svm
from decisionTree import decisionTree
from clustering import clustering

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
        mainscreen = ttk.Frame(self.root,padding = "20 5 20 5", relief='raised')
        mainscreen.pack(pady=10)
        
        # Inseriamo il testo ed i pulsanti iniziali
        ttk.Label(mainscreen, text = 'Select an option').pack(pady=10)

        tk.Button(mainscreen, text="Explore Dataset", command=self.selectDataSet).pack(pady=10)

        tk.Button(mainscreen, text="Train Model", command=self.selectTrain).pack(pady=10)

        # Pulsante per terminare il programma
        ttk.Button(mainscreen, text = "Exit", command = exit).pack(pady=10)

    def selectDataSet(self):
        # Creiamo la schermata per l'analisi del dataset
        dataset_screen = tk.Toplevel(self.root)
        dataset_screen.title("Dataset")
        dataset_screen.geometry("1000x500")
        # Creiamo il frame che conterrà il testo
        descriptor_frame = ttk.Frame(dataset_screen)
        descriptor_frame.pack(side = tk.LEFT, pady = 10)

        # Creiamo il frame che conterrà la visualizzazione del dataset
        # Mettiamo a schermo le informazioni sul dataset
        ds_info = open('ds_info.txt', 'r')
        ttk.Label(descriptor_frame, text = "Descrizione Dataset :").pack(pady = 10)
        ttk.Label(descriptor_frame, text = ds_info.readlines()).pack(pady = 10)

        # Mostriamo una porzione del dataset
        ttk.Label(descriptor_frame, text = "Prime 5 righe del file Shape : ").pack(pady = 10)
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Sha_64.txt", header = None).iloc[:,:15].head()).pack(pady = 10)

        ttk.Label(descriptor_frame, text = f'Nel dataset sono presenti {calc_zeros()}zeri').pack(pady = 10)   

        ttk.Button(descriptor_frame, text = "See more info", command = self.dataStat).pack(pady = 10)
        ttk.Button(descriptor_frame, text = "See graphs", command = self.dataGraph).pack(pady = 10)
        ttk.Button(descriptor_frame, text="Back", command = dataset_screen.destroy).pack(side = tk.BOTTOM, pady = 10)

    def dataStat(self):
        datastat_screen = tk.Toplevel(self.root)
        datastat_screen.title("Dataset")
        datastat_screen.geometry("1000x750")

        # Creiamo il frame che conterrà le info
        descriptor_frame = ttk.Frame(datastat_screen)
        descriptor_frame.pack(side = tk.LEFT, pady = 10)

        # Mostriamo le statistiche delle feature di shape
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Sha_64.txt", header = None).describe()).pack(pady = 10)

        # Mostriamo le statistiche delle feature di margin
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Sha_64.txt", header = None).describe()).pack(pady = 10)

        # Mostriamo le statistiche delle feature di Texture
        ttk.Label(descriptor_frame, text = "Statistiche Texture ").pack(pady=10)
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Sha_64.txt", header = None).describe()).pack(pady = 10)
        
        ttk.Button(descriptor_frame, text = "Back", command = datastat_screen.destroy).pack(side = tk.BOTTOM, pady = 10)
 
    def dataGraph(self):
        datagraph_screen = tk.Toplevel(self.root)
        datagraph_screen.title("Dataset")
        datagraph_screen.geometry("500x600")

        # Creiamo il frame che conterrà le info
        descriptor_frame = ttk.Frame(datagraph_screen)
        descriptor_frame.pack(pady = 10)

        fig,ax = histo('Margin')
        canvas = FigureCanvasTkAgg(fig, master = descriptor_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)
        
        fig,ax = histo('Texture')
        canvas = FigureCanvasTkAgg(fig, master = descriptor_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)

        ttk.Button(descriptor_frame, text = "Back", command = datagraph_screen.destroy).pack(side = tk.BOTTOM, pady = 10)
 
        return
    
    def selectTrain(self):
        # Creiamo la schermata per la scelta del modello
        train_screen = tk.Toplevel(self.root)
        train_screen.title("Training")
        train_screen.geometry("500x300")

        # Creiamo i Frame per separare le varie opzioni
        model_frame = tk.Frame(train_screen)
        model_frame.pack(side = tk.LEFT, padx = 10, pady = 10)
        preproc_frame = tk.Frame(train_screen)
        preproc_frame.pack(side = tk.RIGHT, padx = 10, pady = 10)
        bottom_frame = tk.Frame(train_screen)
        bottom_frame.pack(side = tk.BOTTOM, padx = 20)

        # Seleziona il classificatore
        ModelType = ['SVM','DecisionTree','Clustering','Custom_kNN','Custom_RForest']
        ttk.Label(model_frame, text = "Select Classifier : ").pack(side = tk.TOP, anchor = "nw")

        # Creazione scelta del modello tramite RadioButton (mutualmente esclusivi)
        self.model = tk.StringVar()
        ttk.Radiobutton(model_frame, text = 'SVM', variable = self.model, value = ModelType[0]).pack()
        ttk.Radiobutton(model_frame, text = 'Decision Tree', variable = self.model, value = ModelType[1]).pack()
        ttk.Radiobutton(model_frame, text = 'Clustering', variable = self.model, value = ModelType[2]).pack()
        ttk.Radiobutton(model_frame, text = 'Custom K-NN', variable = self.model, value = ModelType[3]).pack()
        ttk.Radiobutton(model_frame, text = 'Custom Random Forest', variable = self.model, value = ModelType[4]).pack()
 
        # Creazione scelta preprocessing come checkboxes (non mutualmente esclusivi)
        ttk.Label(preproc_frame, text = "Select Preprocessing : ").pack(side = tk.TOP, anchor = "ne")
        self.checkbox_values = {
                "Normalization": tk.IntVar(),
                "Aggregation": tk.IntVar(),
                "Feature Selection": tk.IntVar(),
                "Add Synthetic Record": tk.IntVar(),
                "Sampling": tk.IntVar()
            }
            
        # Creazione delle checkbox per il preprocessing
        for option, var in self.checkbox_values.items():
            tk.Checkbutton(preproc_frame, text = option, variable=var).pack()
        
        # Tasti per avviare l'addestramento o tornare indietro
        ttk.Button(bottom_frame, text="Train", command= lambda: self.get_preproc(self.model.get())).pack(side = tk.RIGHT, pady = 10)
        ttk.Button(bottom_frame, text="Back", command = train_screen.destroy).pack(side = tk.LEFT, pady = 10) 

    def get_preproc(self, model):
        # Raccoglie le opzioni di preprocessing 
        preproc_options=[option for option, var in self.checkbox_values.items() if var.get() == 1] if hasattr(self, 'checkbox_values') else []

        self.trainModel(model, preproc_options)

    def trainModel(self, model, preprocessing):
        # Funzione che istanzia il classificatore selezionato con le tecniche di preprocessing scelte
        data = 'Total'
        print(preprocessing)

        train_x,test_x, train_y, test_y = DS_Splitter(data)

        if "Normalization" in preprocessing:
                train_x, test_x = normalizeDataset(train_x, test_x)
        if "Aggregation" in preprocessing:
                train_x,test_x = aggregateFeatures(train_x, test_x)
        if "Add Synthetic Record" in preprocessing:
                return
        if model == 'SVM':
            return self.SVM_clas(train_x,test_x, train_y, test_y)
        elif model == 'DecisionTree':
            return self.DecisionTree(train_x,test_x, train_y, test_y)
        elif model == 'Clustering':
            return self.Clustering(train_x,test_x, train_y, test_y)
        elif model == 'Custom_kNN':
            return self.CustomKNN(train_x,test_x, train_y, test_y)
        elif model == 'Custom_RForest':
            return self.CustomRF(train_x,test_x, train_y, test_y)
        else:
            messagebox.showinfo("Selezione", model)
            raise TypeError("modello o preprocessing non corretti")

    def SVM_clas(self,train_x,test_x, train_y, test_y):
        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("SupportVectorMachine results ")

        # Funzione per istanziare il classificatore SVM

        ttk.Label(window, text ="SVM Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =calculateMetrics(svm.svm_fit(train_x, test_x, train_y, test_y),test_y)).pack(pady=10)
        
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
    
    def DecisionTree(self,train_x,test_x, train_y, test_y):
        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("DecisionTree results ")

        # Funzione per istanziare il classificatore Decision Tree
        
        ttk.Label(window, text ="Decision Tree Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =calculateMetrics(decisionTree.decisionTree_fit(train_x, test_x, train_y, test_y),test_y)).pack(pady=10)

        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
    
    def Clustering(self,train_x,test_x, train_y, test_y):
        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("Clustering results ")

        # Funzione per istanziare il classificatore di clustering
        
        ttk.Label(window, text ="Clustering Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =clustering.cluster_fit(train_x, test_x, train_y, test_y)).pack(pady=10)

        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
    
    def CustomKNN(self,train_x,test_x, train_y, test_y):
        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom K-NN results ")
        
        # Funzione per istanziare il classificatore custom kNN

        kNN_clas = cNN()
        
        ttk.Label(window, text ="K-NN Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text = ("K-Nearest Neighbors with distance type =",kNN_clas.distance,"and k = ",kNN_clas.k)).pack(pady = 5)
        
        ttk.Label(window, text = calculateMetrics(kNN_clas.fit_predict(train_x,train_y,test_x),test_y)).pack()
        #MatriciDiConfusione(pred_y, test_y)

        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)

    def CustomRF(self,train_x,test_x, train_y, test_y):

        window = tk.Toplevel(self.root)
        window.geometry("500x400")
        window.title("Custom Random Forest results ")

        # Funzione per istanziare il classificatore custom Random Forest

        Multi_clas = CRF()

        Multi_clas.fit_predict(train_x, train_y, test_x)

        ttk.Label(window, text ="Random Forest Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text = (Multi_clas.num_trees," trees with depth = ",Multi_clas.depth)).pack(pady=10)
        ttk.Label(window, text =calculateMetrics(Multi_clas.fit_predict(train_x, train_y, test_x),test_y)).pack(pady=10)
        
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)



root = tk.Tk()
menu = Menu(root)
root.mainloop()