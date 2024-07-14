# Import per il menù GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import di librerie standard
import pandas as pd

# Import di funzioni create
from custom_kNN import Custom_kNN as cNN
from DataSetSplitter import DS_Splitter
from metrics import calculateMetrics, histo, calc_zeros, show_auc, find_outliers, calc_nan, scatterPlot
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
        mainscreen = ttk.Frame(self.root, padding = "40 20 40 20", relief = "raised")
        mainscreen.pack(pady=10)
        
        # Inseriamo il testo ed i pulsanti iniziali
        ttk.Label(mainscreen, text = 'Select an option').pack(pady=10)

        tk.Button(mainscreen, text="Explore Dataset", command=self.selectDataSet).pack(pady=10)

        tk.Button(mainscreen, text="Train Model", command=self.selectTrain).pack(pady=10)

        tk.Button(mainscreen, text="Best Model", command=self.selectBest).pack(pady=10)

        # Pulsante per terminare il programma
        ttk.Button(mainscreen, text = "Exit", command = exit).pack(pady=10)

    def selectDataSet(self):
        # Creiamo la schermata per l'analisi del dataset
        dataset_screen = tk.Toplevel(self.root)
        dataset_screen.title("Dataset")
        dataset_screen.geometry("1000x700")
        # Creiamo il frame che conterrà il testo
        descriptor_frame = ttk.Frame(dataset_screen)
        descriptor_frame.pack(side = tk.LEFT, pady = 10)

        # Creiamo il frame che conterrà la visualizzazione del dataset
        # Mettiamo a schermo le informazioni sul dataset
        ds_info = open('ds_info.txt', 'r')
        ttk.Label(descriptor_frame, text = "Descrizione Dataset :").pack(pady = 10)
        ttk.Label(descriptor_frame, text = ds_info.readlines()).pack(pady = 10)

        # Mostriamo una porzione del dataset
        ttk.Label(descriptor_frame, text = "Prime 5 righe del file Texture : ").pack(pady = 10)
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Tex_64.txt", header = None).iloc[:,:15].head()).pack(pady = 10)

        # Calcoliamo e mostriamo il numero di zeri nel dataset
        zeros, dict_zeros = calc_zeros()
        f1 = open("diz_zeri.txt",'w+') # Dopo aver visualizzato per la prima volta le informazioni sul dataset, il numero di zeri per feature sarà consultabile in questo file
        f1.write(f'{(dict_zeros)}')
        f1.close()

        ttk.Label(descriptor_frame, text = f'Nel dataset sono presenti {zeros} zeri su 307200 valori').pack(pady = 10)  

        # Calcoliamo e mostriamo il numero di Nan nel dataset
        ttk.Label(descriptor_frame, text = f'Nel dataset sono presenti {calc_nan()} Nan').pack(pady = 10)   

        # Calcoliamo e mostriamo il numero di outliers nel dataset
        num_outliers, dict_outliers = find_outliers()
        f2 = open("diz_out.txt", 'w') # Dopo aver visualizzato per la prima volta le informazioni sul dataset, il numero di outliers per feature sarà consultabile in questo file
        f2.write(f'{dict_outliers}')
        f2.close()

        ttk.Label(descriptor_frame, text = f'Nel dataset sono presenti {num_outliers} outliers').pack(pady = 10)

        ttk.Button(descriptor_frame, text = "See more statistics", command = self.dataStat).pack(pady = 10)
        ttk.Button(descriptor_frame, text = "See graphs", command = self.dataGraph).pack(pady = 10)
        ttk.Button(descriptor_frame, text = "Back", command = dataset_screen.destroy).pack(side = tk.BOTTOM, pady = 10)

    def dataStat(self):
        datastat_screen = tk.Toplevel(self.root)
        datastat_screen.title("Dataset")
        datastat_screen.geometry("1000x750")

        # Creiamo il frame che conterrà le info
        descriptor_frame = ttk.Frame(datastat_screen)
        descriptor_frame.pack(side = tk.LEFT, pady = 10)

        # Mostriamo le statistiche delle feature di shape
        ttk.Label(descriptor_frame, text = "Statistiche Shape ").pack(pady=10)
        ttk.Label(descriptor_frame, text = pd.read_csv("Dataset/data_Sha_64.txt", header = None).describe()).pack(pady = 10)

        # Mostriamo le statistiche delle feature di margin
        ttk.Label(descriptor_frame, text = "Statistiche Margin ").pack(pady=10)
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
        train_screen.geometry("500x400")

        # Creiamo i Frame per separare le varie opzioni
        title_frame = tk.Frame(train_screen, relief = "raised")
        title_frame.pack(side = tk.TOP, padx = 10, pady = 10)
        model_frame = tk.Frame(train_screen)
        model_frame.pack(side = tk.LEFT, padx = 10, pady = 10)
        feature_frame = tk.Frame(train_screen)
        feature_frame.pack( padx = 10, pady = 10)
        preproc_frame = tk.Frame(train_screen)
        preproc_frame.pack( padx = 10, pady = 10)
        bottom_frame = tk.Frame(train_screen)
        bottom_frame.pack(side = tk.BOTTOM, padx = 10)

        ttk.Label(title_frame, text = "Model Selection").pack(side = tk.TOP, anchor = "nw")

        # Seleziona il classificatore
        ModelType = ['SVM','DecisionTree','Clustering','Custom_kNN','Custom_RForest']
        ttk.Label(model_frame, text = "          Select Classifier : ").pack(side = tk.TOP, anchor = "nw")

        # Creazione scelta del modello tramite RadioButton (mutualmente esclusivi)
        self.model = tk.StringVar()
        ttk.Radiobutton(model_frame, text = 'SVM', variable = self.model, value = ModelType[0]).pack()
        ttk.Radiobutton(model_frame, text = 'Decision Tree', variable = self.model, value = ModelType[1]).pack()
        ttk.Radiobutton(model_frame, text = 'Clustering', variable = self.model, value = ModelType[2]).pack()
        ttk.Radiobutton(model_frame, text = 'Custom K-NN', variable = self.model, value = ModelType[3]).pack()
        ttk.Radiobutton(model_frame, text = 'Custom Random Forest', variable = self.model, value = ModelType[4]).pack()
 
        # Creazione scelta preprocessing come checkboxes (non mutualmente esclusivi)
        ttk.Label(preproc_frame, text = "Select Preprocessing :   ").pack(side = tk.TOP, anchor = "ne")
        self.checkbox_values = {
                "MinMax": tk.IntVar(),    
                "Standard": tk.IntVar(),
                "Robust": tk.IntVar(),
                "Normalization": tk.IntVar(),
                "Aggregation*16": tk.IntVar(),
                "Aggregation*32": tk.IntVar()
            }
            
        # Creazione delle checkbox per il preprocessing
        for option, var in self.checkbox_values.items():
            tk.Checkbutton(preproc_frame, text = option, variable=var).pack()

        self.dropdown_value = tk.StringVar(value="Feature Selection : ")
        options = ["Total", "Texture", "Shape","Margin","Tex + Sha", "Tex + Mar",  "Sha + Mar"]
        dropdown = tk.OptionMenu(feature_frame, self.dropdown_value, *options)
        dropdown.pack(pady=20)
        
        # Tasti per avviare l'addestramento o tornare indietro
        ttk.Button(bottom_frame, text="Train", command= lambda: self.get_preproc(self.model.get(), self.dropdown_value.get())).pack(side = tk.RIGHT, pady = 10)
        ttk.Button(bottom_frame, text="Back", command = train_screen.destroy).pack(side = tk.LEFT, pady = 10) 

    def get_preproc(self, model, feature):
        # Raccoglie le opzioni di preprocessing 
        preproc_options=[option for option, var in self.checkbox_values.items() if var.get() == 1] if hasattr(self, 'checkbox_values') else []

        self.trainModel(model, preproc_options, feature)

    def trainModel(self, model, preprocessing, feature):
        # Funzione che istanzia il classificatore selezionato con le tecniche di preprocessing scelte
        # Split del dataset in base alla  feature selection : 
        if feature == "Feature Selection : ": feature = "Total"

        data = feature
        train_x,test_x, train_y, test_y = DS_Splitter(data)

        if ("Aggregation*16" in preprocessing) & ("Aggregation*32" in preprocessing):
            messagebox.showinfo("Selezionata doppia aggregazione, selezionare solo una modalità")
            raise TypeError("modello o preprocessing non corretti")
        if len(preprocessing)>2:
            messagebox.showinfo("Selezionate troppe opzioni")
            raise TypeError("modello o preprocessing non corretti")
        
        # Applicazione preprocessing
        if "Standard" in preprocessing:
                train_x, test_x = normalizeDataset(train_x, test_x, n_type = "Standard")
        if "MinMax" in preprocessing:
                train_x, test_x = normalizeDataset(train_x, test_x, n_type = "MinMax")
        if "Robust" in preprocessing:
                train_x, test_x = normalizeDataset(train_x, test_x, n_type = "Robust")
        if "Normalization" in preprocessing:
                train_x, test_x = normalizeDataset(train_x, test_x, n_type = "Normalization")
        if "Aggregation*16" in preprocessing:
                train_x,test_x = aggregateFeatures(train_x, test_x, mode = 16)
        if "Aggregation*32" in preprocessing:
                train_x,test_x = aggregateFeatures(train_x, test_x, mode = 32)

        preprocessing.append(feature)
        # Istanzia il classificatore
        if model == 'SVM':
            return self.SVM_clas(train_x,test_x, train_y, test_y,preprocessing)
        elif model == 'DecisionTree':
            return self.DecisionTree(train_x,test_x, train_y, test_y,preprocessing)
        elif model == 'Clustering':
            return self.Clustering(train_x,test_x, train_y, test_y,preprocessing)
        elif model == 'Custom_kNN':
            return self.CustomKNN(train_x,test_x, train_y, test_y,preprocessing)
        elif model == 'Custom_RForest':
            return self.CustomRF(train_x,test_x, train_y, test_y,preprocessing)
        else:
            messagebox.showinfo("Selezione", model)
            raise TypeError("modello o preprocessing non corretti")

    def SVM_clas(self,train_x,test_x, train_y, test_y,preprocessing):
        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("SupportVectorMachine results ")

        # Funzione per istanziare il classificatore SVM e mostrare le metriche
        predictions = svm.svm_fit(train_x, test_x, train_y, test_y)
        ttk.Label(window, text ="SVM Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =("Selected options :",preprocessing)).pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =calculateMetrics(predictions,test_y)).pack(pady=10)
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)

        # Riportiamo la curva ROC per le prime 10 classi su 100 per non saturare il grafico
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)

        fig= show_auc(test_y,predictions)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)
    
    def DecisionTree(self,train_x,test_x, train_y, test_y,preprocessing):
        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("DecisionTree results ")

        # Funzione per istanziare il classificatore Decision Tree e mostrare le metriche
        predictions = decisionTree.decisionTree_fit(train_x, test_x, train_y, test_y)
        ttk.Label(window, text ="Decision Tree Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =("Selected options :",preprocessing)).pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =calculateMetrics(predictions,test_y)).pack(pady=10)
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
                
        # Riportiamo la curva ROC per le prime 10 classi su 100 per non saturare il grafico
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)

        fig= show_auc(test_y,predictions)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)
    
    def Clustering(self,train_x,test_x, train_y, test_y,preprocessing):
        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("Clustering results ")
    
        # Funzione per istanziare il classificatore di clustering e mostrare le metriche
        ari, silhouette_avg, X, clusters = clustering.cluster_fit(train_x, test_x, train_y, test_y)
        ttk.Label(window, text ="Clustering Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =("Selected options :",preprocessing)).pack(side = tk.TOP, pady=10)
        ttk.Label(window, text = (f"Adjusted Rand Index: {ari}\n",f"Silhouette Score: {silhouette_avg}")).pack(pady=10)
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
        
        # Riportiamo il grafico di clustering
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)
    
        fig = scatterPlot(X, clusters)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)
    
    def CustomKNN(self,train_x,test_x, train_y, test_y,preprocessing):
        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("Custom K-NN results ")
        
        # Funzione per istanziare il classificatore custom kNN e mostrare le metriche

        kNN_clas = cNN()
        
        ttk.Label(window, text ="K-NN Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =("Selected options :",preprocessing)).pack(side = tk.TOP, pady=10)
        ttk.Label(window, text = "K-Nearest Neighbours with distance type = Manhattan and k = 16").pack(pady = 5)
        
        predictions = kNN_clas.fit_predict(train_x,train_y,test_x)
        ttk.Label(window, text = calculateMetrics(predictions ,test_y)).pack()
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
        
        # Riportiamo la curva ROC per le prime 10 classi su 100 per non saturare il grafico
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)

        fig= show_auc(test_y,predictions)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)

    def CustomRF(self,train_x,test_x, train_y, test_y,preprocessing):

        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("Custom Random Forest results ")

        # Funzione per istanziare il classificatore custom Random Forest e mostrare le metriche

        Multi_clas = CRF()

        predictions = Multi_clas.fit_predict(train_x, train_y, test_x)

        ttk.Label(window, text = "Random Forest Results :").pack(side = tk.TOP, pady = 10)
        ttk.Label(window, text =("Selected options :",preprocessing)).pack(side = tk.TOP, pady=10)
        ttk.Label(window, text = (f'{Multi_clas.num_trees} trees with depth = {Multi_clas.depth}')).pack(pady = 10)
        ttk.Label(window, text = calculateMetrics(predictions, test_y)).pack(pady = 10)
        
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)
                
        # Riportiamo la curva ROC per le prime 10 classi su 100 per non saturare il grafico
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)

        fig= show_auc(test_y,predictions)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)

    def selectBest(self):
        # Combinazione migliore di modello e preprocessing
        window = tk.Toplevel(self.root)
        window.geometry("1000x750")
        window.title("SupportVectorMachine results ")

        data = 'Tex + Mar'
        train_x,test_x, train_y, test_y = DS_Splitter(data)
        train_x, test_x = normalizeDataset(train_x, test_x)

        ttk.Label(window, text ="Il modello migliore risulta la Support Vector Machine addestrata sulle feature di texture e margin\nLe performance migliorano di vari punti percentuali con la normalizzazione MinMax").pack(side = tk.TOP, pady=10)

        # Funzione per istanziare il classificatore SVM e mostrare le metriche
        predictions = svm.svm_fit(train_x, test_x, train_y, test_y)
        ttk.Label(window, text ="SVM Results :").pack(side = tk.TOP, pady=10)
        ttk.Label(window, text =calculateMetrics(predictions,test_y)).pack(pady=10)
        ttk.Button(window, text="Back", command = window.destroy).pack(side = tk.BOTTOM, pady = 10)

        # Riportiamo la curva ROC per le prime 10 classi su 100 per non saturare il grafico
        graph_frame = ttk.Frame(window)
        graph_frame.pack(side = tk.BOTTOM, padx = 10, pady = 10)

        fig= show_auc(test_y,predictions)
        canvas = FigureCanvasTkAgg(fig, master = graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady = 20)

root = tk.Tk()
menu = Menu(root)
root.mainloop()