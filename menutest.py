import tkinter as tk
from tkinter import messagebox

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Finestra Principale")
        
        # Configura la dimensione della finestra principale
        self.root.geometry("300x250")

        # Creazione del pulsante per aprire la nuova finestra
        open_button = tk.Button(self.root, text="Apri Nuova Finestra", command=self.open_new_window)
        open_button.pack(pady=20)

        # Creazione del menù a tendina
        self.dropdown_value = tk.StringVar(value="Seleziona un'opzione")
        options = ["Opzione 1", "Opzione 2", "Opzione 3"]
        dropdown = tk.OptionMenu(self.root, self.dropdown_value, *options)
        dropdown.pack(pady=20)

    def open_new_window(self):
        # Crea una nuova finestra
        new_win = tk.Toplevel(self.root)
        new_win.title("Nuova Finestra")
        
        # Configura la dimensione della nuova finestra
        new_win.geometry("300x200")

        # Variabili per memorizzare i valori selezionati delle checkbox
        self.checkbox_values = {
            "Opzione 1": tk.IntVar(),
            "Opzione 2": tk.IntVar(),
            "Opzione 3": tk.IntVar()
        }

        # Creazione delle checkbox
        tk.Label(new_win, text="Seleziona le opzioni:").pack(anchor=tk.W, pady=10)
        for option, var in self.checkbox_values.items():
            checkbox = tk.Checkbutton(new_win, text=option, variable=var)
            checkbox.pack(anchor=tk.W)

        # Bottone per inviare i valori delle checkbox
        submit_button = tk.Button(new_win, text="Invia", command=self.collect_checkbox_values)
        submit_button.pack(pady=20)

    def collect_checkbox_values(self):
        # Raccogli i valori selezionati delle checkbox
        selected_checkboxes = [option for option, var in self.checkbox_values.items() if var.get() == 1]
        
        # Passa i valori alla funzione di elaborazione
        self.process_selection(selected_checkboxes)

    def process_selection(self, selected_checkboxes):
        # Stampa i valori selezionati
        print("Checkbox selezionate:", selected_checkboxes)

        # Mostra i valori selezionati in un messaggio di avviso
        messagebox.showinfo("Selezioni", f"Checkbox selezionate: {', '.join(selected_checkboxes)}")

        # Passa i valori selezionati alla funzione per creare una nuova finestra
        self.show_selected_values(selected_checkboxes)

    def show_selected_values(self, selected_values):
        # Crea una nuova finestra
        result_win = tk.Toplevel(self.root)
        result_win.title("Valori Selezionati")
        
        # Configura la dimensione della nuova finestra
        result_win.geometry("300x200")

        # Mostra i valori selezionati nella nuova finestra
        tk.Label(result_win, text="Hai selezionato:").pack(pady=10)
        for value in selected_values:
            tk.Label(result_win, text=value).pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
