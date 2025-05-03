import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class Interfaz:
    def __init__(self, root):
        self.root = root
        self.root.title("Tarea 1 - Aida Cardenas")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        # estilos de la interfaz
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#E6F3FF') #fondo
        self.style.configure('TLabel', background='#E6F3FF', foreground='#003366', font=('Consolas', 10)) #texto
        self.style.configure('TButton', background='#4D94FF', foreground='#003366', font=('Consolas', 10)) #boton  
        self.style.configure('TLabelframe', background='#E6F3FF', foreground='#003366', font=('Consolas', 10)) 
        self.style.configure('TLabelframe.Label', background='#E6F3FF', foreground='#003366', font=('Consolas', 10))
        self.style.configure('TRadiobutton', background='#E6F3FF', foreground='#003366', font=('Consolas', 10))
        self.style.configure('TNotebook', background='#E6F3FF')
        self.style.configure('TNotebook.Tab', background='#4D94FF', foreground='#003366', font=('Consolas', 10))
        
        # variables de calculo
        self.sesgo = 0
        self.pesos = []
        self.vectores = []
        self.funcion_activacion = tk.StringVar(value="escalon")
        
        # se crea la interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # configuracion
        config_frame = ttk.LabelFrame(main_frame, text="Configuración del perceptron", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(config_frame, text="Cargar archivo de configuración", command=self.cargar_configuracion).pack(fill=tk.X, pady=5)
        
        self.config_label = ttk.Label(config_frame, text="Para cargar la configuracion presione el boton de arriba. Solo acepta archivos .txt")
        self.config_label.pack(fill=tk.X, pady=5)
        
        # funciones de activacion
        activacion_frame = ttk.LabelFrame(main_frame, text="Función de activación", padding="10")
        activacion_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(activacion_frame, text="Escalón", variable=self.funcion_activacion, value="escalon").pack(anchor=tk.W)
        ttk.Radiobutton(activacion_frame, text="Sigmoide", variable=self.funcion_activacion, value="sigmoide").pack(anchor=tk.W)
        
        # entrada de datos
        entrada_frame = ttk.LabelFrame(main_frame, text="Entrada de datos", padding="10")
        entrada_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
            # pestañas para los modos de entrada
        tab_control = ttk.Notebook(entrada_frame)
        
                # pestaña manual
        tab_manual = ttk.Frame(tab_control)
        tab_control.add(tab_manual, text="Entrada manual")
        
                # frame manual
        manual_frame = ttk.Frame(tab_manual, padding="10")
        manual_frame.pack(fill=tk.BOTH, expand=True)
        
        self.entradas_frame = ttk.Frame(manual_frame)
        self.entradas_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(manual_frame, text="Calcular", command=self.calcular_manual).pack(fill=tk.X, pady=5)
        
                # pestaña archivo
        tab_archivo = ttk.Frame(tab_control)
        tab_control.add(tab_archivo, text="Entrada de un archivo")
        
                # frame archivo
        archivo_frame = ttk.Frame(tab_archivo, padding="10")
        archivo_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(archivo_frame, text="Cargar archivo con vectores", command=self.cargar_vectores).pack(fill=tk.X, pady=5)
        
        ttk.Button(archivo_frame, text="Calcular", command=self.calcular_archivo).pack(fill=tk.X, pady=5)
        
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # resultados por favor funciona
        resultados_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        resultados_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.resultados_text = ScrolledText(resultados_frame, wrap=tk.WORD, height=10, 
                                          bg='#E6F3FF', fg='#003366',
                                          font=('Consolas', 10))
        self.resultados_text.pack(fill=tk.BOTH, expand=True)

    #cargar configuracion     
    def cargar_configuracion(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de configuracion",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not archivo:
            return
            
        try:
            with open(archivo, 'r') as f:
                linea = f.readline().strip()
                valores = [float(x) for x in linea.split(',')]
                
                if len(valores) < 2:
                    messagebox.showerror("Error", "El archivo de configuracion debe tener al menos 2 valores (el sesgo y min. un peso)")
                    return
                    
                self.sesgo = valores[0]
                self.pesos = valores[1:]
                
                self.config_label.config(text=f"Sesgo: {self.sesgo}, Pesos: {self.pesos}")
                self.actualizar_entradas_manuales()
        # validaciones y errores
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el archivo {archivo}")
        except ValueError:
            messagebox.showerror("Error", "El archivo de configuracion solo puede contener numeros separados por comas :(")
    
    def actualizar_entradas_manuales(self):
        # limpiar historial
        for widget in self.entradas_frame.winfo_children():
            widget.destroy()
        
        # crear nueva entrada manual
        self.entradas_manuales = []
        for i in range(len(self.pesos)):
            frame = ttk.Frame(self.entradas_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"x{i+1}:").pack(side=tk.LEFT, padx=5)
            entrada = ttk.Entry(frame)
            entrada.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.entradas_manuales.append(entrada)
    # cargar vectores :)
    def cargar_vectores(self):
        archivo = filedialog.askopenfilename(
            title="cargar archivo de vectores",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        if not archivo:
            return
            
        try:
            with open(archivo, 'r') as f:
                self.vectores = []
                for linea in f:
                    valores = [float(x) for x in linea.strip().split(',')]
                    self.vectores.append(valores)
                # ventana de info jeje
                messagebox.showinfo("Información", f"Se cargaron {len(self.vectores)} vectores del archivo")
         # errores de archivo 
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontro el archivo {archivo}")
        except ValueError:
            messagebox.showerror("Error", "El archivo debe contener solo números separados por comas")
    
    def suma_ponderada(self, vector):
        if len(vector) != len(self.pesos):
            return None
        
        resultado = self.sesgo
        for i in range(len(vector)):
            resultado += vector[i] * self.pesos[i]
        
        return resultado
    
    def activacion_escalon(self, valor):
        return 1 if valor >= 0 else 0
    
    def activacion_sigmoide(self, valor):
        return 1 / (1 + np.exp(-valor))
    
    def calcular_manual(self):
        if not self.pesos:
            messagebox.showerror("Error", "Por favor cargue una primero")
            return
        
        try:
            vector = [float(entrada.get()) for entrada in self.entradas_manuales]
            suma = self.suma_ponderada(vector)
            
            if suma is not None:
                if self.funcion_activacion.get() == "escalon":
                    resultado = self.activacion_escalon(suma)
                    self.resultados_text.insert(tk.END, f"Vector: {vector}\n")
                    self.resultados_text.insert(tk.END, f"Suma ponderada: {suma}\n")
                    self.resultados_text.insert(tk.END, f"Respuesta del perceptrón (escalón): {resultado}\n")
                    self.resultados_text.insert(tk.END, "-" * 40 + "\n")
                else:
                    resultado = self.activacion_sigmoide(suma)
                    self.resultados_text.insert(tk.END, f"Vector: {vector}\n")
                    self.resultados_text.insert(tk.END, f"Suma ponderada: {suma}\n")
                    self.resultados_text.insert(tk.END, f"Respuesta del perceptrón (sigmoide): {resultado}\n")
                    self.resultados_text.insert(tk.END, "-" * 40 + "\n")
                
                self.resultados_text.see(tk.END)
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese numeros validos")
    
    def calcular_archivo(self):
        if not self.pesos:
            messagebox.showerror("Error", "Por favor cargue una configuracion")
            return
        
        if not self.vectores:
            messagebox.showerror("Error", "Por favor cargue un archivo con vectores")
            return
        
        self.resultados_text.delete(1.0, tk.END)
        self.resultados_text.insert(tk.END, f"Procesando {len(self.vectores)} vectores\n\n")
        
        for i, vector in enumerate(self.vectores):
            if len(vector) != len(self.pesos):
                self.resultados_text.insert(tk.END, f"Error en vector {i+1}: debe tener {len(self.pesos)} componentes\n")
                continue
                
            suma = self.suma_ponderada(vector)
            
            if self.funcion_activacion.get() == "escalon":
                resultado = self.activacion_escalon(suma)
                self.resultados_text.insert(tk.END, f"Vector {i+1}: {vector}\n")
                self.resultados_text.insert(tk.END, f"Suma ponderada: {suma}\n")
                self.resultados_text.insert(tk.END, f"Respuesta del perceptrón (función escalón): {resultado}\n")
            else:
                resultado = self.activacion_sigmoide(suma)
                self.resultados_text.insert(tk.END, f"Vector {i+1}: {vector}\n")
                self.resultados_text.insert(tk.END, f"Suma ponderada: {suma}\n")
                self.resultados_text.insert(tk.END, f"Respuesta del perceptrón (función sigmoide): {resultado}\n")
            
            self.resultados_text.insert(tk.END, "-" * 40 + "\n")
        
        self.resultados_text.see(tk.END)

def main():
    root = tk.Tk()
    app = Interfaz(root)
    root.mainloop()

if __name__ == "__main__":
    main() 