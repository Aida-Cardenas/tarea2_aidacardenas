import numpy as np #operaciones
import tkinter as tk #interfaz
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt #graficos
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle #guardar y cargar

class PerceptronMulticapa:
    def __init__(self, n_entrada, n_salida, n_capas_ocultas, n_neuronas_ocultas):
        self.n_entrada = n_entrada
        self.n_salida = n_salida
        self.n_capas_ocultas = n_capas_ocultas
        self.n_neuronas_ocultas = n_neuronas_ocultas
        
        # pesos y sesgos
        self.pesos = []
        self.sesgos = []
        self.pesos.append(np.random.randn(n_entrada, n_neuronas_ocultas) * 0.01)
        self.sesgos.append(np.zeros((1, n_neuronas_ocultas)))
        
        # capas ocultas
        for _ in range(n_capas_ocultas - 1):
            self.pesos.append(np.random.randn(n_neuronas_ocultas, n_neuronas_ocultas) * 0.01)
            self.sesgos.append(np.zeros((1, n_neuronas_ocultas)))
        # capa oculta a salida
        self.pesos.append(np.random.randn(n_neuronas_ocultas, n_salida) * 0.01)
        self.sesgos.append(np.zeros((1, n_salida)))
    
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivada_sigmoide(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activaciones = [X]
        self.z = []
        
        # forward propagation yay
        for i in range(len(self.pesos)):
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.sesgos[i]
            self.z.append(z)
            activacion = self.sigmoide(z)
            self.activaciones.append(activacion)
        
        return self.activaciones[-1]
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # calculo error de salida
        error = self.activaciones[-1] - y
        delta = error * self.derivada_sigmoide(self.activaciones[-1])
        
        # backpropagation ajksdf
        for i in range(len(self.pesos)-1, -1, -1):
            # actualizar valor de pesos y sesgos
            self.pesos[i] -= learning_rate * np.dot(self.activaciones[i].T, delta)
            self.sesgos[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                # calcular error capa anterior
                error = np.dot(delta, self.pesos[i].T)
                delta = error * self.derivada_sigmoide(self.activaciones[i])
    
    def entrenar(self, X, y, epocas, learning_rate=0.1):
        precisiones_entrenamiento = []
        precisiones_prueba = []
        
        for epoca in range(epocas):
            # forward propagation
            salida = self.forward(X)
            
            # backpropagation otra vez
            self.backward(X, y, learning_rate)
            
            # preciision
            predicciones = (salida > 0.5).astype(int)
            precision = np.mean(predicciones == y)
            precisiones_entrenamiento.append(precision)
            if hasattr(self, 'X_prueba') and hasattr(self, 'y_prueba'):
                salida_prueba = self.forward(self.X_prueba)
                predicciones_prueba = (salida_prueba > 0.5).astype(int)
                precision_prueba = np.mean(predicciones_prueba == self.y_prueba)
                precisiones_prueba.append(precision_prueba)
            
            print(f"Época {epoca+1}/{epocas} - Precisión entrenamiento: {precision:.4f}")
            if hasattr(self, 'X_prueba'):
                print(f"Precision prueba: {precision_prueba:.4f}")
        
        return precisiones_entrenamiento, precisiones_prueba
    
    def guardar(self, archivo):
        with open(archivo, 'wb') as f:
            pickle.dump({
                'n_entrada': self.n_entrada,
                'n_salida': self.n_salida,
                'n_capas_ocultas': self.n_capas_ocultas,
                'n_neuronas_ocultas': self.n_neuronas_ocultas,
                'pesos': self.pesos,
                'sesgos': self.sesgos
            }, f)
    
    @classmethod
    def cargar(cls, archivo):
        with open(archivo, 'rb') as f:
            datos = pickle.load(f)
            red = cls(
                datos['n_entrada'],
                datos['n_salida'],
                datos['n_capas_ocultas'],
                datos['n_neuronas_ocultas']
            )
            red.pesos = datos['pesos']
            red.sesgos = datos['sesgos']
            return red 

class Interfaz:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Multicapa - Aida Cardenas")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#E6F3FF')
        self.style.configure('TLabel', background='#E6F3FF', foreground='#003366', font=('Consolas', 10))
        self.style.configure('TButton', background='#4D94FF', foreground='#003366', font=('Consolas', 10))
        self.style.configure('TLabelframe', background='#E6F3FF', foreground='#003366', font=('Consolas', 10))
        self.style.configure('TLabelframe.Label', background='#E6F3FF', foreground='#003366', font=('Consolas', 10))
        
        # variables
        self.red = None
        self.X_entrenamiento = None
        self.y_entrenamiento = None
        self.X_prueba = None
        self.y_prueba = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # opciones 1
        opciones_frame = ttk.LabelFrame(main_frame, text="Opciones iniciales", padding="10")
        opciones_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(opciones_frame, text="Crear Nueva Red", command=self.crear_nueva_red).pack(side=tk.LEFT, padx=5)
        ttk.Button(opciones_frame, text="Cargar Red Existente", command=self.cargar_red).pack(side=tk.LEFT, padx=5)
        
        # config
        self.config_frame = ttk.LabelFrame(main_frame, text="Configuracion", padding="10")
        self.config_frame.pack(fill=tk.X, pady=5)
        
        # entrenamiento
        self.entrenamiento_frame = ttk.LabelFrame(main_frame, text="Entrenamiento", padding="10")
        self.entrenamiento_frame.pack(fill=tk.X, pady=5)
        
        # prueba
        self.prueba_frame = ttk.LabelFrame(main_frame, text="Prueba", padding="10")
        self.prueba_frame.pack(fill=tk.X, pady=5)
        
        # resultados
        self.resultados_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        self.resultados_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # resultados
        self.resultados_text = ScrolledText(self.resultados_frame, wrap=tk.WORD, height=10, bg='#E6F3FF', fg='#003366', font=('Consolas', 10))
        self.resultados_text.pack(fill=tk.BOTH, expand=True)
        
        # grafico, lo que hace uno por un punto extra
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.resultados_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def crear_nueva_red(self):
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        # entradas
        ttk.Label(self.config_frame, text="Num. neuronas de entrada:").pack()
        self.n_entrada = ttk.Entry(self.config_frame)
        self.n_entrada.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.config_frame, text="Num. neuronas de salida:").pack()
        self.n_salida = ttk.Entry(self.config_frame)
        self.n_salida.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.config_frame, text="Num. capas ocultas:").pack()
        self.n_capas_ocultas = ttk.Entry(self.config_frame)
        self.n_capas_ocultas.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.config_frame, text="Num. neuronas por capa oculta:").pack()
        self.n_neuronas_ocultas = ttk.Entry(self.config_frame)
        self.n_neuronas_ocultas.pack(fill=tk.X, pady=2)
        
        ttk.Button(self.config_frame, text="Crear Red", command=self.inicializar_red).pack(pady=10)
    
    def inicializar_red(self):
        try:
            n_entrada = int(self.n_entrada.get())
            n_salida = int(self.n_salida.get())
            n_capas_ocultas = int(self.n_capas_ocultas.get())
            n_neuronas_ocultas = int(self.n_neuronas_ocultas.get())
            
            self.red = PerceptronMulticapa(n_entrada, n_salida, n_capas_ocultas, n_neuronas_ocultas)
            self.mostrar_opciones_entrenamiento()
            
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa valores numericos validos")
    
    def mostrar_opciones_entrenamiento(self):
        # entrenamiento
        for widget in self.entrenamiento_frame.winfo_children():
            widget.destroy()
        
        ttk.Button(self.entrenamiento_frame, text="Cargar datos de entrenamiento",
                  command=self.cargar_datos_entrenamiento).pack(fill=tk.X, pady=5)
        
        ttk.Button(self.entrenamiento_frame, text="Cargar datos de prueba",
                  command=self.cargar_datos_prueba).pack(fill=tk.X, pady=5)
        
        ttk.Label(self.entrenamiento_frame, text="Numero de epocas:").pack()
        self.epocas = ttk.Entry(self.entrenamiento_frame)
        self.epocas.pack(fill=tk.X, pady=2)
        
        ttk.Button(self.entrenamiento_frame, text="Entrenar red",
                  command=self.entrenar_red).pack(fill=tk.X, pady=5)
    
    def cargar_datos_entrenamiento(self):
        archivo_entradas = filedialog.askopenfilename(
            title="Seleccionar archivo - entradas de entrenamiento",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        archivo_salidas = filedialog.askopenfilename(
            title="Seleccionar archivo - salidas de entrenamiento",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        if archivo_entradas and archivo_salidas:
            try:
                self.X_entrenamiento = np.loadtxt(archivo_entradas, delimiter=',')
                self.y_entrenamiento = np.loadtxt(archivo_salidas, delimiter=',')
                messagebox.showinfo("Información", "Datos de entrenamiento cargados ")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar datos {str(e)}")
    
    def cargar_datos_prueba(self):
        archivo_entradas = filedialog.askopenfilename(
            title="Seleccionar archivo - entradas de prueba",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        archivo_salidas = filedialog.askopenfilename(
            title="Seleccionar archivo - salidas de prueba",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        if archivo_entradas and archivo_salidas:
            try:
                self.X_prueba = np.loadtxt(archivo_entradas, delimiter=',')
                self.y_prueba = np.loadtxt(archivo_salidas, delimiter=',')
                self.red.X_prueba = self.X_prueba
                self.red.y_prueba = self.y_prueba
                messagebox.showinfo("Información", "Datos de prueba cargados ")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
    
    def entrenar_red(self):
        if self.red is None:
            messagebox.showerror("Error", "Debe crear o cargar una red")
            return
        
        if self.X_entrenamiento is None or self.y_entrenamiento is None:
            messagebox.showerror("Error", "Debe cargar datos de entrenamiento")
            return
        
        try:
            epocas = int(self.epocas.get())
            precisiones_entrenamiento, precisiones_prueba = self.red.entrenar(
                self.X_entrenamiento, self.y_entrenamiento, epocas
            )
            
            # grafico de preciosion
            self.ax.clear()
            self.ax.plot(range(1, epocas + 1), precisiones_entrenamiento, label='Entrenamiento')
            if precisiones_prueba:
                self.ax.plot(range(1, epocas + 1), precisiones_prueba, label='Prueba')
            self.ax.set_xlabel('Epoca')
            self.ax.set_ylabel('Precision')
            self.ax.set_title('Precision por Epoca')
            self.ax.legend()
            self.canvas.draw()
            
            # resultados 
            self.resultados_text.delete(1.0, tk.END)
            self.resultados_text.insert(tk.END, f"Precision final de entrenamiento: {precisiones_entrenamiento[-1]:.4f}\n")
            if precisiones_prueba:
                self.resultados_text.insert(tk.END, f"Precision final de prueba: {precisiones_prueba[-1]:.4f}\n")
            
            # guardar
            self.mostrar_opciones_post_entrenamiento()
            
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese un numero válido de epocas")
    
    def mostrar_opciones_post_entrenamiento(self):
        # limpiar prueba
        for widget in self.prueba_frame.winfo_children():
            widget.destroy()
        
        ttk.Button(self.prueba_frame, text="Guardar red",
                  command=self.guardar_red).pack(fill=tk.X, pady=5)
        
        ttk.Button(self.prueba_frame, text="Probar con vector manual",
                  command=self.probar_vector_manual).pack(fill=tk.X, pady=5)
        
        ttk.Button(self.prueba_frame, text="Probar con archivo",
                  command=self.probar_archivo).pack(fill=tk.X, pady=5)
        
        ttk.Button(self.prueba_frame, text="Seguir entrenando",
                  command=self.mostrar_opciones_entrenamiento).pack(fill=tk.X, pady=5)
    
    def guardar_red(self):
        if self.red is None:
            messagebox.showerror("Error", "Escoja una red para guardar")
            return
        
        archivo = filedialog.asksaveasfilename(
            title="Guardar red",
            filetypes=[("Archivos de red", "*.pkl")],
            defaultextension=".pkl"
        )
        
        if archivo:
            try:
                self.red.guardar(archivo)
                messagebox.showinfo("Información", "Red guardada")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar la red: {str(e)}")
    
    def cargar_red(self):
        archivo = filedialog.askopenfilename(
            title="Cargar red",
            filetypes=[("Archivos de red", "*.pkl")]
        )
        
        if archivo:
            try:
                self.red = PerceptronMulticapa.cargar(archivo)
                messagebox.showinfo("Información", "Red cargada")
                self.mostrar_opciones_entrenamiento()
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar la red: {str(e)}")
    
    def probar_vector_manual(self):
        if self.red is None:
            messagebox.showerror("Error", "No hay red para probar")
            return
        
        # entrada manual de vectores
        ventana = tk.Toplevel(self.root)
        ventana.title("Probar vector")
        ventana.geometry("400x300")
        
        frame = ttk.Frame(ventana, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        entradas = []
        for i in range(self.red.n_entrada):
            ttk.Label(frame, text=f"Entrada {i+1}:").pack()
            entrada = ttk.Entry(frame)
            entrada.pack(fill=tk.X, pady=2)
            entradas.append(entrada)
        
        def calcular():
            try:
                vector = np.array([float(entrada.get()) for entrada in entradas])
                if len(vector) != self.red.n_entrada:
                    raise ValueError("Numero incorrecto de entradas")
                
                salida = self.red.forward(vector.reshape(1, -1))
                prediccion = (salida > 0.5).astype(int)
                
                self.resultados_text.delete(1.0, tk.END)
                self.resultados_text.insert(tk.END, f"Vector de entrada: {vector}\n")
                self.resultados_text.insert(tk.END, f"Salida: {salida}\n")
                self.resultados_text.insert(tk.END, f"Prediccion: {prediccion}\n")
                
            except ValueError as e:
                messagebox.showerror("Error", str(e))
        
        ttk.Button(frame, text="Calcular", command=calcular).pack(pady=10)
    
    def probar_archivo(self):
        if self.red is None:
            messagebox.showerror("Error", "No hay red para probar")
            return
        
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de prueba",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        if archivo:
            try:
                X = np.loadtxt(archivo, delimiter=',')
                if X.shape[1] != self.red.n_entrada:
                    raise ValueError("Numero incorrecto de entradas en el archivo")
                
                salidas = self.red.forward(X)
                predicciones = (salidas > 0.5).astype(int)
                
                self.resultados_text.delete(1.0, tk.END)
                self.resultados_text.insert(tk.END, "Resultados de la prueba:\n\n")
                for i, (entrada, salida, prediccion) in enumerate(zip(X, salidas, predicciones)):
                    self.resultados_text.insert(tk.END, f"Vector {i+1}:\n")
                    self.resultados_text.insert(tk.END, f"Entrada: {entrada}\n")
                    self.resultados_text.insert(tk.END, f"Salida: {salida}\n")
                    self.resultados_text.insert(tk.END, f"Prediccion: {prediccion}\n")
                    self.resultados_text.insert(tk.END, "-" * 40 + "\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar archivo: {str(e)}")

def main():
    root = tk.Tk()
    app = Interfaz(root)
    root.mainloop()

if __name__ == "__main__":
    main() 