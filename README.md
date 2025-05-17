# Perceptrón Multicapa - Aida Cardenas

Este proyecto implementa un Perceptrón Multicapa con interfaz gráfica que permite crear, entrenar, guardar y probar redes neuronales.

## Requisitos

Antes de ejecutar el programa, asegúrate de tener instaladas las siguientes dependencias:

```
numpy>=1.20.0
tk>=0.1.0
matplotlib>=3.5.0
```

Puedes instalarlas ejecutando:

```
pip install -r requerimientos.txt
```

## Ejecución del programa

Para iniciar el programa, ejecuta el archivo principal:

```
python tarea2_aidacardenas.py
```

## Guía de uso

### 1. Crear una nueva red neuronal

1. Al iniciar el programa, haz clic en el botón "Crear Nueva Red".
2. Completa los siguientes campos:
   - Número de neuronas de entrada: cantidad de características de entrada.
   - Número de neuronas de salida: cantidad de salidas deseadas.
   - Número de capas ocultas: cantidad de capas intermedias en la red.
   - Número de neuronas por capa oculta: cantidad de neuronas en cada capa oculta.
3. Haz clic en "Crear Red" para inicializar la red con los parámetros especificados.

### 2. Cargar datos de entrenamiento y prueba

1. Después de crear la red, aparecerán opciones para cargar datos:
   - "Cargar Datos de Entrenamiento": selecciona los archivos de entradas y salidas de entrenamiento.
     - Formato esperado: archivos de texto (.txt) con valores separados por comas.
     - Ejemplo (entradas_entrenamiento.txt): `0,0`, `0,1`, `1,0`, `1,1`
     - Ejemplo (salidas_entrenamiento.txt): `0`, `1`, `1`, `0`
   - "Cargar Datos de Prueba": selecciona los archivos de entradas y salidas de prueba.

### 3. Entrenar la red

1. Ingresa el número de épocas para el entrenamiento.
2. Haz clic en "Entrenar Red".
3. El proceso de entrenamiento comenzará y podrás observar:
   - El progreso en la consola
   - Un gráfico que muestra la precisión de entrenamiento y prueba a lo largo de las épocas

### 4. Guardar y cargar una red

Después de entrenar la red, puedes:

- "Guardar Red": guarda la red entrenada para su uso posterior.
- "Cargar Red Existente" (desde la pantalla inicial): carga una red previamente guardada.

### 5. Probar la red

Una vez entrenada la red, puedes probarla de dos formas:

- "Probar Vector Manual": ingresa valores manualmente para cada neurona de entrada.
- "Probar con Archivo": carga un archivo de vectores de entrada para evaluar múltiples casos.

## Estructura de archivos

- `tarea2_aidacardenas.py`: programa principal con la implementación del Perceptrón Multicapa.
- `entradas_entrenamiento.txt`: datos de entrada para entrenar la red.
- `salidas_entrenamiento.txt`: datos de salida esperados para el entrenamiento.
- `entradas_prueba.txt`: datos de entrada para probar la red.
- `salidas_prueba.txt`: datos de salida esperados para las pruebas.
- `requerimientos.txt`: lista de dependencias del proyecto.

## Ejemplo de uso

El programa incluye datos de ejemplo para implementar una puerta lógica XOR:

| Entrada 1 | Entrada 2 | Salida (XOR) |
|-----------|-----------|--------------|
| 0         | 0         | 0            |
| 0         | 1         | 1            |
| 1         | 0         | 1            |
| 1         | 1         | 0            |

Este es un problema clásico que no puede resolverse con un perceptrón simple, pero sí con un perceptrón multicapa.

## Consejos para obtener buenos resultados

- Para problemas simples como XOR, una configuración de 2 entradas, 1 salida, 1 capa oculta con 4 neuronas suele ser suficiente.
- El número de épocas recomendado está entre 1000-5000 para problemas simples.
- Si la red no aprende correctamente, intenta aumentar el número de neuronas en las capas ocultas o añadir más capas ocultas. 