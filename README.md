# IA Project

![AI Icon](https://img.icons8.com/ios-filled/50/000000/artificial-intelligence.png)

## Descripción

Este proyecto es una libreria del alto nivel que facilita el entrenamiento de modelos de inteligencia artificial.
La intención es facilitar una interfaz amigable al usuario, sin la necesidad de profundizar en los detalles.

## Características

- **Entrenamiento de modelos:** Soporte para frameworks como TensorFlow y Sklearn.  
- **Visualización:** Por medio de la matriz de confusión se analiza el rendimiento del modelo.  
- **Automatización:** Scripts para preprocesamiento de datos y optimización de hiperparámetros.  
- **Compatibilidad:** Funciona en Windows, macOS y Linux.  

## Requisitos

- Python 3.8 o superior  
- Bibliotecas necesarias (ver `requirements.txt`)  
- GPU compatible (opcional para entrenamiento acelerado)  

## Instalación

1. Clona el repositorio:  

    ```bash
    git clone https://github.com/AI-unir-group/IA-PROJECT.git
    cd ia-project
    ```

2. Instala las dependencias:  

    ```bash
    pip install -r requirements.txt
    ```

3. Activa el ambiente:

    ```bash
    env/Scripts/Activate

    ```

4. Ejecuta el proyecto:  

    ```bash
    python App.py
    
    ```

## Variables de entrenamiento para los modelos, estas pueden ser opcionales

```bash 
Modelo tensorflow

trainDic = {
    "metric": "rmse",
    "cv": 15,
    "jobs": -1,
    "epoch": 10,
    "batch_size": 5,
    "epsilon": 0.5,
    "penalty": "l2",
    "verbose": 1,
    "alpha": 0.0002,
    "n_iter_stop": 10,
    "random_state": 50,
    "early_stopping": True,
    "shuffle": True
}

Modelo sgd

trainDic = {
    "jobs": -1,
    "epoch": 500,
    "tol": 1e-3,
    "loss": "squared_error",
    "alpha": 0.0001,
    "l1": 0.15,
    "shuffle": True,
    "lr": "invscaling",

}

Modelo tree

trainDic = {
    "jobs": -1,
    "epoch": 500,
    "tol": 1e-3,
    "loss": "squared_error",
    "alpha": 0.0001,
    "l1": 0.15,
    "shuffle": True,
    "lr": "invscaling",

}

```

## Dataset

El dataset es cargado, procesado y dividio en partes, para entrenamiento y para test.
Otra parte se guarda solo para la prueba de estimación.
El dataset debe cumplir estricatemte con las columnas y tipos de datos del modelo.
Se propone como ejemplo el dataset que se encuentra en el directorio dataset

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).

---

![Python Icon](https://img.icons8.com/ios-filled/50/000000/python.png)  