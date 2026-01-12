_**```markdown
# Manual de Uso de TenMiNaTor

Este manual proporciona una guía detallada para el uso del framework TenMiNaTor, desde la instalación hasta la implementación de modelos avanzados.

## 1. Introducción

TenMiNaTor es un framework de Deep Learning diseñado para la flexibilidad y la experimentación. Su arquitectura modular permite la integración de conceptos de vanguardia como Nested Learning, aprendizaje relacional y steering de modelos.

## 2. Instalación

```bash
# Clonar el repositorio 
  git clone https://github.com/yoqer/TenMiNaTor.git
  cd TenMiNaTor

# Instalar en modo editable
pip install -e .
```

## 3. Módulo de Datos Avanzado

El módulo `data.py` proporciona herramientas para la carga y el preprocesamiento de datos desde diversas fuentes.

### 3.1. Carga desde Bases de Datos SQL

```python
from minitorch_lite.data import from_sql

# Conexión a MariaDB
mariadb_conn_str = "mysql+pymysql://user:password@host/db"
df_mariadb = from_sql("SELECT * FROM your_table", mariadb_conn_str)

# Conexión a SQLite
df_sqlite = from_sql("SELECT * FROM your_table", "sqlite:///example.db")
```

### 3.2. Carga desde Bases de Datos NoSQL

```python
from minitorch_lite.data import from_mongodb, from_firestore

# Conexión a MongoDB
df_mongo = from_mongodb({}, 'your_db', 'your_collection')

# Conexión a Firestore
df_firestore = from_firestore('your_collection', 'path/to/your/credentials.json')
```

### 3.3. Web Scraping y Construcción de Grafos

```python
from minitorch_lite.data import build_graph_from_web

# Construir un grafo de enlaces desde una URL
graph = build_graph_from_web("https://www.google.com", max_depth=1)
```

## 4. El Núcleo: Tensores y Autograd

El corazón de TenMiNaTor es su motor de autograd y la clase `Tensor`.

```python
from minitorch_lite import Tensor

# Crear tensores
x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)

# Realizar operaciones
z = x * y + x

# Calcular gradientes
z.sum().backward()

print(x.grad) # [5, 6, 7]
print(y.grad) # [1, 2, 3]
```

## 5. Construcción de Modelos con `nn.Module`

Todos los modelos en TenMiNaTor heredan de `nn.Module`.

```python
from minitorch_lite import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

## 6. Entrenamiento de Modelos

El `TrainingController` gestiona el bucle de entrenamiento.

```python
from minitorch_lite.training import TrainingController
from minitorch_lite.optim import Adam

model = MyModel()
optimizer = Adam(model.parameters(), lr=0.001)
controller = TrainingController(model, optimizer, nn.MSELoss())

# Entrenar el modelo
controller.train(train_dataloader, epochs=10)
```

## 7. Funcionalidades Avanzadas

### 7.1. Nested Learning

El Nested Learning se implementa a través de `NestedOptimizer`.

```python
from minitorch_lite.nested import NestedOptimizer

# ... (definición del modelo)

# Crear un optimizador anidado
optimizer = NestedOptimizer(model.parameters(), lr=0.01)

# ... (bucle de entrenamiento)
```

### 7.2. Steering de Modelos

El `SteeringHook` permite manipular los pesos de una capa.

```python
from minitorch_lite.steering import SteeringHook

# ... (definición del modelo)

# Crear un hook de steering
steering_hook = SteeringHook(model.linear1, steering_vector, strength=0.1)

# Aplicar el steering
steering_hook.apply()
```

## 8. Documentación de la API 

(Esta sección se complementa con la documentación detallada de cada módulo y clase).
```**_

# Puedes comenzar tu recolección especifica:
# Online o con su APP.

Click


# Aplicar API

   Multi Click
```



[<img width="1920" height="1920" alt="IMG_20251218_003049" src="https://github.com/user-attachments/assets/1e31e9a9-2885-4530-ad4c-cd71f2b5f3a6"/>](http://wAI.CaM) 



 
