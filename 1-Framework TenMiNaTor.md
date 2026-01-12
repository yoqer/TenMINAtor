_**```markdown
# TenMiNaTor: Un Framework de Deep Learning Avanzado

**TenMiNaTor** es un framework de Deep Learning de vanguardia, diseñado para la investigación y el desarrollo de modelos de inteligencia artificial de próxima generación. Inspirado en los últimos avances en el campo, TenMiNaTor integra conceptos como Nested Learning, razonamiento lógico, y control de entrenamiento avanzado en un paquete cohesivo y funcional.

## Características Principales

| Característica | Descripción |
|---|---|
| **Núcleo de Autograd Robusto** | Un motor de diferenciación automática estable con soporte para backends (NumPy, JAX, Numba, CuPy). |
| **Módulos de Datos Avanzados** | Carga de datos desde SQL (MariaDB, PostgreSQL, etc.), NoSQL (MongoDB, Firebase), web scraping, y construcción de grafos con NetworkX. |
| **Nested Learning** | Implementación de la arquitectura de aprendizaje anidado para un entrenamiento más eficiente y jerárquico. |
| **Absolute Zero Reasoner** | Un módulo para el razonamiento lógico y la generación de tareas, inspirado en los últimos avances en IA. |
| **Gradientes por Capas y Steering** | Control granular sobre el proceso de entrenamiento, permitiendo la manipulación de gradientes y el "steering" de modelos. |
| **Tokenización Flexible** | Opciones para el procesamiento de texto a nivel de carácter, palabra o byte. |
| **Sistema de Memoria Relacional** | Un sistema de memoria diferenciable para el razonamiento sobre relaciones. |
| **Control de Entrenamiento Avanzado** [10x12]" |  Un sistema de control de entrenamiento interactivo con early stopping, checkpoints, y más. |
| **Integración con Herramientas Externas** | Compatibilidad con Unsloth, LangChain, y LangGraph. |
| **vLLM y Transformer RLM** | Un motor de inferencia rápido y una arquitectura de Transformer avanzada. |



## [Instalación](https://github.com/yoqer/TenMINAtor/blob/main/0-TenMiNaTor%20-%20Deep%20Learning%20Ultraligero.md) 



```bash
pip install -e .
```

## [Uso Rápido](https://github.com/yoqer/TenMINAtor/blob/main/2-Guía%20de%20Inicio%20Rápido.md) 


```python
import numpy as np
from minitorch_lite import Tensor
from minitorch_framework.models.transformer_rlm import TransformerRLM

# Crear un modelo
model = TransformerRLM(vocab_size=1000, d_model=128, nhead=4, num_layers=2, d_ff=256)

# Crear datos de entrada
input_data = np.random.randint(0, 1000, size=(2, 10))

# Realizar un forward pass
output = model(input_data)

print(output.shape)
```

## Documentación

Para una guía más detallada, consulta el [MANUAL_DE_USO](https://github.com/yoqer/TenMINAtor/blob/main/3-Manual%20de%20Uso%20TenMiNaTor.md) 
```**_
