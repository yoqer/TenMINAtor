# Guía de Inicio Rápido - MiniTorch Framework

Esta guía te ayudará a entrenar tu primer modelo en **menos de 5 minutos**.

---

## Paso 1: Instalación

```bash
# Extraer archivos
unzip minitorch_lite.zip
unzip minitorch_framework.zip

# Añadir al path de Python (en tu script)
import sys
sys.path.insert(0, '/ruta/a/minitorch_lite')
sys.path.insert(0, '/ruta/a/minitorch_framework')
```

---

## Paso 2: Entrenar Modelo Completo

```bash
cd minitorch_framework
python examples/train_complete_model.py --epochs 10 --generate
```

**Salida esperada:**

```
======================================================================
MiniTorch Framework - Pipeline de Entrenamiento Completo
======================================================================

[1/5] Inicializando Transformer RLM...
[TransformerRLM] Inicializado con:
  - Vocabulario: 50000
  - Dimensión del modelo: 512
  - Cabezas de atención: 8
  - Capas: 6
  - Neuronas Feed-Forward: 2048
  - Parámetros totales: ~XX,XXX,XXX

[2/5] Inicializando optimizador Adam...
[3/5] Inicializando función de pérdida...
[4/5] Inicializando controlador de entrenamiento...
[5/5] Inicializando vLLM para inferencia rápida...

======================================================================
Inicialización completada
======================================================================

======================================================================
Iniciando Entrenamiento
======================================================================

[Datos] Generando 1000 muestras sintéticas...
[Datos] Forma de inputs: (1000, 128)
[Datos] Forma de targets: (1000, 128)

--- Época 1/10 ---
[Iter 10/69] Loss: 0.234567 | Best: 0.234567 | Sin mejora: 0/12
[Iter 20/69] Loss: 0.198765 | Best: 0.198765 | Sin mejora: 0/12
...

Pérdida promedio de la época: 0.187654

--- Época 2/10 ---
...

======================================================================
Entrenamiento Completado
======================================================================

[MiniTorch Lite] Checkpoint guardado: ./checkpoints/model_final.pkl
  - Iteración: 310
  - Mejor pérdida: 0.123456
  - Razón de parada: Máximo de iteraciones alcanzado (69)

[Generación] Prompt: 'Hello, this is a test'
[Generación] Texto generado: '...'

[Exportación] Exportando modelo a formato numpy...
[Exportación] Modelo exportado: ./checkpoints/model_export.numpy

======================================================================
Pipeline Completo Finalizado
======================================================================
```

---

## Paso 3: Usar el Modelo Entrenado

```python
import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/minitorch_framework')

from examples.train_complete_model import CompleteTrainingPipeline
import json

# Cargar configuración
with open('config.json', 'r') as f:
    config = json.load(f)

# Crear pipeline
pipeline = CompleteTrainingPipeline(config)

# Cargar checkpoint
pipeline.load_checkpoint('./checkpoints/model_final.pkl')

# Generar texto
generated = pipeline.generate_text("Hello world", max_length=100)
print(f"Texto generado: {generated}")
```

---

## Paso 4: Personalizar el Entrenamiento

Edita `config.json`:

```json
{
  "model": {
    "vocab_size": 10000,    // Reducir para pruebas rápidas
    "d_model": 256,         // Reducir para menos memoria
    "num_heads": 4,
    "num_layers": 3,
    "d_ff": 1024
  },
  "training": {
    "batch_size": 16,       // Reducir si hay problemas de memoria
    "learning_rate": 0.001,
    "max_iterations": 50,
    "early_stop_patience": 10
  }
}
```

Luego ejecuta:

```bash
python examples/train_complete_model.py --config config.json --epochs 5
```

---

## Paso 5: Entrenar con Reinforcement Learning

```python
import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/minitorch_framework')

from rl.reinforcement_learning import PPOAgent
import numpy as np

# Crear agente PPO
agent = PPOAgent(
    state_dim=84,
    action_dim=4,
    learning_rate=0.0003
)

# Simular entorno (ejemplo)
class SimpleEnv:
    def reset(self):
        return np.random.randn(84)
    
    def step(self, action):
        next_state = np.random.randn(84)
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        return next_state, reward, done, {}

env = SimpleEnv()

# Entrenar
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_reward(reward, done)
        state = next_state
        total_reward += reward
    
    loss = agent.train_step()
    print(f"Episodio {episode}: Recompensa = {total_reward:.2f}, Pérdida = {loss:.4f}")

# Guardar modelo
agent.save('ppo_model.pkl')
print("Modelo guardado: ppo_model.pkl")
```

---

## Paso 6: Usar vLLM para Inferencia Rápida

```python
import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/minitorch_framework')

from models.transformer_rlm import TransformerRLM
from vllm.relational_vllm import vLLM
import numpy as np

# Crear modelo
model = TransformerRLM(vocab_size=10000, d_model=256, num_layers=3)

# Crear vLLM
vllm_engine = vLLM(model, max_batch_size=8)

# Generar en batch
prompts = [
    np.array([1, 2, 3, 4, 5]),
    np.array([6, 7, 8]),
    np.array([9, 10, 11, 12])
]

results = vllm_engine.batch_generate(
    prompts,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

for i, result in enumerate(results):
    print(f"Prompt {i}: {result}")
```

---

## Opciones de Control de Entrenamiento

### Opción 1: Early Stopping (NEW)

Detiene el entrenamiento tras **12 iteraciones** sin mejora:

```python
from minitorch_lite import create_training_controller

controller = create_training_controller(
    early_stop=True,
    early_stop_patience=12
)
```

### Opción 2: Límite de Tokens

Detiene tras procesar **10 tokens** con máximo **12 iteraciones**:

```python
controller = create_training_controller(
    max_tokens=10,
    token_limit_iterations=12
)
```

### Opción 3: Máximo de Iteraciones (Por Defecto)

Detiene tras **69 iteraciones**:

```python
controller = create_training_controller(
    max_iterations=69
)
```

---

## Troubleshooting Rápido

### Problema: ImportError

```python
# Asegúrate de añadir las rutas correctas
import sys
sys.path.insert(0, '/ruta/correcta/a/minitorch_lite')
sys.path.insert(0, '/ruta/correcta/a/minitorch_framework')
```

### Problema: Out of Memory

Reduce el tamaño del modelo en `config.json`:

```json
{
  "model": {
    "d_model": 128,
    "num_layers": 2,
    "d_ff": 512
  },
  "training": {
    "batch_size": 8
  }
}
```

### Problema: Entrenamiento muy lento

Usa backend acelerado:

```python
from minitorch_lite import set_backend
set_backend('numba')  # Requiere: pip install numba
```

---

## Próximos Pasos

1. **Lee el Manual Completo**: `docs/MANUAL_DE_USO.md`
2. **Explora los Ejemplos**: `examples/train_complete_model.py`
3. **Personaliza tu Modelo**: Edita `config.json`
4. **Entrena en tus Datos**: Reemplaza `generate_synthetic_data()` con tus datos reales

---

## Recursos Adicionales

- **README Principal**: `README.md`
- **Manual Completo**: `docs/MANUAL_DE_USO.md`
- **Código Fuente**: Todos los módulos en `models/`, `rl/`, `vllm/`
- **Configuración**: `config.json`

---

**¡Feliz Entrenamiento!** 🚀
