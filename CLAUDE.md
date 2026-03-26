# CLAUDE.md — Módulo RL de Orientación de Fibras Musculares (HU5)

## Contexto del proyecto

Este módulo es parte de un sistema mayor de análisis histológico automatizado de
fibras musculares (tesis de maestría UNIR). El sistema completo usa Mask R-CNN para
segmentación de instancias (HU2/HU3); este módulo (HU5) recibe las máscaras de salida
de Mask R-CNN y estima el ángulo de orientación dominante de cada fibra.

**Problema que resuelve:** Los métodos geométricos clásicos (ajuste de elipse mínima)
tienen error angular de ~12–18° en fibras irregulares o dañadas. Este módulo debe
alcanzar error < 10° (prototipo) y < 5° (objetivo final) usando RL entrenado con
imágenes sintéticas de Ground Truth absoluto.

**Innovación clave:** El entrenamiento es completamente sintético — no requiere
anotaciones angulares en imágenes reales, que son prácticamente inexistentes en
datasets públicos.

---

## Stack tecnológico

- **Python:** 3.9+
- **RL framework:** `stable-baselines3` con `gymnasium`
- **Algoritmo RL:** PPO (Proximal Policy Optimization) con política CNN
- **Visión:** `opencv-python`, `scikit-image`, `numpy`
- **Deep learning backend:** PyTorch (`torch`, `torchvision`)
- **Visualización:** `matplotlib`
- **Tests:** `pytest`

---

## Estructura de archivos

```
rl_orientation/
├── CLAUDE.md                        ← este archivo
├── requirements.txt
├── main.py                          ← punto de entrada: train / evaluate / infer
│
├── env/
│   ├── __init__.py
│   ├── fiber_env.py                 ← entorno Gymnasium personalizado (CORE)
│   └── synthetic_generator.py      ← generador de imágenes sintéticas (CORE)
│
├── agent/
│   ├── __init__.py
│   ├── train.py                     ← loop de entrenamiento PPO
│   └── evaluate.py                  ← evaluación: MAE angular en n=100 imágenes
│
├── utils/
│   ├── __init__.py
│   ├── reward.py                    ← función de recompensa SSIM/MSE
│   └── histogram.py                 ← genera y exporta histograma angular
│
└── tests/
    ├── test_generator.py
    ├── test_env.py
    └── test_reward.py
```

---

## Orden de implementación

Implementa en este orden exacto. No avances al siguiente paso sin que el anterior pase sus tests.

### Paso 1 — `requirements.txt`

```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-image>=0.21.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
pytest>=7.4.0
```

### Paso 2 — `env/synthetic_generator.py`

Genera imágenes sintéticas de fibras musculares con orientación angular controlada.

**Especificaciones:**
- Tamaño de imagen de salida: `128 × 128` px, escala de grises (1 canal)
- Simula fibras como elipses paralelas con eje mayor alineado al ángulo `theta`
- Parámetros configurables: `n_fibers` (5–20), `fiber_width` (4–8 px),
  `fiber_length` (30–60 px), `noise_std` (0–15)
- Ángulo `theta` en rango `[0°, 180°)` — orientación de fibra, no dirección
- Añade ruido gaussiano con `cv2.randn` para simular variabilidad de tinción
- Devuelve: `(image: np.ndarray uint8, theta: float)`

**Función pública principal:**
```python
def generate_fiber_image(theta: float, n_fibers: int = 12,
                         noise_std: float = 8.0,
                         size: int = 128) -> np.ndarray:
    """
    Genera imagen sintética de fibras con orientación theta (grados).
    Retorna array uint8 de shape (size, size).
    """
```

**Test mínimo:** `test_generator.py` verifica que:
- La imagen generada tiene shape `(128, 128)` y dtype `uint8`
- Diferentes valores de `theta` producen imágenes distintas
- La función no lanza excepciones para theta en [0, 180)

---

### Paso 3 — `utils/reward.py`

Función de recompensa basada en similitud estructural.

**Especificaciones:**
- Usa `skimage.metrics.structural_similarity` (SSIM) como métrica principal
- Rango de recompensa normalizado: `[-1.0, 1.0]`
- Formula: `reward = 2 * SSIM(img_real, img_synthetic) - 1`
  (SSIM nativo está en [0,1]; así el agente recibe recompensa negativa al inicio)
- Añade penalización por pasos: `-0.01` por cada step (incentiva converger rápido)

**Función pública:**
```python
def compute_reward(img_real: np.ndarray,
                   img_synthetic: np.ndarray,
                   step_penalty: float = 0.01) -> float:
    """
    Calcula recompensa como función de similitud estructural.
    Ambas imágenes deben tener el mismo shape.
    """
```

**Test mínimo:** Verifica que imágenes idénticas dan recompensa ≈ 1.0 e imágenes
ortogonales (theta vs theta+90°) dan recompensa < 0.

---

### Paso 4 — `env/fiber_env.py`

Entorno Gymnasium. Es el componente más crítico del módulo.

**Especificaciones del entorno:**

```python
class FiberOrientationEnv(gymnasium.Env):
    """
    Entorno de orientación de fibras musculares.

    Observation: imagen de fibras, shape (1, 128, 128), dtype float32, rango [0,1]
    Action:      continua, shape (1,), rango [-1, 1]
                 se mapea a delta_theta: acción * MAX_DELTA_DEG
                 MAX_DELTA_DEG = 10.0  (máximo cambio por step)
    Reward:      compute_reward(img_objetivo, img_estimada_actual)
    Episode:     termina si |error_angular| < 5° O si steps > MAX_STEPS (200)
    """
```

**Lógica del episodio:**
1. `reset()`: sortea un `theta_objetivo` aleatorio en [0, 180). Genera
   `img_objetivo = generate_fiber_image(theta_objetivo)`. Inicializa
   `theta_estimado = 90.0` (neutro). Devuelve observación = `img_objetivo`
   normalizada a float32.
2. `step(action)`: actualiza `theta_estimado += action * MAX_DELTA_DEG`.
   Clampea a [0, 180). Genera `img_estimada = generate_fiber_image(theta_estimado)`.
   Calcula `reward`. Calcula `error = angular_distance(theta_estimado, theta_objetivo)`.
   `terminated = error < 5.0`. Devuelve `(obs, reward, terminated, truncated, info)`.
   `info` debe incluir `{"error_deg": error, "theta_target": theta_objetivo,
   "theta_estimated": theta_estimado}`.
3. `angular_distance(a, b)`: distancia angular en [0, 90°] — las fibras son
   simétricas (0° == 180°), así que `min(|a-b|, 180-|a-b|)`.

**Render:** implementa modo `"human"` con matplotlib mostrando imagen objetivo
e imagen estimada lado a lado con los ángulos.

**Test mínimo:** `test_env.py` verifica que:
- El entorno pasa `gymnasium.utils.env_checker.check_env(env)`
- Un episodio completo (reset → steps → terminated) no lanza excepciones
- `info["error_deg"]` disminuye en promedio en los últimos 10 steps de un
  agente que siempre aplica acción `sign(theta_target - theta_estimated)`

---

### Paso 5 — `agent/train.py`

Loop de entrenamiento con stable-baselines3.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def train(total_timesteps: int = 500_000,
          save_path: str = "models/ppo_fiber_orientation",
          log_dir: str = "logs/"):
    """
    Entrena agente PPO en FiberOrientationEnv.
    Guarda checkpoints cada 50_000 steps.
    Usa EvalCallback en entorno de validación separado.
    """
```

**Configuración PPO recomendada:**
```python
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs={"normalize_images": True}
)
```

**Wrapping obligatorio del entorno:**
```python
env = DummyVecEnv([lambda: FiberOrientationEnv()])
env = VecTransposeImage(env)  # PPO CnnPolicy espera (C, H, W)
```

**Criterio de parada:** si el error angular medio en el conjunto de eval baja de 8°
durante 3 evaluaciones consecutivas, detener el entrenamiento anticipadamente.

---

### Paso 6 — `agent/evaluate.py`

Evaluación formal sobre n=100 imágenes sintéticas con ángulos conocidos.

```python
def evaluate(model_path: str,
             n_images: int = 100,
             output_csv: str = "results/evaluation.csv") -> dict:
    """
    Carga modelo entrenado, genera n_images sintéticas con ángulos uniformemente
    distribuidos en [0, 180), ejecuta inferencia, calcula:
      - MAE angular (Mean Absolute Error en grados)
      - porcentaje de imágenes con error < 5°
      - porcentaje de imágenes con error < 10°
    Guarda CSV con columnas: [theta_true, theta_predicted, error_deg]
    Retorna dict con métricas.
    """
```

**Criterios de aceptación (HU5):**
- `MAE < 10°` → prototipo válido (hipótesis H2 cumplida)
- `MAE < 5°`  → objetivo de producción
- Si `MAE > 15°` → activar Plan B (ver sección Plan B)

---

### Paso 7 — `utils/histogram.py`

```python
def plot_angular_histogram(angles: list[float],
                           output_path: str = "results/angular_histogram.png",
                           title: str = "Distribución Angular de Fibras") -> None:
    """
    Genera histograma polar de distribución angular.
    - Bins: 18 (cada 10°) en rango [0°, 180°)
    - Usa proyección polar con matplotlib
    - Guarda imagen PNG y CSV con frecuencias por bin
    - La simetría 0°==180° se maneja duplicando el histograma en [180°, 360°)
      para visualización polar completa
    """
```

---

### Paso 8 — `main.py`

CLI simple con tres modos:

```bash
python main.py train   --timesteps 500000 --save models/ppo_v1
python main.py eval    --model models/ppo_v1 --n 100 --output results/eval.csv
python main.py infer   --model models/ppo_v1 --image ruta/imagen.png
```

El modo `infer` recibe una imagen real o máscara binaria, la redimensiona a
128×128, ejecuta el agente, y retorna el ángulo estimado + imagen con vector
de orientación superpuesto.

---

## Plan B (activar si MAE > 15° al final del Sprint 4, semana 1)

Si el agente RL no converge, implementar fallback geométrico en `utils/ellipse_fallback.py`:

```python
def estimate_orientation_ellipse(mask: np.ndarray) -> float:
    """
    Ajuste de elipse mínima a la máscara de instancia.
    Usa cv2.fitEllipse sobre los contornos de la máscara binaria.
    Retorna ángulo del eje mayor en [0°, 180°).
    Error esperado: ~12° en fibras regulares, hasta 18° en irregulares.
    """
```

El módulo principal debe poder switchear entre RL y fallback con un flag
`--method {rl, ellipse}`.

---

## Convenciones de código

- **Docstrings:** Google style en todas las funciones públicas
- **Type hints:** obligatorios en todas las firmas
- **Logging:** usar `logging` estándar de Python, no `print`
- **Configuración:** parámetros hardcoded solo en `main.py`; el resto
  recibe parámetros por argumento
- **Reproducibilidad:** fijar semillas con `np.random.seed(42)`,
  `torch.manual_seed(42)` al inicio de train y eval

---

## Criterios de Done (DoD) del módulo

- [ ] `pytest tests/` pasa al 100% sin warnings
- [ ] `evaluate.py` reporta MAE < 10° en n=100 imágenes sintéticas
- [ ] El histograma angular se genera correctamente y es exportable como PNG + CSV
- [ ] `main.py infer` funciona con una imagen PNG arbitraria de 128×128
- [ ] Código sin errores de linting (`flake8` o `ruff`)
- [ ] `requirements.txt` instalable en entorno limpio con `pip install -r requirements.txt`

---

## Integración con el sistema mayor (para referencia)

Este módulo se conecta con Mask R-CNN de la siguiente forma:

```
[Imagen histológica]
       ↓
[Preprocesamiento HU1]
       ↓
[Mask R-CNN HU2/HU3] → máscara binaria por fibra (shape: H×W, dtype uint8)
       ↓
[Este módulo HU5] → recorta ROI de cada máscara a 128×128
                  → ejecuta agente PPO
                  → retorna ángulo estimado por fibra
       ↓
[Histograma angular + CSV de salida HU6/GUI]
```

La interfaz de entrada esperada del módulo en producción:
```python
def estimate_fiber_orientation(mask: np.ndarray,
                               model_path: str) -> float:
    """
    mask: máscara binaria de una fibra, shape arbitrario, dtype uint8
    Retorna: ángulo estimado en grados [0, 180)
    """
```
