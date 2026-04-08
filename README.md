# Orientador de Fibras Musculares — Módulo RL (HU5)

Módulo de estimación de orientación angular de fibras musculares mediante
Reinforcement Learning (PPO). Forma parte de un sistema mayor de análisis
histológico automatizado (tesis de maestría UNIR).

## Problema

Los métodos geométricos clásicos (ajuste de elipse mínima) tienen un error
angular de ~12–18° en fibras irregulares o dañadas. Este módulo busca reducir
ese error a < 10° (prototipo) y < 5° (producción) usando un agente PPO
entrenado completamente con imágenes sintéticas de Ground Truth absoluto —
sin necesidad de anotaciones manuales.

## Arquitectura

```
[Mask R-CNN HU2/HU3]
        ↓ máscara binaria por fibra
[Este módulo HU5]
        ↓ recorta ROI a 128×128 → agente PPO → ángulo estimado
[Histograma angular + CSV HU6/GUI]
```

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| RL framework | `stable-baselines3` + `gymnasium` |
| Algoritmo | PPO con CnnPolicy |
| Visión | `opencv-python`, `scikit-image` |
| Backend DL | PyTorch |
| Visualización | `matplotlib`, `pandas` |
| Tests | `pytest` |

## Estructura del proyecto

```
rl_orientation/
├── main.py                      ← CLI: train / eval / infer
├── requirements.txt
│
├── env/
│   ├── fiber_env.py             ← Entorno Gymnasium (CORE)
│   └── synthetic_generator.py  ← Generador de imágenes sintéticas (CORE)
│
├── agent/
│   ├── train.py                 ← Loop de entrenamiento PPO
│   └── evaluate.py             ← Evaluación MAE angular
│
├── utils/
│   ├── reward.py                ← Función de recompensa SSIM
│   └── histogram.py            ← Histograma polar angular
│
└── tests/
    ├── test_generator.py
    ├── test_env.py
    └── test_reward.py
```

## Instalación

```bash
pip install -r requirements.txt
```

## Usogit

### Entrenar

```bash
python main.py train --timesteps 500000 --save models/ppo_v1
```

Opciones:
- `--timesteps` — pasos de entrenamiento (default: 500 000)
- `--save` — ruta del modelo guardado (default: `models/ppo_fiber_orientation`)
- `--log-dir` — directorio de logs TensorBoard (default: `logs/`)

### Evaluar

```bash
python main.py eval --model models/ppo_v1 --n 100 --output results/eval.csv
```

Genera un CSV con columnas `[theta_true, theta_predicted, error_deg]` e imprime:

```
=== Resultados de evaluación ===
  MAE angular : 7.32°
  Error < 5°  : 41.0%
  Error < 10° : 78.0%
```

### Inferencia sobre imagen real

```bash
python main.py infer --model models/ppo_v1 --image ruta/imagen.png
```

Redimensiona la imagen a 128×128, ejecuta el agente y muestra el ángulo
estimado con un vector de orientación superpuesto.

#### Fallback geométrico (Plan B)

```bash
python main.py infer --model models/ppo_v1 --image imagen.png --method ellipse
```

### Visualizar evaluación

```bash
# Usar el CSV generado por eval (por defecto: results/eval_v2.csv)
python plot_evaluation.py

# Especificar CSV
python plot_evaluation.py --csv results/eval.csv

# Comparar dos evaluaciones
python plot_evaluation.py --csv results/eval_v2.csv --compare results/eval.csv

# Guardar figura en lugar de mostrarla
python plot_evaluation.py --save results/eval_plots.png
```

Genera una figura con 7 paneles:
- Scatter θ_real vs θ_predicho (coloreado por error)
- Error angular por ángulo real con umbrales HU5
- Distribución acumulada (CDF) del error
- Histograma de errores con MAE y mediana
- Diagrama polar del error por ángulo
- Boxplot por cuadrante angular
- Tabla de métricas con indicadores ✓/✗ vs umbrales

El CSV de entrada debe tener las columnas `theta_true`, `theta_predicted`, `error_deg`
(formato generado automáticamente por `python main.py eval`).

### Tests

```bash
pytest tests/ -v
```

## Entorno Gymnasium

| Componente | Especificación |
|---|---|
| Observación | imagen objetivo `(128, 128, 1)` uint8 |
| Acción | continua `(1,)` en `[-1, 1]` → Δθ ∈ `[-10°, 10°]` |
| Recompensa | `2·SSIM(img_objetivo, img_estimada) − 1 − 0.01` |
| Terminación | `error_angular < 5°` o `steps > 200` |

## Generador sintético

Las imágenes simulan fibras musculares como elipses paralelas alineadas al
ángulo θ con ruido gaussiano. Parámetros configurables:

| Parámetro | Rango | Default |
|---|---|---|
| `theta` | [0°, 180°) | — |
| `n_fibers` | 5–20 | 12 |
| `fiber_width` | 4–8 px | aleatorio |
| `fiber_length` | 30–60 px | aleatorio |
| `noise_std` | 0–15 | 8.0 |

## Criterios de aceptación (HU5)

| Criterio | Umbral | Estado |
|---|---|---|
| MAE prototipo | < 10° | pendiente eval |
| MAE producción | < 5° | pendiente eval |
| Tests | 100% sin warnings | ✅ 15/15 |
| `check_env` Gymnasium | pasa | ✅ |

## Integración con Mask R-CNN

```python
from main import estimate_fiber_orientation

angle = estimate_fiber_orientation(mask=mask_binaria, model_path="models/ppo_v1")
# → float, ángulo en grados [0°, 180°)
```

## Plan B (fallback geométrico)

Si `MAE > 15°` al final del sprint, se activa `utils/ellipse_fallback.py`:
ajuste de elipse mínima con `cv2.fitEllipse`. Error esperado: ~12–18°.
