# Orientador de Fibras Musculares — Modelo Preentrenado

Módulo de inferencia listo para usar. Estima el ángulo de orientación dominante
(0° – 180°) de fibras musculares a partir de una imagen o máscara binaria.

---

## Requisitos

Python 3.9 o superior.

```bash
pip install -r requirements.txt
```

---

## Estructura

```
minreq/
├── pretrained_model.py        ← punto de entrada principal
├── requirements.txt
├── models/
│   ├── ppo_v1.zip             ← agente PPO entrenado (500 000 steps)
│   └── ppo_v2.zip             ← segunda versión del agente
├── env/
│   ├── fiber_env.py           ← entorno de ejecución del agente
│   └── synthetic_generator.py ← generador de imágenes sintéticas
└── utils/
    ├── reward.py              ← función de recompensa SSIM
    └── ellipse_fallback.py    ← método geométrico alternativo (Plan B)
```

---

## Uso desde Python

```python
from pretrained_model import FiberOrientationModel

# Cargar el modelo una sola vez
model = FiberOrientationModel("models/ppo_v1")

# Inferencia básica — devuelve el ángulo en grados [0°, 180°)
angle = model.predict("ruta/imagen.png")
print(f"Orientación estimada: {angle:.1f}°")

# Con visualización — devuelve también la imagen con el vector superpuesto (PNG base64)
angle, viz_b64 = model.predict("ruta/imagen.png", return_visualization=True)

# El string base64 se puede usar directamente en una UI web:
# <img src="data:image/png;base64,{viz_b64}" />
```

También acepta arrays numpy directamente:

```python
import cv2
img = cv2.imread("fibra.png", cv2.IMREAD_GRAYSCALE)
angle = model.predict(img)
```

---

## Uso desde línea de comandos

```bash
# Inferencia con modelo RL
python pretrained_model.py imagen.png --model models/ppo_v1

# Fallback geométrico (no requiere modelo RL)
python pretrained_model.py imagen.png --method ellipse

# Segunda versión del modelo
python pretrained_model.py imagen.png --model models/ppo_v2
```

La visualización se guarda automáticamente como `imagen_result.png` en la misma carpeta.

---

## Métodos disponibles

| Método | Descripción | MAE (rango efectivo) |
|---|---|---|
| `rl` (ppo_v1) | Agente PPO — preciso en θ ∈ [90°, 180°) | ~3.5° |
| `rl` (ppo_v2) | Agente PPO v2 — preciso en θ ∈ [95°, 180°) | ~3.3° |
| `ellipse` | Ajuste de elipse mínima (fallback geométrico) | ~12–18° |

> **Nota:** Los modelos RL tienen baja precisión para ángulos en el rango [16°, 88°].
> Para ese rango se recomienda usar `--method ellipse` hasta que se entrene una nueva versión.

---

## Comportamiento automático

- Si el archivo `.zip` del modelo no existe, el sistema cae automáticamente al método `ellipse` sin lanzar error.
- Las imágenes de cualquier tamaño se redimensionan internamente a 128 × 128 px.
- Las imágenes en color se convierten automáticamente a escala de grises.
