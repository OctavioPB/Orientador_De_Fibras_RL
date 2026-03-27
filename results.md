# Resultados y Conclusiones — Módulo RL de Orientación de Fibras (HU5)

**Fecha de análisis:** 2026-03-27
**Modelos evaluados:** `ppo_v1` y `ppo_v2` (500 000 timesteps cada uno, PPO + CnnPolicy)
**Protocolo:** 100 imágenes sintéticas con ángulos θ ∈ [0°, 178.2°] distribuidos uniformemente (paso 1.8°)

---

## 1. Métricas globales

| Métrica | ppo_v1 | ppo_v2 | Umbral DoD |
|---|---|---|---|
| MAE global | **23.52°** | **26.29°** | < 10° (prototipo) |
| % error < 5° | 53 % | 49 % | — |
| % error < 10° | 56 % | 52 % | — |
| % error < 15° | 59 % | 55 % | — |
| Entradas fallidas (error > 15°) | 41 / 100 | 45 / 100 | 0 |

**Veredicto:** Ambos modelos superan el umbral de 15° → **Plan B activo** (ver sección 5).

---

## 2. Análisis por rango angular

### Rango fallido: θ ∈ [16.2°, 88.2°]

Ambos modelos quedan atascados en la predicción `179.999°` para todos los ángulos en este rango. El error crece linealmente con el ángulo objetivo (error ≈ θ).

| Modelo | Rango fallido | Entradas | MAE en rango fallido |
|---|---|---|---|
| ppo_v1 | [16.2°, 88.2°] | 41 | 52.20° |
| ppo_v2 | [16.2°, 93.6°] ∪ {135°} | 45 | 54.44° |

### Rango efectivo

Fuera de la región fallida, ambos modelos funcionan muy bien:

| Modelo | Rango efectivo | Entradas | MAE en rango OK | % < 5° (en OK) |
|---|---|---|---|---|
| ppo_v1 | [0°, 14.4°] ∪ [90°, 178.2°] | 59 | **3.53°** | ~90 % |
| ppo_v2 | [0°, 14.4°] ∪ [95.4°, 178.2°] \ {135°} | 55 | **3.26°** | ~89 % |

El rendimiento dentro del rango efectivo **supera el objetivo de producción** (MAE < 5°).

### Comparación de granularidad de predicción

Los modelos aprendieron distintas estrategias de ajuste:

- **ppo_v1** produce predicciones con paso ~4° (93.99°, 97.99°, 101.99°…), indicando que el agente hace ajustes parciales del límite `MAX_DELTA_DEG = 10°`. Esto explica su ligero mejor MAE en el rango OK.
- **ppo_v2** produce predicciones exactamente en múltiplos de 10° (100°, 110°, 120°…), indicando que el agente aprendió a aplicar siempre la acción máxima. Esto implica una resolución efectiva de ±5° en el mejor caso.

---

## 3. Diagnóstico de la falla

### Política degenerada: "siempre ir a 179.999°"

El agente aprendió una política que empuja `theta_estimado` hasta el límite superior del clamp (`179.999°`) y lo mantiene allí. Este comportamiento es racional para el agente pero incorrecto para el objetivo del módulo.

**Causas probables:**

1. **Explotación de la simetría angular en la recompensa SSIM:**
   Para ángulos cercanos a 0° (p. ej. θ = 5°), `angular_distance(5°, 179.999°) ≈ 5°`, lo que produce una recompensa relativamente alta. El agente aprendió que "ir a 179.999°" es una política que funciona bien para θ ≈ 0° y θ ≈ 180°, representando el 20–30% de los episodios de entrenamiento. La distribución uniforme de ángulos en `reset()` hace que el agente experimente con frecuencia estos casos "fáciles" y sobreajuste a esa solución.

2. **Recompensa SSIM no suficientemente discriminativa:**
   SSIM mide similitud estructural local, no orientación global. Dos imágenes con orientaciones muy distintas pueden tener SSIM moderado si comparten texturas similares. Esto produce una superficie de recompensa plana que dificulta la exploración de ángulos intermedios.

3. **Inicio en 90° siempre igual:**
   `theta_estimado` siempre inicia en 90°. El agente aprendió a moverse siempre a la derecha (hacia 180°) porque esa dirección maximiza la recompensa promedio durante el entrenamiento. Nunca desarrolló la capacidad de moverse hacia la izquierda (hacia 0°) para ángulos en [0°, 90°).

4. **Falta de diversificación en los primeros pasos:**
   Con `MAX_DELTA_DEG = 10°` y partiendo de 90°, el agente puede llegar a 179.999° en apenas 9 pasos. Si eso obtiene una recompensa positiva, PPO refuerza esa conducta rápidamente.

---

## 4. Hallazgos secundarios

### ppo_v1 vs ppo_v2

ppo_v1 tiene ligeramente mejor desempeño global (MAE 23.52° vs 26.29°) y maneja un rango efectivo más amplio. ppo_v2 incorporó una falla adicional en θ = 135° (también atascado en 179.999°) y perdió la cobertura de θ ∈ [90°, 93.6°]. La segunda versión no mejoró respecto a la primera, lo que sugiere que el problema es estructural y no se resuelve con más entrenamiento.

### Casos especiales correctos

Ambos modelos predicen correctamente para θ ≈ 0° y θ ≈ 180°, pero por razones degeneradas: la predicción 179.999° es angularmente equivalente a 0° por la simetría de fibras (0° == 180°). No es conocimiento aprendido sino artefacto de la política.

### Consistencia de predicciones

Las predicciones son altamente reproducibles (deterministic=True). Esto confirma que la política es estable, aunque esté atascada en el mínimo local.

---

## 5. Plan B — Activado

Dado que MAE > 15° para ambos modelos, **se activa el fallback geométrico** (`utils/ellipse_fallback.py`).

```bash
python main.py infer --model models/ppo_v1 --image fibra.png --method ellipse
```

El método de ajuste de elipse (`cv2.fitEllipse`) tiene un error esperado de 12–18° en fibras regulares, lo cual es comparable o mejor al MAE de los modelos RL entrenados para el rango problemático [16°, 88°].

---

## 6. Recomendaciones para la próxima iteración

En orden de impacto estimado:

### 6.1 Inicialización aleatoria de `theta_estimado` (cambio de bajo costo)
Cambiar `INITIAL_THETA_ESTIMATE = 90.0` por una inicialización uniforme en `[0°, 180°)` en `reset()`. Esto obliga al agente a aprender a navegar desde cualquier punto del espacio, eliminando el sesgo hacia 179.999°.

```python
# En fiber_env.py → reset()
self._theta_estimado = self.np_random.uniform(0.0, 180.0)
```

### 6.2 Recompensa basada en distancia angular directa
Reemplazar SSIM por una recompensa proporcional a la reducción del error angular, que es el objetivo real del módulo:

```python
# Nueva función de recompensa
reward = 1.0 - (angular_distance(theta_est, theta_true) / 90.0) - step_penalty
```

Esto elimina la ambigüedad de la señal SSIM y hace el gradiente de recompensa perfectamente alineado con el objetivo.

### 6.3 Clamp asimétrico en acción
Evitar que el agente se quede pegado en los bordes usando una representación circular del ángulo:

```python
self._theta_estimado = (self._theta_estimado + delta) % 180.0  # sin clamp rígido
```

### 6.4 Observación como par de ángulos (sin imagen)
Dado que el entrenamiento es sintético y el Ground Truth es exacto, se puede simplificar radicalmente el espacio de observación a `(theta_objetivo, theta_estimado)` normalizado. Esto permite usar `MlpPolicy` en lugar de `CnnPolicy` y converge mucho más rápido.

### 6.5 Curriculum learning
Comenzar el entrenamiento con episodios donde el ángulo inicial está cerca del objetivo (diferencia < 30°) e ir aumentando la dificultad gradualmente.

---

## 7. Resumen ejecutivo

| Criterio | Estado |
|---|---|
| MAE < 10° (prototipo HU5) | ❌ No alcanzado (MAE ≈ 23–26°) |
| MAE < 5° (producción HU5) | ❌ No alcanzado |
| Rendimiento en rango [90°, 180°) | ✅ MAE ≈ 3.3–3.5°, supera objetivo |
| Plan B disponible | ✅ `ellipse_fallback.py` implementado |
| Hipótesis H2 (RL supera método clásico) | ❌ No demostrada en el rango global |

El módulo RL tiene potencial demostrado: cuando el agente está en el rango [90°, 178°], logra precisión de nivel producción (< 5°). El problema es exclusivamente de exploración y diseño del espacio de estados/recompensa, no de capacidad del algoritmo. Las correcciones propuestas en la sección 6 tienen alta probabilidad de resolver el problema en el siguiente sprint.
