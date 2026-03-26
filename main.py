"""Punto de entrada CLI del módulo de orientación de fibras musculares (HU5).

Modos de uso:
    python main.py train  --timesteps 500000 --save models/ppo_v1
    python main.py eval   --model models/ppo_v1 --n 100 --output results/eval.csv
    python main.py infer  --model models/ppo_v1 --image ruta/imagen.png
"""

import argparse
import logging
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Entrena el agente PPO."""
    from agent.train import train

    train(
        total_timesteps=args.timesteps,
        save_path=args.save,
        log_dir=args.log_dir,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    """Evalúa el modelo entrenado."""
    from agent.evaluate import evaluate

    metrics = evaluate(
        model_path=args.model,
        n_images=args.n,
        output_csv=args.output,
    )
    print("\n=== Resultados de evaluación ===")
    print(f"  MAE angular : {metrics['mae']:.2f}°")
    print(f"  Error < 5°  : {metrics['pct_lt5']:.1f}%")
    print(f"  Error < 10° : {metrics['pct_lt10']:.1f}%")

    if metrics["mae"] > 15.0:
        print("\n[ADVERTENCIA] MAE > 15°. Considerar activar Plan B (fallback geométrico).")
    elif metrics["mae"] < 5.0:
        print("\n[OK] Objetivo de producción alcanzado (MAE < 5°).")
    elif metrics["mae"] < 10.0:
        print("\n[OK] Prototipo válido (MAE < 10°).")


def cmd_infer(args: argparse.Namespace) -> None:
    """Infiere el ángulo de orientación de una imagen PNG."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    from env.fiber_env import FiberOrientationEnv, angular_distance

    # Cargar imagen
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("No se pudo cargar la imagen: %s", args.image)
        sys.exit(1)

    img = cv2.resize(img, (128, 128))

    # Cargar modelo
    model = PPO.load(args.model)

    # Preparar entorno para inferencia
    def make_env():
        return FiberOrientationEnv()

    vec_env = DummyVecEnv([make_env])
    vec_env = VecTransposeImage(vec_env)

    obs = vec_env.reset()

    # Inyectar la imagen real en el entorno
    raw_env: FiberOrientationEnv = vec_env.envs[0]  # type: ignore[attr-defined]
    raw_env._img_objetivo = img
    raw_env._theta_estimado = 90.0
    raw_env._step_count = 0

    img_norm = img.astype(np.float32) / 255.0
    obs = img_norm[..., np.newaxis][np.newaxis, ...]  # (1, H, W, 1)  # (1, 1, H, W)

    done = False
    theta_estimated = 90.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        theta_estimated = infos[0].get("theta_estimated", theta_estimated)

    vec_env.close()

    print(f"\nÁngulo estimado: {theta_estimated:.2f}°")

    # Visualización con vector de orientación superpuesto
    import matplotlib.pyplot as plt

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cx, cy = 64, 64
    length = 50
    angle_rad = np.deg2rad(theta_estimated)
    x2 = int(cx + length * np.cos(angle_rad))
    y2 = int(cy + length * np.sin(angle_rad))
    x1 = int(cx - length * np.cos(angle_rad))
    y1 = int(cy - length * np.sin(angle_rad))
    cv2.arrowedLine(img_color, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2, tipLength=0.2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Ángulo estimado: {theta_estimated:.1f}°")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Función pública de integración con Mask R-CNN
# ---------------------------------------------------------------------------

def estimate_fiber_orientation(mask: np.ndarray, model_path: str) -> float:
    """Estima el ángulo de orientación de una fibra a partir de su máscara.

    Args:
        mask: Máscara binaria de una fibra, shape arbitrario, dtype uint8.
        model_path: Ruta al modelo PPO guardado.

    Returns:
        Ángulo estimado en grados [0°, 180°).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    from env.fiber_env import FiberOrientationEnv

    img = cv2.resize(mask, (128, 128))

    model = PPO.load(model_path)

    def make_env():
        return FiberOrientationEnv()

    vec_env = DummyVecEnv([make_env])
    vec_env = VecTransposeImage(vec_env)
    vec_env.reset()

    raw_env: FiberOrientationEnv = vec_env.envs[0]  # type: ignore[attr-defined]
    raw_env._img_objetivo = img
    raw_env._theta_estimado = 90.0
    raw_env._step_count = 0

    img_norm = img.astype(np.float32) / 255.0
    obs = img_norm[..., np.newaxis][np.newaxis, ...]  # (1, H, W, 1)

    done = False
    theta_estimated = 90.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        theta_estimated = infos[0].get("theta_estimated", theta_estimated)

    vec_env.close()
    return float(theta_estimated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Módulo RL de orientación de fibras musculares (HU5)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Entrenar agente PPO")
    p_train.add_argument("--timesteps", type=int, default=500_000, help="Pasos de entrenamiento")
    p_train.add_argument("--save", type=str, default="models/ppo_fiber_orientation", help="Ruta de guardado del modelo")
    p_train.add_argument("--log-dir", type=str, default="logs/", dest="log_dir", help="Directorio de logs TensorBoard")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluar modelo entrenado")
    p_eval.add_argument("--model", type=str, required=True, help="Ruta al modelo PPO")
    p_eval.add_argument("--n", type=int, default=100, help="Número de imágenes de evaluación")
    p_eval.add_argument("--output", type=str, default="results/evaluation.csv", help="Ruta del CSV de salida")

    # infer
    p_infer = subparsers.add_parser("infer", help="Inferir ángulo de una imagen PNG")
    p_infer.add_argument("--model", type=str, required=True, help="Ruta al modelo PPO")
    p_infer.add_argument("--image", type=str, required=True, help="Ruta a la imagen PNG")
    p_infer.add_argument(
        "--method",
        choices=["rl", "ellipse"],
        default="rl",
        help="Método de estimación: rl (PPO) o ellipse (fallback geométrico)",
    )

    return parser


def main() -> None:
    """Punto de entrada principal."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "infer":
        if args.method == "ellipse":
            _infer_ellipse(args)
        else:
            cmd_infer(args)


def _infer_ellipse(args: argparse.Namespace) -> None:
    """Fallback geométrico usando ajuste de elipse (Plan B)."""
    try:
        from utils.ellipse_fallback import estimate_orientation_ellipse
    except ImportError:
        logger.error("utils/ellipse_fallback.py no está implementado todavía.")
        sys.exit(1)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("No se pudo cargar la imagen: %s", args.image)
        sys.exit(1)

    img = cv2.resize(img, (128, 128))
    angle = estimate_orientation_ellipse(img)
    print(f"\nÁngulo estimado (ellipse fallback): {angle:.2f}°")


if __name__ == "__main__":
    main()
