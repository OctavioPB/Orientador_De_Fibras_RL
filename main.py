"""Punto de entrada CLI del módulo de orientación de fibras musculares.

Uso:
    python main.py train  --timesteps 500000 --save models/ppo_v1
    python main.py eval   --model models/ppo_v1 --n 100 --output results/eval.csv
    python main.py infer  --model models/ppo_v1 --image ruta/imagen.png
"""

import argparse
import base64
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    from agent.train import train
    train(total_timesteps=args.timesteps, save_path=args.save, log_dir=args.log_dir)


def cmd_eval(args: argparse.Namespace) -> None:
    from agent.evaluate import evaluate
    metrics = evaluate(model_path=args.model, n_images=args.n, output_csv=args.output)
    print("\n=== Resultados de evaluación ===")
    print(f"  MAE angular : {metrics['mae']:.2f}°")
    print(f"  Error < 5°  : {metrics['pct_lt5']:.1f}%")
    print(f"  Error < 10° : {metrics['pct_lt10']:.1f}%")
    if metrics["mae"] > 15.0:
        print("\n[ADVERTENCIA] MAE > 15°. Considerar activar fallback geométrico.")
    elif metrics["mae"] < 5.0:
        print("\n[OK] Objetivo de producción alcanzado (MAE < 5°).")
    elif metrics["mae"] < 10.0:
        print("\n[OK] Prototipo válido (MAE < 10°).")


def cmd_infer(args: argparse.Namespace) -> None:
    from pretrained_model import FiberOrientationModel

    method = "ellipse" if args.method == "ellipse" else "auto"
    fiber_model = FiberOrientationModel(model_path=args.model, method=method)
    angle, viz_b64 = fiber_model.predict(args.image, return_visualization=True)
    print(f"\nÁngulo estimado: {angle:.2f}°  (método: {fiber_model.active_method})")

    out_path = Path(args.image).with_stem(Path(args.image).stem + "_result").with_suffix(".png")
    out_path.write_bytes(base64.b64decode(viz_b64))
    print(f"Visualización guardada en: {out_path}")

    import matplotlib.pyplot as plt
    import cv2
    img_result = cv2.imread(str(out_path))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Ángulo estimado: {angle:.1f}°")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def estimate_fiber_orientation(mask: np.ndarray, model_path: str) -> float:
    """Estima el ángulo de orientación de una fibra a partir de su máscara binaria.

    Punto de integración con Mask R-CNN (HU2/HU3).

    Args:
        mask: Máscara binaria de la fibra, uint8, shape arbitrario.
        model_path: Ruta al modelo PPO guardado.

    Returns:
        Ángulo estimado en grados [0°, 180°).
    """
    from pretrained_model import FiberOrientationModel
    return FiberOrientationModel(model_path=model_path).predict(mask)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Módulo RL de orientación de fibras musculares")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entrenar agente PPO")
    p_train.add_argument("--timesteps", type=int, default=500_000)
    p_train.add_argument("--save", type=str, default="models/ppo_fiber_orientation")
    p_train.add_argument("--log-dir", type=str, default="logs/", dest="log_dir")

    p_eval = sub.add_parser("eval", help="Evaluar modelo entrenado")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--n", type=int, default=100)
    p_eval.add_argument("--output", type=str, default="results/evaluation.csv")

    p_infer = sub.add_parser("infer", help="Inferir ángulo de una imagen PNG")
    p_infer.add_argument("--model", type=str, required=True)
    p_infer.add_argument("--image", type=str, required=True)
    p_infer.add_argument("--method", choices=["rl", "ellipse"], default="rl")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    {"train": cmd_train, "eval": cmd_eval, "infer": cmd_infer}[args.command](args)


if __name__ == "__main__":
    main()
