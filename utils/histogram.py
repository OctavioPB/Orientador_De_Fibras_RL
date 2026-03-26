"""Generación de histograma polar de distribución angular de fibras."""

import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

N_BINS = 18
BIN_WIDTH_DEG = 180.0 / N_BINS  # 10° por bin


def plot_angular_histogram(
    angles: list[float],
    output_path: str = "results/angular_histogram.png",
    title: str = "Distribución Angular de Fibras",
) -> None:
    """Genera histograma polar de distribución angular de fibras.

    Crea un histograma polar con 18 bins de 10° cada uno en el rango [0°, 180°).
    La simetría 0°==180° se maneja duplicando el histograma en [180°, 360°)
    para visualización polar completa. Guarda la imagen PNG y un CSV con
    frecuencias por bin.

    Args:
        angles: Lista de ángulos en grados en el rango [0°, 180°).
        output_path: Ruta de salida para la imagen PNG.
        title: Título del histograma.
    """
    angles_arr = np.array(angles, dtype=np.float64) % 180.0

    bin_edges = np.linspace(0.0, 180.0, N_BINS + 1)
    counts, _ = np.histogram(angles_arr, bins=bin_edges)

    # Centros de bins en radianes para el plot polar
    bin_centers_deg = bin_edges[:-1] + BIN_WIDTH_DEG / 2.0

    # Duplicar para el hemisfério completo (0°–360°)
    counts_full = np.concatenate([counts, counts])
    centers_full_rad = np.deg2rad(np.concatenate([bin_centers_deg, bin_centers_deg + 180.0]))
    bar_width = np.deg2rad(BIN_WIDTH_DEG)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.bar(
        centers_full_rad,
        counts_full,
        width=bar_width,
        align="center",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_theta_zero_location("E")   # 0° a la derecha (estilo matemático)
    ax.set_theta_direction(1)          # sentido antihorario
    ax.set_title(title, pad=20)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "results", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Histograma PNG guardado en '%s'.", output_path)

    # Guardar CSV con frecuencias por bin
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_center_deg", "count"])
        for center, count in zip(bin_centers_deg, counts):
            writer.writerow([round(float(center), 2), int(count)])
    logger.info("Histograma CSV guardado en '%s'.", csv_path)
