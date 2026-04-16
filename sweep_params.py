#!/usr/bin/env python3
"""
Script para barrido sistemático de parámetros (alpha, C_pw, c) en MCTS Scalable.

Lógica del barrido (según análisis de alpha-c-profundidad):
- alpha: rango [0.15, 0.55] con paso 0.05 (controla profundidad vs amplitud)
- C_pw: inversamente proporcional a alpha (menor alpha → menor C_pw para conservar presupuesto)
- c: inversamente proporcional a alpha (menor alpha → menor c para no competir con presión vertical)

Relaciones sugeridas:
  alpha=0.15-0.25: C_pw=1.0-1.5, c=1.5-1.8 (máxima profundidad, CC=32)
  alpha=0.25-0.35: C_pw=1.5-2.0, c=1.8-2.2 (balance)
  alpha=0.35-0.50: C_pw=2.0-2.5, c=2.0-2.5 (amplitud para problemas chicos)
"""

import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime
import sys

def run_mcts(bits, alpha, C_pw, c, iterations=10000, n_rollouts=32):
    """Ejecuta mcts_scalable.py con los parámetros dados."""

    output_dir = f"mcts_sweep_alpha{alpha}_Cpw{C_pw}_c{c}"

    cmd = [
        "python3", "/home/servergmun/MCTS_CUCA_CACA/mcts_scalable.py",
        "--bits", str(bits),
        "--height", str(bits),
        "--iterations", str(iterations),
        "--alpha", str(alpha),
        "--C-pw", str(C_pw),
        "--c", str(c),
        "--n-rollouts", str(n_rollouts),
        "--output", output_dir,
    ]

    print(f"\n{'='*70}")
    print(f"Ejecutando: alpha={alpha}, C_pw={C_pw}, c={c}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Error en ejecución:")
            print(result.stderr)
            return None

        # Lee stats.json
        stats_file = Path(output_dir) / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            print(f"✅ Completado: reward={stats['best_reward']:.4f}, depth={stats['final_depth']}, nodes={stats['final_nodes']}")
            return stats
        else:
            print(f"⚠️  No se encontró stats.json")
            return None

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout en ejecución")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Barrido inteligente de parámetros."""

    bits = 4  # 4 bits es el objetivo
    iterations = 10000
    n_rollouts = 32

    # Definir puntos del barrido según la lógica analítica
    # Agrupamos en "regímenes" según alpha
    sweep_configs = [
        # Régimen 1: Máxima profundidad (alpha bajo)
        {"alpha": 0.15, "C_pw": 1.0, "c": 1.5},
        {"alpha": 0.20, "C_pw": 1.0, "c": 1.5},
        {"alpha": 0.25, "C_pw": 1.5, "c": 1.8},

        # Régimen 2: Balance profundidad-amplitud
        {"alpha": 0.30, "C_pw": 1.5, "c": 2.0},
        {"alpha": 0.35, "C_pw": 2.0, "c": 2.0},

        # Régimen 3: Balance intermedio
        {"alpha": 0.40, "C_pw": 2.0, "c": 2.2},
        {"alpha": 0.45, "C_pw": 2.0, "c": 2.2},

        # Régimen 4: Más amplitud (baseline actual)
        {"alpha": 0.50, "C_pw": 2.0, "c": 2.0},
        {"alpha": 0.50, "C_pw": 2.0, "c": 2.5},  # Tu baseline actual
    ]

    results = []

    for config in sweep_configs:
        stats = run_mcts(
            bits=bits,
            alpha=config["alpha"],
            C_pw=config["C_pw"],
            c=config["c"],
            iterations=iterations,
            n_rollouts=n_rollouts
        )

        if stats:
            row = {
                "alpha": config["alpha"],
                "C_pw": config["C_pw"],
                "c": config["c"],
                "best_reward": stats.get("best_reward"),
                "final_depth": stats.get("final_depth"),
                "final_nodes": stats.get("final_nodes"),
                "branching_factor": stats.get("final_branching_factor"),
                "elapsed_time_s": stats.get("elapsed_time_s"),
            }
            results.append(row)

    # Guardar resultados en CSV
    csv_file = "sweep_results.csv"
    if results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Resultados guardados en {csv_file}")

        # Mostrar tabla resumen
        print("\n" + "="*90)
        print("RESUMEN DEL BARRIDO")
        print("="*90)
        print(f"{'Alpha':<8} {'C_pw':<8} {'c':<8} {'Reward':<12} {'Depth':<8} {'Nodes':<8} {'BF':<8} {'Time(s)':<8}")
        print("-"*90)
        for row in sorted(results, key=lambda r: r["best_reward"], reverse=True):
            print(f"{row['alpha']:<8.2f} {row['C_pw']:<8.2f} {row['c']:<8.2f} {row['best_reward']:<12.4f} "
                  f"{row['final_depth']:<8} {row['final_nodes']:<8} {row['branching_factor']:<8.2f} {row['elapsed_time_s']:<8.2f}")

        # Mostrar TOP 3
        sorted_results = sorted(results, key=lambda r: (r["final_depth"], r["best_reward"]), reverse=True)
        print("\n" + "="*90)
        print("TOP 3 (por profundidad, luego reward)")
        print("="*90)
        for i, row in enumerate(sorted_results[:3], 1):
            print(f"{i}. alpha={row['alpha']}, C_pw={row['C_pw']}, c={row['c']}")
            print(f"   → reward={row['best_reward']:.4f}, depth={row['final_depth']}, nodes={row['final_nodes']}")
    else:
        print("\n❌ No se completó ningún experimento.")

if __name__ == "__main__":
    main()
