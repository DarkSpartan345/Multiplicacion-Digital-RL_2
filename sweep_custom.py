#!/usr/bin/env python3
"""
Script para barrido personalizado de parámetros.
Permite especificar rangos personalizados de alpha, C_pw, c.

Uso:
  python3 sweep_custom.py --alpha-range 0.15 0.55 0.05 --C_pw-range 1.0 2.5 0.5 --c-range 1.5 2.5 0.2
  python3 sweep_custom.py --preset deep      # Optimizado para profundidad máxima
  python3 sweep_custom.py --preset balanced  # Balance profundidad-amplitud
  python3 sweep_custom.py --preset broad     # Máxima amplitud
"""

import subprocess
import json
import csv
from pathlib import Path
import argparse
import sys

def run_mcts(bits, alpha, C_pw, c, iterations=10000, n_rollouts=32, height=None):
    """Ejecuta mcts_scalable.py con los parámetros dados."""

    if height is None:
        height = bits

    output_dir = f"sweep_a{alpha:.2f}_cp{C_pw:.2f}_c{c:.2f}"

    cmd = [
        "python3", "/home/servergmun/MCTS_CUCA_CACA/mcts_scalable.py",
        "--bits", str(bits),
        "--height", str(height),
        "--iterations", str(iterations),
        "--alpha", str(alpha),
        "--C-pw", str(C_pw),
        "--c", str(c),
        "--n-rollouts", str(n_rollouts),
        "--output", output_dir,
    ]

    print(f"  → a={alpha:.2f}, Cp={C_pw:.2f}, c={c:.2f}...", end=" ", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Error")
            return None

        stats_file = Path(output_dir) / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            print(f"✅ r={stats['best_reward']:.4f}, d={stats['final_depth']}")
            return stats
        else:
            print(f"⚠️  No stats.json")
            return None

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout")
        return None
    except Exception as e:
        print(f"❌ {e}")
        return None

def generate_configs_from_ranges(alpha_range, C_pw_range, c_range):
    """Genera combinaciones a partir de rangos (min, max, step)."""
    configs = []
    alpha_start, alpha_end, alpha_step = alpha_range
    C_pw_start, C_pw_end, C_pw_step = C_pw_range
    c_start, c_end, c_step = c_range

    alpha_vals = [round(alpha_start + i * alpha_step, 2) for i in range(int((alpha_end - alpha_start) / alpha_step) + 1)]
    C_pw_vals = [round(C_pw_start + i * C_pw_step, 2) for i in range(int((C_pw_end - C_pw_start) / C_pw_step) + 1)]
    c_vals = [round(c_start + i * c_step, 2) for i in range(int((c_end - c_start) / c_step) + 1)]

    for alpha in alpha_vals:
        for C_pw in C_pw_vals:
            for c in c_vals:
                configs.append({"alpha": alpha, "C_pw": C_pw, "c": c})

    return configs

def get_preset(preset_name):
    """Retorna configuraciones predefinidas."""
    presets = {
        "deep": {
            "desc": "Optimizado para máxima profundidad (CC=32)",
            "alpha_range": (0.15, 0.30, 0.05),
            "C_pw_range": (1.0, 1.5, 0.25),
            "c_range": (1.5, 1.8, 0.15),
        },
        "balanced": {
            "desc": "Balance profundidad-amplitud",
            "alpha_range": (0.25, 0.40, 0.05),
            "C_pw_range": (1.5, 2.0, 0.25),
            "c_range": (1.8, 2.2, 0.2),
        },
        "broad": {
            "desc": "Máxima amplitud (baseline actual)",
            "alpha_range": (0.45, 0.55, 0.05),
            "C_pw_range": (2.0, 2.5, 0.25),
            "c_range": (2.0, 2.5, 0.25),
        },
    }

    if preset_name not in presets:
        print(f"❌ Preset desconocido: {preset_name}")
        print(f"   Disponibles: {', '.join(presets.keys())}")
        sys.exit(1)

    return presets[preset_name]

def main():
    parser = argparse.ArgumentParser(description="Barrido personalizado de parámetros MCTS")
    parser.add_argument("--bits", type=int, default=4, help="Número de bits (default: 4)")
    parser.add_argument("--iterations", type=int, default=10000, help="Iteraciones MCTS (default: 10000)")
    parser.add_argument("--n-rollouts", type=int, default=32, help="Rollouts paralelos (default: 32)")
    parser.add_argument("--preset", choices=["deep", "balanced", "broad"],
                        help="Usa preset de parámetros")
    parser.add_argument("--alpha-range", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                        help="Rango de alpha: MIN MAX STEP")
    parser.add_argument("--C-pw-range", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                        help="Rango de C_pw: MIN MAX STEP")
    parser.add_argument("--c-range", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                        help="Rango de c: MIN MAX STEP")

    args = parser.parse_args()

    # Determinar configuraciones
    if args.preset:
        preset = get_preset(args.preset)
        print(f"Usando preset: {preset['desc']}")
        configs = generate_configs_from_ranges(
            preset["alpha_range"],
            preset["C_pw_range"],
            preset["c_range"]
        )
    elif args.alpha_range and args.C_pw_range and args.c_range:
        configs = generate_configs_from_ranges(
            tuple(args.alpha_range),
            tuple(args.C_pw_range),
            tuple(args.c_range)
        )
    else:
        print("❌ Especifica --preset o los tres rangos (--alpha-range, --C-pw-range, --c-range)")
        parser.print_help()
        sys.exit(1)

    print(f"\n🚀 Iniciando barrido con {len(configs)} configuraciones")
    print(f"   Bits={args.bits}, Iteraciones={args.iterations}, Rollouts={args.n_rollouts}\n")

    results = []

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", end=" ")
        stats = run_mcts(
            bits=args.bits,
            alpha=config["alpha"],
            C_pw=config["C_pw"],
            c=config["c"],
            iterations=args.iterations,
            n_rollouts=args.n_rollouts
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

    # Guardar resultados
    csv_file = f"sweep_{args.preset or 'custom'}_results.csv"
    if results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Tabla resumen
        print("\n" + "="*100)
        print("RESUMEN DEL BARRIDO")
        print("="*100)
        print(f"{'Alpha':<8} {'C_pw':<8} {'c':<8} {'Reward':<12} {'Depth':<8} {'Nodes':<8} {'BF':<8} {'Time':<8}")
        print("-"*100)

        for row in sorted(results, key=lambda r: (r["final_depth"], r["best_reward"]), reverse=True):
            print(f"{row['alpha']:<8.2f} {row['C_pw']:<8.2f} {row['c']:<8.2f} {row['best_reward']:<12.4f} "
                  f"{row['final_depth']:<8} {row['final_nodes']:<8} {row['branching_factor']:<8.2f} {row['elapsed_time_s']:<8.2f}")

        # TOP 3
        sorted_results = sorted(results, key=lambda r: (r["final_depth"], r["best_reward"]), reverse=True)
        print("\n" + "="*100)
        print("🏆 TOP 3 (por profundidad, luego reward)")
        print("="*100)
        for i, row in enumerate(sorted_results[:3], 1):
            print(f"{i}. α={row['alpha']:.2f}, C_pw={row['C_pw']:.2f}, c={row['c']:.2f} "
                  f"→ d={row['final_depth']}, r={row['best_reward']:.4f}, nodes={row['final_nodes']}")

        print(f"\n✅ Resultados completos en: {csv_file}")

if __name__ == "__main__":
    main()
