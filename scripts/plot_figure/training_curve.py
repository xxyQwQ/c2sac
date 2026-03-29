import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-config"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


TASK_ORDER = [
    ("walk", "medium", "Walk-Medium"),
    ("run", "medium", "Run-Medium"),
    ("walk", "replay", "Walk-Replay"),
    ("run", "replay", "Run-Replay"),
]

METHOD_ORDER = ["bc", "gail", "bcq", "cql", "c2sac"]

METHOD_LABELS = {
    "bcq": "BCQ",
    "gail": "GAIL",
    "cql": "CQL",
    "bc": "BC",
    "c2sac": "C2SAC",
}

METHOD_COLORS = {
    "bcq": "#f2b38e",
    "gail": "#97c443",
    "cql": "#ef8fe9",
    "bc": "#79d3f3",
    "c2sac": "#ff5a66",
}

LINE_PATTERN = re.compile(
    r"Evaluate\s*\|\s*Epoch:\s*(?P<epoch>\d+)\s*\|\s*Task:\s*walker_(?P<task>\w+)\s*\|"
    r".*?Average Return:\s*(?P<score>[-+]?\d+(?:\.\d+)?)\s*\|"
)


def find_wandb_files_dir(seed_dir):
    wandb_root = seed_dir / "wandb"
    if not wandb_root.is_dir():
        return None

    latest_files = wandb_root / "latest-run" / "files"
    if latest_files.is_dir():
        return latest_files

    run_dirs = sorted(
        [path for path in wandb_root.glob("run-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        files_dir = run_dir / "files"
        if files_dir.is_dir():
            return files_dir
    return None


def smooth(values, window):
    if window <= 1 or len(values) <= 1:
        return values.copy()
    window = min(window, len(values))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def read_text_log(seed_dir):
    wandb_files_dir = find_wandb_files_dir(seed_dir)
    candidates = [
        wandb_files_dir / "output.log" if wandb_files_dir else None,
        seed_dir / "trainer.log",
    ]
    for path in candidates:
        if path and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"No wandb/output log found under {seed_dir}")


def parse_seed_log(seed_dir):
    records = {}
    content = read_text_log(seed_dir)
    for match in LINE_PATTERN.finditer(content):
        task_name = match.group("task")
        epoch = int(match.group("epoch"))
        score = float(match.group("score"))
        records.setdefault(task_name, []).append((epoch, score))

    parsed = {}
    for task_name, entries in records.items():
        entries.sort(key=lambda item: item[0])
        epochs = np.array([item[0] for item in entries], dtype=float)
        scores = np.array([item[1] for item in entries], dtype=float)
        parsed[task_name] = (epochs, scores)
    return parsed


def collect_runs(checkpoints_root):
    data = {}
    dataset_dirs = {
        "medium": "all-medium",
        "replay": "all-replay",
    }
    for method in METHOD_ORDER:
        method_root = checkpoints_root / method
        method_data = {}
        for dataset_key, dataset_dir in dataset_dirs.items():
            run_root = method_root / dataset_dir
            seed_runs = []
            for seed_dir in sorted(run_root.glob("seed-*")):
                seed_runs.append(parse_seed_log(seed_dir))

            for task_name in ("walk", "run"):
                task_runs = []
                for seed_run in seed_runs:
                    if task_name not in seed_run:
                        continue
                    task_runs.append(seed_run[task_name])
                if not task_runs:
                    raise ValueError(f"Missing task '{task_name}' for {method}/{dataset_dir}")
                shared_epochs = sorted(
                    set(task_runs[0][0].astype(int)).intersection(
                        *(set(epochs.astype(int)) for epochs, _ in task_runs[1:])
                    )
                )
                if not shared_epochs:
                    raise ValueError(f"No shared evaluation epochs for {method}/{dataset_dir}/{task_name}")
                shared_epochs = np.array(shared_epochs, dtype=float)
                aligned_scores = []
                for epochs, scores in task_runs:
                    score_by_epoch = {
                        int(epoch): score for epoch, score in zip(epochs.astype(int), scores)
                    }
                    aligned_scores.append([score_by_epoch[int(epoch)] for epoch in shared_epochs])
                stacked_scores = np.array(aligned_scores, dtype=float)
                method_data[(task_name, dataset_key)] = {
                    "epochs": shared_epochs,
                    "scores": stacked_scores,
                }
        data[method] = method_data
    return data


def read_final_summary(seed_dir, task_name):
    wandb_files_dir = find_wandb_files_dir(seed_dir)
    if not wandb_files_dir:
        return None
    summary_path = wandb_files_dir / "wandb-summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary.get(f"eval/{task_name}/average_return")


def build_summary_table(checkpoints_root):
    rows = []
    dataset_dirs = {
        "medium": "all-medium",
        "replay": "all-replay",
    }
    for task_name, dataset_key, title in TASK_ORDER:
        row = {"Task": title}
        for method in METHOD_ORDER:
            seed_values = []
            run_root = checkpoints_root / method / dataset_dirs[dataset_key]
            for seed_dir in sorted(run_root.glob("seed-*")):
                value = read_final_summary(seed_dir, task_name)
                if value is not None:
                    seed_values.append(value)
            if seed_values:
                values = np.array(seed_values, dtype=float)
                row[METHOD_LABELS[method]] = f"{values.mean():.2f} ± {values.std():.2f}"
            else:
                row[METHOD_LABELS[method]] = "N/A"
        rows.append(row)
    return rows


def print_summary_table(rows):
    headers = ["Task"] + [METHOD_LABELS[method] for method in METHOD_ORDER]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(line)
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def configure_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "DejaVu Serif" in available_fonts:
        plt.rcParams["font.family"] = "DejaVu Serif"
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman"]
        print("Warning: 'DejaVu Serif' is not installed. Falling back to the closest available serif font.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 16


def plot_figure(data, output_path, smooth_window, dpi):
    configure_plot_style()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for task_name, dataset_key, title in TASK_ORDER:
        fig, axis = plt.subplots(figsize=(10, 6))
        legend_handles = []
        legend_labels = []

        for method in METHOD_ORDER:
            record = data[method][(task_name, dataset_key)]
            epochs = record["epochs"]
            scores = record["scores"]
            mean = scores.mean(axis=0)
            std = scores.std(axis=0)
            center = smooth(mean, smooth_window)
            lower = smooth(mean - std, smooth_window)
            upper = smooth(mean + std, smooth_window)
            line, = axis.plot(
                epochs,
                center,
                color=METHOD_COLORS[method],
                linewidth=2.0,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                epochs,
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.18,
                linewidth=0,
            )
            legend_handles.append(line)
            legend_labels.append(METHOD_LABELS[method])

        axis.set_title(title, fontsize=22, fontweight="semibold")
        axis.set_xlabel("Step", fontsize=18)
        axis.set_ylabel("Score", fontsize=18)
        axis.set_xlim(left=0)
        axis.grid(True, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.0,
            handlelength=1.0,
            handletextpad=0.4,
        )

        figure_name = f"{task_name}_{dataset_key}.png"
        figure_path = output_path.parent / figure_name
        fig.tight_layout()
        fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot four training curves from local wandb logs in checkpoints/."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory containing method checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures") / "training_curves.png",
        help="Output path whose parent directory will receive the four rendered figures.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average smoothing window applied to mean/std curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Figure DPI.",
    )
    args = parser.parse_args()

    data = collect_runs(args.checkpoints)
    plot_figure(data, args.output, args.smooth_window, args.dpi)
    print(f"Saved figure to {args.output}")
    print()
    print_summary_table(build_summary_table(args.checkpoints))


if __name__ == "__main__":
    main()
