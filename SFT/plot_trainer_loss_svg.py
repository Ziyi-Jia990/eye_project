from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Trainer loss curves to an SVG without external deps.")
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Trainer Loss Curve")
    return parser.parse_args()


def extract_series(trainer_state_path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])

    train_points: list[dict[str, float]] = []
    eval_points: list[dict[str, float]] = []

    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_points.append(
                {
                    "step": float(step),
                    "epoch": float(epoch) if epoch is not None else 0.0,
                    "loss": float(entry["loss"]),
                }
            )
        if "eval_loss" in entry:
            eval_points.append(
                {
                    "step": float(step),
                    "epoch": float(epoch) if epoch is not None else 0.0,
                    "eval_loss": float(entry["eval_loss"]),
                }
            )
    return train_points, eval_points


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def build_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_csv(path: Path, train_points: list[dict[str, float]], eval_points: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "step", "epoch", "loss"])
        writer.writeheader()
        for point in train_points:
            writer.writerow(
                {
                    "kind": "train",
                    "step": int(point["step"]),
                    "epoch": point["epoch"],
                    "loss": point["loss"],
                }
            )
        for point in eval_points:
            writer.writerow(
                {
                    "kind": "eval",
                    "step": int(point["step"]),
                    "epoch": point["epoch"],
                    "loss": point["eval_loss"],
                }
            )


def write_svg(
    path: Path,
    train_points: list[dict[str, float]],
    eval_points: list[dict[str, float]],
    title: str,
) -> None:
    width = 1400
    height = 820
    left = 110
    right = 60
    top = 90
    bottom = 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_steps = [point["step"] for point in train_points] + [point["step"] for point in eval_points]
    all_losses = [point["loss"] for point in train_points] + [point["eval_loss"] for point in eval_points]

    min_step = min(all_steps)
    max_step = max(all_steps)
    min_loss = min(all_losses)
    max_loss = max(all_losses)

    loss_pad = max((max_loss - min_loss) * 0.08, 0.005)
    min_loss -= loss_pad
    max_loss += loss_pad

    train_xy = [
        (
            scale(point["step"], min_step, max_step, left, left + plot_w),
            scale(point["loss"], min_loss, max_loss, top + plot_h, top),
        )
        for point in train_points
    ]
    eval_xy = [
        (
            scale(point["step"], min_step, max_step, left, left + plot_w),
            scale(point["eval_loss"], min_loss, max_loss, top + plot_h, top),
        )
        for point in eval_points
    ]

    x_ticks = 6
    y_ticks = 6

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" font-size="28" font-family="Arial, sans-serif" fill="#1f2937">{title}</text>',
        f'<text x="{width/2:.1f}" y="72" text-anchor="middle" font-size="15" font-family="Arial, sans-serif" fill="#6b7280">Train loss (blue) and eval loss (red) from Trainer log_history</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>',
    ]

    for i in range(x_ticks + 1):
        step_value = min_step + (max_step - min_step) * i / x_ticks
        x = scale(step_value, min_step, max_step, left, left + plot_w)
        svg_lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eef2f7" stroke-width="1"/>')
        svg_lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#4b5563">{int(round(step_value))}</text>')

    for i in range(y_ticks + 1):
        loss_value = min_loss + (max_loss - min_loss) * i / y_ticks
        y = scale(loss_value, min_loss, max_loss, top + plot_h, top)
        svg_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eef2f7" stroke-width="1"/>')
        svg_lines.append(f'<text x="{left - 16}" y="{y + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial, sans-serif" fill="#4b5563">{loss_value:.3f}</text>')

    svg_lines.extend(
        [
            f'<text x="{width/2:.1f}" y="{height - 32}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111827">Step</text>',
            f'<text x="36" y="{height/2:.1f}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111827" transform="rotate(-90 36,{height/2:.1f})">Loss</text>',
        ]
    )

    if train_xy:
        svg_lines.append(
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{build_polyline(train_xy)}"/>'
        )

    if eval_xy:
        svg_lines.append(
            f'<polyline fill="none" stroke="#dc2626" stroke-width="2.2" points="{build_polyline(eval_xy)}"/>'
        )
        for x, y in eval_xy:
            svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#dc2626"/>')

    legend_x = left + 24
    legend_y = top + 24
    svg_lines.extend(
        [
            f'<rect x="{legend_x - 12}" y="{legend_y - 18}" width="220" height="62" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 30}" y2="{legend_y}" stroke="#2563eb" stroke-width="3"/>',
            f'<text x="{legend_x + 40}" y="{legend_y + 5}" font-size="15" font-family="Arial, sans-serif" fill="#111827">train loss</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 26}" x2="{legend_x + 30}" y2="{legend_y + 26}" stroke="#dc2626" stroke-width="3"/>',
            f'<circle cx="{legend_x + 15}" cy="{legend_y + 26}" r="4.5" fill="#dc2626"/>',
            f'<text x="{legend_x + 40}" y="{legend_y + 31}" font-size="15" font-family="Arial, sans-serif" fill="#111827">eval loss</text>',
        ]
    )

    best_eval = min(eval_points, key=lambda item: item["eval_loss"]) if eval_points else None
    if best_eval is not None:
        svg_lines.append(
            f'<text x="{left + plot_w - 8}" y="{top + 24}" text-anchor="end" font-size="15" font-family="Arial, sans-serif" fill="#6b7280">best eval_loss = {best_eval["eval_loss"]:.6f} @ step {int(best_eval["step"])}</text>'
        )

    svg_lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_points, eval_points = extract_series(args.trainer_state)
    if not train_points and not eval_points:
        raise ValueError(f"No loss points found in {args.trainer_state}")
    if args.output_csv is not None:
        write_csv(args.output_csv, train_points, eval_points)
    write_svg(args.output_svg, train_points, eval_points, args.title)
    print(f"train_points={len(train_points)}")
    print(f"eval_points={len(eval_points)}")
    print(f"output_svg={args.output_svg}")
    if args.output_csv is not None:
        print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
