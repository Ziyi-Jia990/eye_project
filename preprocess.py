import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import fundus_image_toolbox as fit

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path):
    return path.suffix.lower() in IMG_EXTS


def collect_images(root_dir):
    root = Path(root_dir)
    return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def resize_with_pad_bg(img, out_size=(1024, 1024), bg_color=None):
    target_h, target_w = out_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if bg_color is None:
        bg_color = estimate_bg_color_from_corners(img)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def estimate_bg_color_from_corners(img_rgb, patch_ratio=0.08):
    h, w = img_rgb.shape[:2]
    ph = max(10, int(h * patch_ratio))
    pw = max(10, int(w * patch_ratio))

    patches = [
        img_rgb[:ph, :pw],
        img_rgb[:ph, -pw:],
        img_rgb[-ph:, :pw],
        img_rgb[-ph:, -pw:],
    ]

    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    bg_color = np.median(pixels, axis=0).astype(np.uint8)
    return bg_color


def largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    out = (labels == max_idx).astype(np.uint8) * 255
    return out


def robust_fundus_mask(img_rgb):
    h, w = img_rgb.shape[:2]

    bg_color = estimate_bg_color_from_corners(img_rgb)
    diff = img_rgb.astype(np.float32) - bg_color.astype(np.float32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    ph = max(10, int(h * 0.08))
    pw = max(10, int(w * 0.08))
    corner_mask = np.zeros((h, w), dtype=np.uint8)
    corner_mask[:ph, :pw] = 1
    corner_mask[:ph, -pw:] = 1
    corner_mask[-ph:, :pw] = 1
    corner_mask[-ph:, -pw:] = 1
    corner_dist = dist[corner_mask == 1]

    thr = max(18.0, float(np.median(corner_dist) + 15.0))
    mask = (dist > thr).astype(np.uint8) * 255

    k = max(3, int(min(h, w) * 0.01))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = largest_connected_component(mask)
    if mask is None:
        return None

    k2 = max(5, int(min(h, w) * 0.02))
    if k2 % 2 == 0:
        k2 += 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    return mask


def crop_from_mask(img_rgb, mask, margin_ratio=0.02):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgb

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = img_rgb.shape[:2]
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    side = max(bw, bh)

    margin = int(side * margin_ratio)
    side = side + 2 * margin

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    half = side // 2
    left = max(0, cx - half)
    right = min(w, cx + half)
    top = max(0, cy - half)
    bottom = min(h, cy + half)

    crop = img_rgb[top:bottom, left:right]
    return crop


def remove_red_annotations(img_rgb):
    h, w = img_rgb.shape[:2]
    img = img_rgb.copy()

    border = np.zeros((h, w), dtype=np.uint8)
    margin_h = max(30, int(h * 0.18))
    margin_w = max(30, int(w * 0.18))

    border[:margin_h, :] = 1
    border[-margin_h:, :] = 1
    border[:, :margin_w] = 1
    border[:, -margin_w:] = 1

    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)

    red_mask = (
        (r > 140) &
        (r - g > 50) &
        (r - b > 50) &
        (border == 1)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    clean_mask = np.zeros_like(red_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]

        if 20 <= area <= 5000:
            if x < margin_w or x + ww > w - margin_w or y < margin_h or y + hh > h - margin_h:
                clean_mask[labels == i] = 255

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.inpaint(img_bgr, clean_mask, 5, cv2.INPAINT_TELEA)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return out_rgb, clean_mask


def custom_crop(img_rgb, out_size=(1024, 1024)):
    mask = robust_fundus_mask(img_rgb)

    if mask is None:
        return resize_with_pad_bg(
            img_rgb,
            out_size,
            bg_color=estimate_bg_color_from_corners(img_rgb)
        ), "fallback_raw"

    crop = crop_from_mask(img_rgb, mask, margin_ratio=0.03)
    crop = resize_with_pad_bg(
        crop,
        out_size,
        bg_color=estimate_bg_color_from_corners(img_rgb)
    )
    return crop, "custom_mask_crop"


def robust_crop(img_rgb, out_size=(1024, 1024)):
    try:
        out = fit.crop(img_rgb, size=out_size)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        return out, "fit_crop"
    except Exception:
        out, mode = custom_crop(img_rgb, out_size)
        return out, mode


def process_one(args):
    img_path, input_root, output_root, out_size, skip_existing = args
    img_path = Path(img_path)
    input_root = Path(input_root)
    output_root = Path(output_root)

    rel_path = img_path.relative_to(input_root)
    out_path = output_root / rel_path

    try:
        # 已经处理过就跳过
        if skip_existing and out_path.exists():
            return {
                "input": str(img_path),
                "output": str(out_path),
                "status": "skipped",
                "reason": "output_exists"
            }

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"cv2.imread failed: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb, _ = remove_red_annotations(img_rgb)
        cropped, mode = robust_crop(img_rgb, out_size)

        save_rgb(out_path, cropped)

        return {
            "input": str(img_path),
            "output": str(out_path),
            "status": "ok",
            "mode": mode
        }

    except Exception as e:
        return {
            "input": str(img_path),
            "output": str(out_path),
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }


def load_processed_from_jsonl(log_path):
    processed = set()
    if not log_path.exists():
        return processed

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "input" in item:
                    processed.add(item["input"])
            except json.JSONDecodeError:
                pass
    return processed


def append_jsonl(log_path, record):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(
    input_dir,
    output_dir,
    out_size=(1024, 1024),
    max_images=None,
    skip_existing=True,
    resume_from_log=True,
    num_workers=1,
):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / "crop_log.jsonl"

    images = collect_images(input_root)
    if max_images is not None:
        images = images[:max_images]

    processed_inputs = set()
    if resume_from_log:
        processed_inputs = load_processed_from_jsonl(log_path)

    todo = []
    skipped_by_output = 0
    skipped_by_log = 0

    for img_path in images:
        rel_path = img_path.relative_to(input_root)
        out_path = output_root / rel_path

        if skip_existing and out_path.exists():
            skipped_by_output += 1
            continue

        if resume_from_log and str(img_path) in processed_inputs:
            skipped_by_log += 1
            continue

        todo.append(img_path)

    print(f"Found {len(images)} images total.")
    print(f"Skip by existing output: {skipped_by_output}")
    print(f"Skip by existing log   : {skipped_by_log}")
    print(f"Need process           : {len(todo)}")

    ok_cnt = 0
    err_cnt = 0
    skip_cnt = skipped_by_output + skipped_by_log
    fit_cnt = 0
    custom_cnt = 0
    raw_cnt = 0

    if num_workers == 1:
        for i, img_path in enumerate(todo, 1):
            res = process_one((img_path, input_root, output_root, out_size, skip_existing))
            append_jsonl(log_path, res)

            if res["status"] == "ok":
                ok_cnt += 1
                if res.get("mode") == "fit_crop":
                    fit_cnt += 1
                elif res.get("mode") == "custom_mask_crop":
                    custom_cnt += 1
                elif res.get("mode") == "fallback_raw":
                    raw_cnt += 1
            elif res["status"] == "error":
                err_cnt += 1
            elif res["status"] == "skipped":
                skip_cnt += 1

            if i % 50 == 0 or i == len(todo):
                print(
                    f"[{i}/{len(todo)}] "
                    f"ok={ok_cnt}, error={err_cnt}, skipped={skip_cnt}, "
                    f"fit_crop={fit_cnt}, custom_mask_crop={custom_cnt}, fallback_raw={raw_cnt}"
                )
    else:
        tasks = [
            (img_path, input_root, output_root, out_size, skip_existing)
            for img_path in todo
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process_one, task) for task in tasks]

            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                append_jsonl(log_path, res)

                if res["status"] == "ok":
                    ok_cnt += 1
                    if res.get("mode") == "fit_crop":
                        fit_cnt += 1
                    elif res.get("mode") == "custom_mask_crop":
                        custom_cnt += 1
                    elif res.get("mode") == "fallback_raw":
                        raw_cnt += 1
                elif res["status"] == "error":
                    err_cnt += 1
                elif res["status"] == "skipped":
                    skip_cnt += 1

                if i % 50 == 0 or i == len(todo):
                    print(
                        f"[{i}/{len(todo)}] "
                        f"ok={ok_cnt}, error={err_cnt}, skipped={skip_cnt}, "
                        f"fit_crop={fit_cnt}, custom_mask_crop={custom_cnt}, fallback_raw={raw_cnt}"
                    )

    print(f"Done. Log saved to: {log_path}")


if __name__ == "__main__":
    main(
        input_dir="/mnt/hdd/jiazy/eye_project/image_data/raw",
        output_dir="/mnt/hdd/jiazy/eye_project/image_data/name",
        out_size=(1024, 1024),
        max_images=None,
        skip_existing=True,
        resume_from_log=True,
        num_workers=4,   # 先用1测试；确认稳定后可改成4/8
    )