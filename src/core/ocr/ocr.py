#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Whiteboard ID extractor for 360° videos
#
# Scans the first N seconds of a .360/.mp4 video, tries to detect a whiteboard
# as a big rectangular bright region, deskews it, enhances it, and OCRs an ID.
# Saves BOTH color and thresholded crops + annotated frames for debugging.
#
# Requirements:
#   pip install opencv-python numpy pytesseract
# System OCR:
#   Ubuntu: sudo apt-get install -y tesseract-ocr
#   macOS:  brew install tesseract
#   Win:    UB Mannheim build

import argparse
import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional


# ---------- helpers ----------

def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    d = deg % 360
    if d == 0:
        return img
    if d == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def read_video_meta(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    return fps, total


def frame_indices_for_window(fps: float, seconds: float, sample_fps: float) -> List[int]:
    step = max(int(round(fps / max(sample_fps, 0.1))), 1)
    max_idx = int(seconds * fps)
    return list(range(0, max_idx, step))


def crop_center_band(img: np.ndarray, band_ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    band_h = int(h * band_ratio)
    top = max((h - band_h) // 2, 0)
    return img[top:top + band_h, :]


# ---------- detection ----------

def find_whiteboard_quads(gray: np.ndarray,
                          min_area_ratio: float,
                          min_aspect: float) -> List[np.ndarray]:
    # Adaptive threshold on blurred gray; invert to make 'white' bright
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 5)
    th = 255 - th

    # Open/close to clean speckles and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(th, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    area_min = min_area_ratio * w * h
    quads = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            (_, _), (rw, rh), _ = rect
            if rw == 0 or rh == 0:
                continue
            ar = max(rw, rh) / max(min(rw, rh), 1e-6)
            if ar >= min_aspect:  # likely a whiteboard (wide-ish)
                quads.append(approx.reshape(4, 2).astype(np.float32))

    quads = sorted(quads, key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)), reverse=True)
    return quads


def order_quad(quad: np.ndarray) -> np.ndarray:
    pts = quad.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img: np.ndarray, quad: np.ndarray, out_w: int) -> Optional[np.ndarray]:
    q = order_quad(quad)
    w1 = np.linalg.norm(q[1] - q[0])
    w2 = np.linalg.norm(q[2] - q[3])
    h1 = np.linalg.norm(q[3] - q[0])
    h2 = np.linalg.norm(q[2] - q[1])
    w = int(max(w1, w2))
    h = int(max(h1, h2))
    if w <= 0 or h <= 0:
        return None
    scale = out_w / float(w)
    out_h = max(int(h * scale), 1)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


# ---------- OCR prep ----------

def preprocess_for_ocr(img_bgr: np.ndarray,
                       use_clahe: bool,
                       gamma: float,
                       use_sharpen: bool) -> np.ndarray:
    # Gamma (<1 brightens mids). Build LUT for speed.
    if abs(gamma - 1.0) > 1e-3:
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]).astype("uint8")
        img_bgr = cv2.LUT(img_bgr, lut)

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    else:
        g = cv2.equalizeHist(g)

    if use_sharpen:
        blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
        g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)

    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return th


def ocr_text(img: np.ndarray, psm: int, oem: int, whitelist: str) -> List[Tuple[str, int]]:
    cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
    out = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            c = int(conf)
        except Exception:
            c = -1
        t = (txt or "").strip()
        if t and c >= 0:
            out.append((t, c))
    return out


def best_match_from_tokens(tokens: List[Tuple[str, int]], pattern: re.Pattern) -> Optional[Tuple[str, float]]:
    if not tokens:
        return None
    joined = " ".join(t for t, _ in tokens)
    cand = None

    m = pattern.search(joined)
    if m:
        span = m.span()
        confs = []
        idx = 0
        for t, c in tokens:
            s = idx
            e = idx + len(t)
            if not (e <= span[0] or s >= span[1]):
                confs.append(c)
            idx = e + 1
        if confs:
            cand = (m.group(0), float(sum(confs)) / len(confs))

    if cand is None:
        best = (None, -1.0)
        for t, c in tokens:
            m2 = pattern.search(t)
            if m2 and c > best[1]:
                best = (m2.group(0), float(c))
        if best[0] is not None:
            cand = best

    return cand


# ---------- per-frame pipeline ----------

def extract_id_from_frame(frame_bgr: np.ndarray,
                          pattern: re.Pattern,
                          band_ratio: float,
                          try_center_band_first: bool,
                          psm: int, oem: int,
                          warp_width: int,
                          scale: float,
                          gamma: float,
                          use_clahe: bool,
                          use_sharpen: bool,
                          max_quads: int,
                          min_area_ratio: float,
                          min_aspect: float,
                          save_crops_dir: Optional[Path],
                          save_annotated_dir: Optional[Path],
                          frame_idx: Optional[int],
                          rot_deg: Optional[int]) -> Optional[Tuple[str, float]]:
    candidates = []
    regions = []

    if try_center_band_first:
        regions.append(crop_center_band(frame_bgr, band_ratio))
    regions.append(frame_bgr)

    for region in regions:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        quads = find_whiteboard_quads(gray, min_area_ratio=min_area_ratio, min_aspect=min_aspect)

        # Annotate quads if requested
        if save_annotated_dir is not None:
            ann = region.copy()
            for q in quads:
                cv2.polylines(ann, [q.astype(np.int32)], True, (0, 255, 0), 2)
            base = f"f{frame_idx if frame_idx is not None else -1}_r{rot_deg if rot_deg is not None else 0}"
            cv2.imwrite(str(save_annotated_dir / f"{base}_annot.jpg"), ann)

        # Try each quad
        for qi, q in enumerate(quads[:max_quads]):
            warped = warp_quad(region, q, out_w=warp_width)
            if warped is None:
                continue
            if scale and scale > 1.0:
                warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Save COLOR before binarization (so you can see it clearly)
            if save_crops_dir is not None:
                base = f"f{frame_idx if frame_idx is not None else -1}_r{rot_deg if rot_deg is not None else 0}_q{qi}"
                cv2.imwrite(str(save_crops_dir / f"{base}_warped_color.jpg"), warped)

            th = preprocess_for_ocr(warped, use_clahe=use_clahe, gamma=gamma, use_sharpen=use_sharpen)
            if save_crops_dir is not None:
                cv2.imwrite(str(save_crops_dir / f"{base}_warped_th.png"), th)

            toks = ocr_text(th, psm=psm, oem=oem, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            hit = best_match_from_tokens(toks, pattern)
            if hit:
                candidates.append(hit)

        # If no candidates from quads, try a bright-region fallback ROI (non-polygon)
        if not candidates:
            # global threshold to find bright blobs
            _, thb = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thb = cv2.morphologyEx(thb, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            cnts, _ = cv2.findContours(thb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for ci, c in enumerate(cnts[:2]):  # check a couple of biggest bright regions
                x, y, w, h = cv2.boundingRect(c)
                if w * h < (min_area_ratio * region.shape[0] * region.shape[1]):
                    continue
                roi = region[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                if scale and scale > 1.0:
                    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                if save_crops_dir is not None:
                    base = f"f{frame_idx if frame_idx is not None else -1}_r{rot_deg if rot_deg is not None else 0}_b{ci}"
                    cv2.imwrite(str(save_crops_dir / f"{base}_bright_color.jpg"), roi)

                th2 = preprocess_for_ocr(roi, use_clahe=use_clahe, gamma=gamma, use_sharpen=use_sharpen)
                if save_crops_dir is not None:
                    cv2.imwrite(str(save_crops_dir / f"{base}_bright_th.png"), th2)

                toks2 = ocr_text(th2, psm=psm, oem=oem, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                hit2 = best_match_from_tokens(toks2, pattern)
                if hit2:
                    candidates.append(hit2)

        if candidates:
            return max(candidates, key=lambda x: x[1])

        # Fallback: OCR entire region (save both looks)
        if save_crops_dir is not None:
            base = f"f{frame_idx if frame_idx is not None else -1}_r{rot_deg if rot_deg is not None else 0}_whole"
            cv2.imwrite(str(save_crops_dir / f"{base}_whole_color.jpg"), region)
        th_whole = preprocess_for_ocr(region, use_clahe=use_clahe, gamma=gamma, use_sharpen=use_sharpen)
        if save_crops_dir is not None:
            cv2.imwrite(str(save_crops_dir / f"{base}_whole_th.png"), th_whole)
        toks_whole = ocr_text(th_whole, psm=psm, oem=oem, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        hit_whole = best_match_from_tokens(toks_whole, pattern)
        if hit_whole:
            return hit_whole

    return None


# ---------- video-level aggregation ----------

def extract_whiteboard_id(video_path: str,
                          seconds: float,
                          sample_fps: float,
                          pattern_str: str,
                          tesseract_cmd: Optional[str],
                          try_rotations: Optional[List[int]],
                          band_ratio: float,
                          center_first: bool,
                          psm: int, oem: int,
                          warp_width: int,
                          scale: float,
                          gamma: float,
                          use_clahe: bool,
                          use_sharpen: bool,
                          max_quads: int,
                          min_area_ratio: float,
                          min_aspect: float,
                          save_crops: Optional[str],
                          save_annotated: Optional[str]) -> Tuple[Optional[str], float, dict]:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    pattern = re.compile(pattern_str)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps, _total = read_video_meta(cap)
    idxs = frame_indices_for_window(fps, seconds, sample_fps)
    rotations = try_rotations or [0]

    votes = Counter()
    confs = defaultdict(list)
    checked = 0

    crops_dir = Path(save_crops) if save_crops else None
    ann_dir = Path(save_annotated) if save_annotated else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
    if ann_dir:
        ann_dir.mkdir(parents=True, exist_ok=True)

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        checked += 1

        best_hit = None
        for deg in rotations:
            fr = rotate_image(frame, deg)
            _hit = extract_id_from_frame(
                fr, pattern,
                band_ratio=band_ratio,
                try_center_band_first=center_first,
                psm=psm, oem=oem,
                warp_width=warp_width,
                scale=scale,
                gamma=gamma,
                use_clahe=use_clahe,
                use_sharpen=use_sharpen,
                max_quads=max_quads,
                min_area_ratio=min_area_ratio,
                min_aspect=min_aspect,
                save_crops_dir=crops_dir,
                save_annotated_dir=ann_dir,
                frame_idx=fi,
                rot_deg=deg,
            )
            if _hit and (best_hit is None or _hit[1] > best_hit[1]):
                best_hit = _hit

        if best_hit:
            val, conf = best_hit
            votes[val] += 1
            confs[val].append(conf)

    cap.release()

    if not votes:
        return None, 0.0, {
            "frames_checked": checked,
            "candidates": {},
            "rotations": rotations,
            "save_crops": str(crops_dir) if crops_dir else None,
            "save_annotated": str(ann_dir) if ann_dir else None,
        }

    # Choose by votes then confidence
    best_key = None
    best_score = None
    for k, cnt in votes.items():
        avg = sum(confs[k]) / max(len(confs[k]), 1)
        sc = (cnt, avg)
        if best_score is None or sc > best_score:
            best_key, best_score = k, sc

    best_id = best_key
    best_conf = sum(confs[best_id]) / max(len(confs[best_id]), 1)
    debug = {
        "frames_checked": checked,
        "candidates": {k: {"votes": votes[k], "avg_conf": sum(confs[k]) / len(confs[k])} for k in votes},
        "rotations": rotations,
        "save_crops": str(crops_dir) if crops_dir else None,
        "save_annotated": str(ann_dir) if ann_dir else None,
    }
    return best_id, float(best_conf), debug


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Extract a whiteboard ID from the start of a 360° video")
    p.add_argument("--video", required=True, help="Path to .360 or other video file")
    p.add_argument("--seconds", type=float, default=20.0, help="How many seconds from the start to scan")
    p.add_argument("--sample-fps", type=float, default=3.0, help="Frames per second to analyze")
    p.add_argument("--try-rotations", default="0,180,90,270", help="Comma list of degrees to try per frame")
    p.add_argument("--pattern", default=r"[A-Z0-9]{4,}", help="Regex for ID (alphanumeric, no spaces)")
    p.add_argument("--tesseract-cmd", default=None, help="Path to tesseract binary if not on PATH")

    # Detector + OCR tuning
    p.add_argument("--band-ratio", type=float, default=0.6, help="Center band height ratio (0..1)")
    p.add_argument("--no-center-first", action="store_true", help="Do NOT prioritize center band first")
    p.add_argument("--psm", type=int, default=11, help="Tesseract PSM (try 6, 7, or 11)")
    p.add_argument("--oem", type=int, default=3, help="Tesseract OEM (0,1,3)")
    p.add_argument("--warp-width", type=int, default=1600, help="Warped board width before OCR")
    p.add_argument("--scale", type=float, default=2.0, help="Extra upscale factor (1.0=off)")
    p.add_argument("--gamma", type=float, default=0.9, help="Gamma correction (<1 brightens mids)")
    p.add_argument("--clahe", action="store_true", help="Use CLAHE contrast boost")
    p.add_argument("--sharpen", action="store_true", help="Apply unsharp mask before binarization")
    p.add_argument("--max-quads", type=int, default=5, help="Max quads to OCR per frame")
    p.add_argument("--min-area-ratio", type=float, default=0.01, help="Min quad area as fraction of frame (e.g., 0.01)")
    p.add_argument("--min-aspect", type=float, default=1.1, help="Min aspect ratio w/h for quads")

    # Debug outputs
    p.add_argument("--save-crops", default=None, help="Directory to save cropped boards (color + threshold)")
    p.add_argument("--save-annotated", default=None, help="Directory to save frames with quad overlays")

    args = p.parse_args()
    rots = [int(x.strip()) for x in args.try_rotations.split(',') if x.strip()]
    center_first = not args.no_center_first

    try:
        best_id, conf, dbg = extract_whiteboard_id(
            video_path=args.video,
            seconds=args.seconds,
            sample_fps=args.sample_fps,
            pattern_str=args.pattern,
            tesseract_cmd=args.tesseract_cmd,
            try_rotations=rots,
            band_ratio=args.band_ratio,
            center_first=center_first,
            psm=args.psm,
            oem=args.oem,
            warp_width=args.warp_width,
            scale=args.scale,
            gamma=args.gamma,
            use_clahe=args.clahe,
            use_sharpen=args.sharpen,
            max_quads=args.max_quads,
            min_area_ratio=args.min_area_ratio,
            min_aspect=args.min_aspect,
            save_crops=args.save_crops,
            save_annotated=args.save_annotated,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return

    if best_id is None:
        print("No ID found in the scanned window.")
        print(f"Frames checked: {dbg.get('frames_checked', 0)}")
    else:
        print(f"ID: {best_id}")
        print(f"Confidence (avg): {conf:.1f}")
        print(f"Frames checked: {dbg.get('frames_checked', 0)}")
        if dbg.get("candidates"):
            print("Candidates:")
            for k, v in sorted(dbg["candidates"].items(), key=lambda kv: (-kv[1]["votes"], -kv[1]["avg_conf"])):
                print(f"  {k}: votes={v['votes']}, avg_conf={v['avg_conf']:.1f}")

    if dbg.get("save_crops") or dbg.get("save_annotated"):
        print("Debug output:")
        if dbg.get("save_crops"):
            print(f"  Crops:     {dbg['save_crops']}")
        if dbg.get("save_annotated"):
            print(f"  Annotated: {dbg['save_annotated']}")


if __name__ == "__main__":
    main()