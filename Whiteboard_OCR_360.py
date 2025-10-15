# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Whiteboard ID extractor for 360° videos
# ---------------------------------------
# Scans the first N seconds of a .360/.mp4 video, detects a whiteboard (bright rectangle),
# deskews it, enhances it, and performs OCR to extract an ID. 
# Saves color & thresholded crops + annotated frames + OCR text files for debugging.
# """

# import argparse
# import cv2
# import numpy as np
# import pytesseract
# import re
# import os
# from pathlib import Path
# from collections import Counter, defaultdict
# from typing import List, Tuple, Optional

# # ---------- Helpers ----------

# def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
#     d = deg % 360
#     if d == 0: return img
#     if d == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
#     if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     return img

# def read_video_meta(cap: cv2.VideoCapture) -> Tuple[float, int]:
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
#     return fps, total

# def frame_indices_for_window(fps: float, seconds: float, sample_fps: float) -> List[int]:
#     step = max(int(round(fps / max(sample_fps, 0.1))), 1)
#     max_idx = int(seconds * fps)
#     return list(range(0, max_idx, step))

# def crop_center_band(img: np.ndarray, band_ratio: float) -> np.ndarray:
#     h, w = img.shape[:2]
#     band_h = int(h * band_ratio)
#     top = max((h - band_h) // 2, 0)
#     return img[top:top + band_h, :]

# # ---------- Detection ----------

# def find_whiteboard_quads(gray: np.ndarray, min_area_ratio: float, min_aspect: float) -> List[np.ndarray]:
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     th = 255 - cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY, 35, 5)
#     kernel = np.ones((3, 3), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

#     edges = cv2.Canny(th, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     h, w = gray.shape[:2]
#     area_min = min_area_ratio * w * h
#     quads = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < area_min: continue
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         if len(approx) == 4 and cv2.isContourConvex(approx):
#             rect = cv2.minAreaRect(approx)
#             (_, _), (rw, rh), _ = rect
#             if rw == 0 or rh == 0: continue
#             ar = max(rw, rh) / max(min(rw, rh), 1e-6)
#             if ar >= min_aspect:
#                 quads.append(approx.reshape(4, 2).astype(np.float32))
#     return sorted(quads, key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)), reverse=True)

# def order_quad(quad: np.ndarray) -> np.ndarray:
#     pts = quad.reshape(4, 2)
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1).reshape(-1)
#     tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
#     tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def warp_quad(img: np.ndarray, quad: np.ndarray, out_w: int) -> Optional[np.ndarray]:
#     q = order_quad(quad)
#     w = int(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])))
#     h = int(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1])))
#     if w <= 0 or h <= 0: return None
#     scale = out_w / float(w)
#     dst = np.array([[0,0],[out_w-1,0],[out_w-1,int(h*scale)-1],[0,int(h*scale)-1]], dtype=np.float32)
#     return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (out_w, int(h*scale)))

# # ---------- OCR ----------

# def preprocess_for_ocr(img_bgr: np.ndarray, use_clahe: bool, gamma: float, use_sharpen: bool) -> np.ndarray:
#     if abs(gamma - 1.0) > 1e-3:
#         lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]).astype("uint8")
#         img_bgr = cv2.LUT(img_bgr, lut)

#     g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     g = cv2.bilateralFilter(g, 7, 50, 50)
#     if use_clahe:
#         g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
#     else:
#         g = cv2.equalizeHist(g)
#     if use_sharpen:
#         blur = cv2.GaussianBlur(g, (0, 0), 1.0)
#         g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)

#     _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)

# # ✅ Modified to save text per crop
# def ocr_text(img: np.ndarray, psm: int, oem: int, whitelist: str, save_txt_path: str = None) -> List[Tuple[str, int]]:
#     cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --user-words user_words.txt"
#     data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
#     out = []
#     for t, c in zip(data.get("text", []), data.get("conf", [])):    
#         t = (t or "").strip()
#         try:
#             c_val = int(c)
#         except Exception:
#             continue
#         if t and c_val >= 0:
#             out.append((t, c_val))

#     if save_txt_path and out:
#         os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
#         with open(save_txt_path, "w") as f:
#             f.write(" ".join([t for t, _ in out]) + "\n")
#     return out

# def best_match_from_tokens(tokens: List[Tuple[str, int]], pattern: re.Pattern) -> Optional[Tuple[str, float]]:
#     if not tokens: return None
#     joined = " ".join(t for t, _ in tokens)
#     m = pattern.search(joined)
#     if m:
#         confs = [c for t, c in tokens if m.group(0) in t]
#         if confs: return (m.group(0), sum(confs)/len(confs))
#     return None

# # ---------- Per-frame pipeline ----------

# def extract_id_from_frame(frame_bgr: np.ndarray, pattern: re.Pattern, band_ratio: float,
#                           try_center_band_first: bool, psm: int, oem: int, warp_width: int,
#                           scale: float, gamma: float, use_clahe: bool, use_sharpen: bool,
#                           max_quads: int, min_area_ratio: float, min_aspect: float,
#                           save_crops_dir: Optional[Path], save_annotated_dir: Optional[Path],
#                           frame_idx: Optional[int], rot_deg: Optional[int]) -> Optional[Tuple[str, float]]:

#     candidates, regions = [], []
#     if try_center_band_first: regions.append(crop_center_band(frame_bgr, band_ratio))
#     regions.append(frame_bgr)

#     for region in regions:
#         gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
#         quads = find_whiteboard_quads(gray, min_area_ratio, min_aspect)
#         if save_annotated_dir:
#             ann = region.copy()
#             for q in quads: cv2.polylines(ann, [q.astype(np.int32)], True, (0,255,0), 2)
#             cv2.imwrite(str(save_annotated_dir / f"f{frame_idx}_r{rot_deg}_annot.jpg"), ann)

#         for qi, q in enumerate(quads[:max_quads]):
#             warped = warp_quad(region, q, warp_width)
#             if warped is None: continue
#             if scale > 1.0: warped = cv2.resize(warped, None, fx=scale, fy=scale)
#             base = f"f{frame_idx}_r{rot_deg}_q{qi}"
#             if save_crops_dir:
#                 cv2.imwrite(str(save_crops_dir / f"{base}_color.jpg"), warped)
#             th = preprocess_for_ocr(warped, use_clahe, gamma, use_sharpen)
#             if save_crops_dir:
#                 cv2.imwrite(str(save_crops_dir / f"{base}_th.png"), th)
#             txt_path = os.path.join(save_crops_dir, f"{base}_ocr.txt") if save_crops_dir else None
#             toks = ocr_text(th, psm, oem, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", txt_path)
#             hit = best_match_from_tokens(toks, pattern)
#             if hit: candidates.append(hit)

#         if candidates: return max(candidates, key=lambda x: x[1])
#     return None

# # ---------- Video-level ----------

# def extract_whiteboard_id(video_path: str, seconds: float, sample_fps: float,
#                           pattern_str: str, tesseract_cmd: Optional[str], try_rotations: Optional[List[int]],
#                           band_ratio: float, center_first: bool, psm: int, oem: int,
#                           warp_width: int, scale: float, gamma: float, use_clahe: bool,
#                           use_sharpen: bool, max_quads: int, min_area_ratio: float,
#                           min_aspect: float, save_crops: Optional[str],
#                           save_annotated: Optional[str]) -> Tuple[Optional[str], float, dict]:

#     if tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
#     pattern = re.compile(pattern_str)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")

#     fps, _ = read_video_meta(cap)
#     idxs = frame_indices_for_window(fps, seconds, sample_fps)
#     rotations = try_rotations or [0]
#     votes, confs, checked = Counter(), defaultdict(list), 0

#     crops_dir = Path(save_crops) if save_crops else None
#     ann_dir = Path(save_annotated) if save_annotated else None
#     if crops_dir: crops_dir.mkdir(parents=True, exist_ok=True)
#     if ann_dir: ann_dir.mkdir(parents=True, exist_ok=True)

#     for fi in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
#         ok, frame = cap.read()
#         if not ok: continue
#         checked += 1
#         best_hit = None
#         for deg in rotations:
#             fr = rotate_image(frame, deg)
#             _hit = extract_id_from_frame(fr, pattern, band_ratio, center_first, psm, oem,
#                                          warp_width, scale, gamma, use_clahe, use_sharpen,
#                                          max_quads, min_area_ratio, min_aspect,
#                                          crops_dir, ann_dir, fi, deg)
#             if _hit and (best_hit is None or _hit[1] > best_hit[1]): best_hit = _hit
#         if best_hit:
#             val, conf = best_hit
#             votes[val] += 1
#             confs[val].append(conf)
#     cap.release()

#     if not votes:
#         return None, 0.0, {"frames_checked": checked, "candidates": {}, "rotations": rotations}

#     best_key = max(votes.keys(), key=lambda k: (votes[k], np.mean(confs[k])))
#     return best_key, np.mean(confs[best_key]), {
#         "frames_checked": checked,
#         "candidates": {k: {"votes": votes[k], "avg_conf": np.mean(confs[k])} for k in votes},
#         "rotations": rotations,
#     }

# # ---------- CLI ----------

# def main():
#     p = argparse.ArgumentParser(description="Extract whiteboard ID from 360° video")
#     p.add_argument("--video", required=True)
#     p.add_argument("--seconds", type=float, default=20.0)
#     p.add_argument("--sample-fps", type=float, default=3.0)
#     p.add_argument("--try-rotations", default="0,180,90,270")
#     p.add_argument("--pattern", default=r"[A-Z0-9]{4,}")
#     p.add_argument("--tesseract-cmd", default=None)
#     p.add_argument("--band-ratio", type=float, default=0.6)
#     p.add_argument("--no-center-first", action="store_true")
#     p.add_argument("--psm", type=int, default=11)
#     p.add_argument("--oem", type=int, default=3)
#     p.add_argument("--warp-width", type=int, default=1600)
#     p.add_argument("--scale", type=float, default=2.0)
#     p.add_argument("--gamma", type=float, default=0.9)
#     p.add_argument("--clahe", action="store_true")
#     p.add_argument("--sharpen", action="store_true")
#     p.add_argument("--max-quads", type=int, default=5)
#     p.add_argument("--min-area-ratio", type=float, default=0.01)
#     p.add_argument("--min-aspect", type=float, default=1.1)
#     p.add_argument("--save-crops", default=None)
#     p.add_argument("--save-annotated", default=None)
#     args = p.parse_args()

#     rots = [int(x.strip()) for x in args.try_rotations.split(',') if x.strip()]
#     center_first = not args.no_center_first

#     best_id, conf, dbg = extract_whiteboard_id(
#         args.video, args.seconds, args.sample_fps, args.pattern, args.tesseract_cmd,
#         rots, args.band_ratio, center_first, args.psm, args.oem, args.warp_width,
#         args.scale, args.gamma, args.clahe, args.sharpen, args.max_quads,
#         args.min_area_ratio, args.min_aspect, args.save_crops, args.save_annotated)

#     if best_id is None:
#         print("⚠️ No ID found.")
#     else:
#         print(f"✅ ID: {best_id} (avg conf: {conf:.1f})")
#         print(f"Frames checked: {dbg['frames_checked']}")
#         print("Candidates:")
#         for k,v in sorted(dbg["candidates"].items(), key=lambda kv: (-kv[1]["votes"], -kv[1]["avg_conf"])):
#             print(f"  {k}: votes={v['votes']}, avg_conf={v['avg_conf']:.1f}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard ID extractor for 360° videos
---------------------------------------
Detects whiteboard region, deskews, enhances, performs OCR to extract ID.
Auto-renames video if confident enough, and saves crops, annotations, OCR text.
"""

import argparse, cv2, numpy as np, pytesseract, re, os, shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

# ---------- Helpers ----------

def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    d = deg % 360
    if d == 0: return img
    if d == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

# ---------- Detection ----------

def find_whiteboard_quads(gray: np.ndarray, min_area_ratio: float, min_aspect: float) -> List[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = 255 - cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, 5)
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
        if area < area_min: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            (_, _), (rw, rh), _ = rect
            if rw == 0 or rh == 0: continue
            ar = max(rw, rh) / max(min(rw, rh), 1e-6)
            if ar >= min_aspect:
                quads.append(approx.reshape(4, 2).astype(np.float32))
    return sorted(quads, key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)), reverse=True)

def order_quad(quad: np.ndarray) -> np.ndarray:
    pts = quad.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_quad(img: np.ndarray, quad: np.ndarray, out_w: int) -> Optional[np.ndarray]:
    q = order_quad(quad)
    w = int(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])))
    h = int(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1])))
    if w <= 0 or h <= 0: return None
    scale = out_w / float(w)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,int(h*scale)-1],[0,int(h*scale)-1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (out_w, int(h*scale)))

# ---------- OCR ----------

def preprocess_for_ocr(img_bgr: np.ndarray, use_clahe: bool, gamma: float, use_sharpen: bool) -> np.ndarray:
    if abs(gamma - 1.0) > 1e-3:
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]).astype("uint8")
        img_bgr = cv2.LUT(img_bgr, lut)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g) if use_clahe else cv2.equalizeHist(g)
    if use_sharpen:
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)

def ocr_text(img: np.ndarray, psm: int, oem: int, whitelist: str, save_txt_path: str = None) -> List[Tuple[str, float]]:
    cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
    if os.path.exists("user_words.txt"):
        cfg += " --user-words user_words.txt"

    # --- First OCR pass ---
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
    out = []

    for t, c in zip(data.get("text", []), data.get("conf", [])):
        t = (t or "").strip()
        try:
            c_val = float(c)
            if np.isnan(c_val) or c_val < 0:
                c_val = 0.0
        except Exception:
            c_val = 0.0
        if t:
            out.append((t, c_val))

    # --- Secondary fallback OCR (if low confidence) ---
    if not out or np.mean([c for _, c in out]) < 10:
        cfg2 = f"--oem {oem} --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data2 = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg2)
        for t, c in zip(data2.get("text", []), data2.get("conf", [])):
            t = (t or "").strip()
            try:
                c_val = float(c)
                if np.isnan(c_val) or c_val < 0:
                    c_val = 0.0
            except Exception:
                c_val = 0.0
            if t:
                out.append((t, c_val))

    # --- Final fallback: try pytesseract.image_to_string() if still zero ---
    if not out or all(c_val == 0.0 for _, c_val in out):
        text = pytesseract.image_to_string(img, config=cfg).strip()
        if text:
            out = [(t, 50.0) for t in re.findall(r"[A-Z0-9]+", text)]  # assume mid confidence

    # --- Save OCR results ---
    if save_txt_path:
        os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
        with open(save_txt_path, "w") as f:
            for t, c in out:
                f.write(f"{t} ({c:.1f})\n")

    return out

def best_match_from_tokens(tokens: List[Tuple[str, int]], pattern: re.Pattern) -> Optional[Tuple[str, float]]:
    if not tokens: return None
    joined = " ".join(t for t, _ in tokens)
    m = pattern.search(joined)
    if m:
        confs = [c for t, c in tokens if m.group(0) in t]
        if confs: return (m.group(0), sum(confs)/len(confs))
    return None

# ---------- Per-frame ----------

def extract_id_from_frame(frame_bgr, pattern, band_ratio, try_center_band_first,
                          psm, oem, warp_width, scale, gamma, use_clahe, use_sharpen,
                          max_quads, min_area_ratio, min_aspect,
                          save_crops_dir, save_annotated_dir, frame_idx, rot_deg):
    candidates, regions = [], []
    if try_center_band_first: regions.append(crop_center_band(frame_bgr, band_ratio))
    regions.append(frame_bgr)
    for region in regions:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        quads = find_whiteboard_quads(gray, min_area_ratio, min_aspect)
        if save_annotated_dir:
            ann = region.copy()
            for q in quads: cv2.polylines(ann, [q.astype(np.int32)], True, (0,255,0), 2)
            cv2.imwrite(str(save_annotated_dir / f"f{frame_idx}_r{rot_deg}_annot.jpg"), ann)
        for qi, q in enumerate(quads[:max_quads]):
            warped = warp_quad(region, q, warp_width)
            if warped is None: continue
            if scale > 1.0: warped = cv2.resize(warped, None, fx=scale, fy=scale)
            base = f"f{frame_idx}_r{rot_deg}_q{qi}"
            if save_crops_dir:
                cv2.imwrite(str(save_crops_dir / f"{base}_color.jpg"), warped)
            th = preprocess_for_ocr(warped, use_clahe, gamma, use_sharpen)
            if save_crops_dir:
                cv2.imwrite(str(save_crops_dir / f"{base}_th.png"), th)
            txt_path = os.path.join(save_crops_dir, f"{base}_ocr.txt") if save_crops_dir else None
            toks = ocr_text(th, psm, oem, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", txt_path)
            hit = best_match_from_tokens(toks, pattern)
            if hit: candidates.append(hit)
        if candidates: return max(candidates, key=lambda x: x[1])
    return None

# ---------- Video-level ----------

def extract_whiteboard_id(video_path, seconds, sample_fps, pattern_str, tesseract_cmd, try_rotations,
                          band_ratio, center_first, psm, oem, warp_width, scale, gamma, use_clahe,
                          use_sharpen, max_quads, min_area_ratio, min_aspect, save_crops, save_annotated):
    if tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pattern = re.compile(pattern_str)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")
    fps, _ = read_video_meta(cap)
    idxs = frame_indices_for_window(fps, seconds, sample_fps)
    rotations = try_rotations or [0]
    votes, confs, checked = Counter(), defaultdict(list), 0
    crops_dir = Path(save_crops) if save_crops else None
    ann_dir = Path(save_annotated) if save_annotated else None
    if crops_dir: crops_dir.mkdir(parents=True, exist_ok=True)
    if ann_dir: ann_dir.mkdir(parents=True, exist_ok=True)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        checked += 1
        best_hit = None
        for deg in rotations:
            fr = rotate_image(frame, deg)
            _hit = extract_id_from_frame(fr, pattern, band_ratio, center_first, psm, oem,
                                         warp_width, scale, gamma, use_clahe, use_sharpen,
                                         max_quads, min_area_ratio, min_aspect,
                                         crops_dir, ann_dir, fi, deg)
            if _hit and (best_hit is None or _hit[1] > best_hit[1]): best_hit = _hit
        if best_hit:
            val, conf = best_hit
            votes[val] += 1
            confs[val].append(conf)
    cap.release()
    if not votes:
        return None, 0.0, {"frames_checked": checked, "candidates": {}, "rotations": rotations}
    best_key = max(votes.keys(), key=lambda k: (votes[k], np.mean(confs[k])))
    return best_key, np.mean(confs[best_key]), {
        "frames_checked": checked,
        "candidates": {k: {"votes": votes[k], "avg_conf": np.mean(confs[k])} for k in votes},
        "rotations": rotations,
    }

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Extract whiteboard ID from 360° video")
    p.add_argument("--video", required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--sample-fps", type=float, default=3.0)
    p.add_argument("--try-rotations", default="0,180,90,270")
    p.add_argument("--pattern", default=r"[A-Z0-9]{4,}")
    p.add_argument("--tesseract-cmd", default=None)
    p.add_argument("--band-ratio", type=float, default=0.6)
    p.add_argument("--no-center-first", action="store_true")
    p.add_argument("--psm", type=int, default=11)
    p.add_argument("--oem", type=int, default=3)
    p.add_argument("--warp-width", type=int, default=1600)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--sharpen", action="store_true")
    p.add_argument("--max-quads", type=int, default=5)
    p.add_argument("--min-area-ratio", type=float, default=0.01)
    p.add_argument("--min-aspect", type=float, default=1.1)
    p.add_argument("--save-crops", default=None)
    p.add_argument("--save-annotated", default=None)
    p.add_argument("--dry-run", action="store_true", help="Show detected ID but skip renaming")
    p.add_argument("--min-confidence", type=float, default=65.0, help="Only rename if ≥ this confidence")
    args = p.parse_args()

    rots = [int(x.strip()) for x in args.try_rotations.split(',') if x.strip()]
    center_first = not args.no_center_first

    best_id, conf, dbg = extract_whiteboard_id(
        args.video, args.seconds, args.sample_fps, args.pattern, args.tesseract_cmd,
        rots, args.band_ratio, center_first, args.psm, args.oem, args.warp_width,
        args.scale, args.gamma, args.clahe, args.sharpen, args.max_quads,
        args.min_area_ratio, args.min_aspect, args.save_crops, args.save_annotated)

    if best_id is None:
        print("⚠️ No ID found.")
        return

    print(f"✅ ID: {best_id} (avg conf: {conf:.1f})")
    print(f"Frames checked: {dbg['frames_checked']}")
    print("Candidates:")
    for k, v in sorted(dbg["candidates"].items(), key=lambda kv: (-kv[1]["votes"], -kv[1]["avg_conf"])):
        print(f"  {k}: votes={v['votes']}, avg_conf={v['avg_conf']:.1f}")

    # --- AUTO-RENAME ---
    src = Path(args.video)
    dst = src.with_name(f"{best_id}.360")
    if conf < args.min_confidence:
        print(f"⚠️ Confidence {conf:.1f}% below threshold ({args.min_confidence}%) → skipping rename.")
        return
    if args.dry_run:
        print(f"💡 Dry-run: would rename {src.name} → {dst.name}")
        return
    if dst.exists():
        print(f"⚠️ Skipped rename: {dst.name} already exists.")
    else:
        shutil.move(src, dst)
        print(f"🎯 Renamed file: {src.name} → {dst.name}")

if __name__ == "__main__":
    main()

