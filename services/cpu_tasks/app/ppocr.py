#!/usr/bin/env python
"""
PP-OCRv6 (small) det + rec ONNX pipeline.

Binds:
  - PaddlePaddle/PP-OCRv6_small_det_onnx  -> finds every text region (DB detector)
  - PaddlePaddle/PP-OCRv6_small_rec_onnx  -> reads the text in each region (CTC recognizer)

Outputs, for each input image, into --out (default ./output):
  <name>_boxes.jpg   original with detected boxes drawn
  <name>_copy.png    "intact copy": recognized text redrawn at the original positions
  <name>_sbs.png     side-by-side original vs copy
  <name>.txt         plain text, reading order
  <name>.json        boxes + text + confidence

Usage:
  python ocr_pipeline.py invoice.jpg
  python ocr_pipeline.py scans/ --limit-side 1600 --min-conf 0.3
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import pyclipper
import yaml
from PIL import Image, ImageDraw, ImageFont

import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
# Vendored from OCR/handoff/reference_pipeline.py — the ground-truth pipeline.
# Only these four path constants are changed: the models live on the RunPod
# network volume rather than beside this file, so the image stays small and the
# weights survive a rebuild. Everything below is untouched, deliberately: the
# spec calls this file the oracle, and the pre/post-processing constants are
# what make or break the port.
OCR_MODEL_DIR = os.environ.get("OCR_MODEL_DIR") or os.path.join(HERE, "models")
DET_MODEL = os.environ.get("OCR_DET_MODEL") or os.path.join(OCR_MODEL_DIR, "ppocrv6_det.onnx")
REC_MODEL = os.environ.get("OCR_REC_MODEL") or os.path.join(OCR_MODEL_DIR, "ppocrv6_rec.onnx")
REC_CONFIG = os.environ.get("OCR_REC_CONFIG") or os.path.join(OCR_MODEL_DIR, "ppocrv6_rec.config.yml")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def make_session(model_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# Detection (DB)                                                              #
# --------------------------------------------------------------------------- #
class TextDetector:
    """PP-OCRv6_small_det with DB postprocess (thresh/box_thresh/unclip from inference.yml)."""

    def __init__(self, model_path, limit_side_len=1280,
                 thresh=0.2, box_thresh=0.45, unclip_ratio=1.4, max_candidates=3000):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.limit_side_len = limit_side_len
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resize(self, img):
        h, w = img.shape[:2]
        ratio = 1.0
        if max(h, w) > self.limit_side_len:
            ratio = self.limit_side_len / max(h, w)
        rh = max(int(round(h * ratio / 32) * 32), 32)
        rw = max(int(round(w * ratio / 32) * 32), 32)
        resized = cv2.resize(img, (rw, rh))
        return resized, rh / h, rw / w

    @staticmethod
    def _poly_area_perimeter(box):
        pts = np.asarray(box, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perimeter = float(np.linalg.norm(pts - np.roll(pts, -1, axis=0),
                                         axis=1).sum())
        return area, perimeter

    def _unclip(self, box):
        area, perimeter = self._poly_area_perimeter(box)
        if perimeter == 0:
            return None
        distance = area * self.unclip_ratio / perimeter
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.astype(np.int64).tolist(),
                       pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0])

    @staticmethod
    def _box_score(bitmap, box):
        h, w = bitmap.shape
        xmin = np.clip(int(box[:, 0].min()), 0, w - 1)
        xmax = np.clip(int(math.ceil(box[:, 0].max())), 0, w - 1)
        ymin = np.clip(int(box[:, 1].min()), 0, h - 1)
        ymax = np.clip(int(math.ceil(box[:, 1].max())), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        shifted = box.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def _min_area_box(contour):
        rect = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
        if points[0][1] <= points[1][1]:
            i0, i3 = 0, 1
        else:
            i0, i3 = 1, 0
        if points[2][1] <= points[3][1]:
            i1, i2 = 2, 3
        else:
            i1, i2 = 3, 2
        box = np.array([points[i0], points[i1], points[i2], points[i3]],
                       dtype=np.float32)
        return box, min(rect[1])

    def __call__(self, img_bgr):
        orig_h, orig_w = img_bgr.shape[:2]
        resized, ratio_h, ratio_w = self._resize(img_bgr)
        blob = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        prob_map = self.session.run(None, {self.input_name: blob})[0][0, 0]

        binary = (prob_map > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[:self.max_candidates]:
            box, min_side = self._min_area_box(contour)
            if min_side < 3:
                continue
            if self._box_score(prob_map, box) < self.box_thresh:
                continue
            expanded = self._unclip(box)
            if expanded is None or len(expanded) == 0:
                continue
            box, min_side = self._min_area_box(expanded.reshape(-1, 1, 2))
            if min_side < 5:
                continue
            box[:, 0] = np.clip(box[:, 0] / ratio_w, 0, orig_w - 1)
            box[:, 1] = np.clip(box[:, 1] / ratio_h, 0, orig_h - 1)
            boxes.append(box)
        return sort_boxes(boxes)


def sort_boxes(boxes):
    """Top-to-bottom, then left-to-right for boxes on roughly the same line."""
    boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and \
                    boxes[j + 1][0][0] < boxes[j][0][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


# --------------------------------------------------------------------------- #
# Recognition (CTC)                                                           #
# --------------------------------------------------------------------------- #
class TextRecognizer:
    """PP-OCRv6_small_rec: 48px-high crops, dynamic width, CTC decode."""

    def __init__(self, model_path, config_path, batch_size=8, max_width=3200):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.batch_size = batch_size
        self.height = 48
        self.max_width = max_width
        self.charset = self._load_charset(config_path)

    def _load_charset(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        chars = [str(c) for c in config["PostProcess"]["character_dict"]]
        num_classes = self.session.get_outputs()[0].shape[-1]
        charset = ["blank"] + chars
        if isinstance(num_classes, int):
            if num_classes == len(chars) + 2:
                charset.append(" ")
            elif num_classes != len(chars) + 1:
                raise ValueError(
                    f"model classes ({num_classes}) do not match "
                    f"dictionary size ({len(chars)})")
        else:  # dynamic output dim: assume space char, verified on first run
            charset.append(" ")
        return charset

    def _resize_norm(self, img, target_w):
        h, w = img.shape[:2]
        scaled_w = min(target_w, max(1, int(math.ceil(self.height * w / h))))
        resized = cv2.resize(img, (scaled_w, self.height)).astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5
        padded = np.zeros((self.height, target_w, 3), dtype=np.float32)
        padded[:, :scaled_w] = resized
        return padded.transpose(2, 0, 1)

    def _decode(self, logits):
        texts = []
        for probs in logits:
            idxs = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            chars, char_confs = [], []
            prev = 0
            for t, idx in enumerate(idxs):
                if idx != 0 and idx != prev:
                    if idx < len(self.charset):
                        chars.append(self.charset[idx])
                        char_confs.append(confs[t])
                prev = idx
            text = "".join(chars)
            conf = float(np.mean(char_confs)) if char_confs else 0.0
            texts.append((text, conf))
        return texts

    def __call__(self, crops):
        if not crops:
            return []
        ratios = [c.shape[1] / c.shape[0] for c in crops]
        order = np.argsort(ratios)
        results = [None] * len(crops)
        for start in range(0, len(crops), self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            max_ratio = max(max(ratios[i] for i in batch_idx), 320 / 48)
            target_w = min(int(math.ceil(self.height * max_ratio)),
                           self.max_width)
            blob = np.stack([self._resize_norm(crops[i], target_w)
                             for i in batch_idx])
            logits = self.session.run(None, {self.input_name: blob})[0]
            for i, res in zip(batch_idx, self._decode(logits)):
                results[i] = res
        return results


def rotate_crop(img, box):
    """Perspective-crop a quad; rotate upright if the crop is vertical."""
    box = box.astype(np.float32)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, matrix, (w, h),
                               borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    if h > 0 and w > 0 and h / w >= 1.5:
        crop = np.rot90(crop, k=3)
    return crop


# --------------------------------------------------------------------------- #
# Reconstruction ("intact copy")                                              #
# --------------------------------------------------------------------------- #
FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _find_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def render_copy(size_wh, items):
    """Redraw recognized text on a white canvas at the detected positions."""
    canvas = Image.new("RGB", size_wh, "white")
    draw = ImageDraw.Draw(canvas)
    font_path = _find_font()
    for item in items:
        box = np.array(item["box"])
        x, y = box[:, 0].min(), box[:, 1].min()
        box_h = np.linalg.norm(box[0] - box[3])
        box_w = np.linalg.norm(box[1] - box[0])
        text = item["text"]
        if not text:
            continue
        font_size = max(int(box_h * 0.8), 8)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            # shrink until the text fits the detected width
            while font_size > 8 and draw.textlength(text, font=font) > box_w * 1.05:
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        draw.text((x, y), text, fill="black", font=font)
    return canvas


def draw_boxes(img_bgr, items):
    vis = img_bgr.copy()
    for item in items:
        pts = np.array(item["box"], dtype=np.int32)
        color = (0, 180, 0) if item["confidence"] >= 0.8 else (0, 120, 255)
        cv2.polylines(vis, [pts], True, color, 2)
    return vis


def side_by_side(img_bgr, copy_img):
    orig = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    gap = 20
    total = Image.new("RGB", (orig.width * 2 + gap, orig.height), "gray")
    total.paste(orig, (0, 0))
    total.paste(copy_img.resize(orig.size), (orig.width + gap, 0))
    return total


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def process_image(path, detector, recognizer, out_dir, min_conf):
    img = cv2.imread(path)
    if img is None:
        print(f"  !! could not read {path}")
        return None
    name = os.path.splitext(os.path.basename(path))[0]

    boxes = detector(img)
    crops = [rotate_crop(img, box) for box in boxes]
    recognized = recognizer(crops)

    items = []
    for box, (text, conf) in zip(boxes, recognized):
        if text and conf >= min_conf:
            items.append({"box": box.round(1).tolist(),
                          "text": text,
                          "confidence": round(conf, 4)})

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{name}_boxes.jpg"),
                draw_boxes(img, items))
    copy_img = render_copy((img.shape[1], img.shape[0]), items)
    copy_img.save(os.path.join(out_dir, f"{name}_copy.png"))
    side_by_side(img, copy_img).save(os.path.join(out_dir, f"{name}_sbs.png"))
    with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(item["text"] for item in items))
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"image": os.path.basename(path), "items": items},
                  f, ensure_ascii=False, indent=2)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input", help="image file or folder of images")
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    parser.add_argument("--limit-side", type=int, default=1280,
                        help="max side length fed to the detector "
                             "(raise to 1600-2000 for very small text)")
    parser.add_argument("--min-conf", type=float, default=0.30,
                        help="drop recognitions below this confidence")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in IMG_EXTS)
    else:
        paths = [args.input]
    if not paths:
        sys.exit("no images found")

    print("loading models...")
    detector = TextDetector(DET_MODEL, limit_side_len=args.limit_side)
    recognizer = TextRecognizer(REC_MODEL, REC_CONFIG)
    print(f"charset size: {len(recognizer.charset)}")

    for path in paths:
        print(f"\n== {path}")
        items = process_image(path, detector, recognizer, args.out, args.min_conf)
        if items is None:
            continue
        print(f"   {len(items)} text lines recognized -> {args.out}/")
        for item in items:
            print(f"   [{item['confidence']:.2f}] {item['text']}")


if __name__ == "__main__":
    main()
