"""
YOLOv8 helpers: load a fine-tuned checkpoint, filter detections, crop regions for downstream VLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
from ultralytics import YOLO

PathLike = Union[str, Path]

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "best.pt"

_model: Optional[YOLO] = None
_model_weights: Optional[str] = None


def reset_model_cache() -> None:
    """Drop the cached model (e.g. after swapping weights)."""
    global _model, _model_weights
    _model = None
    _model_weights = None


def get_yolo(weights_path: PathLike = _DEFAULT_WEIGHTS) -> YOLO:
    """Load and cache a YOLO model from ``weights_path``."""
    global _model, _model_weights
    resolved = str(Path(weights_path).expanduser().resolve())
    if _model is None or _model_weights != resolved:
        _model = YOLO(resolved)
        _model_weights = resolved
    return _model


def get_class_names(weights_path: PathLike = _DEFAULT_WEIGHTS) -> Dict[int, str]:
    """Return ``class_id -> name`` for the checkpoint (from training metadata)."""
    names = get_yolo(weights_path).names
    return {int(k): str(v) for k, v in names.items()}


def _resolve_class_indices(
    id_to_name: Dict[int, str],
    class_ids: Optional[Sequence[int]],
    name_substrings: Optional[Sequence[str]],
) -> Optional[List[int]]:
    """
    Choose which class indices to run.

    - ``class_ids`` wins if provided (empty sequence => run nothing).
    - Else if ``name_substrings`` is set, match case-insensitive substrings in class names.
    - Else ``None`` = all classes.
    """
    if class_ids is not None:
        return [int(i) for i in class_ids]
    if name_substrings:
        lowered = [s.lower() for s in name_substrings]
        out: List[int] = []
        for cid, name in id_to_name.items():
            nl = name.lower()
            if any(sub in nl for sub in lowered):
                out.append(cid)
        if not out:
            raise ValueError(
                f"No class names matched {list(name_substrings)!r}. "
                f"Available classes: {id_to_name}"
            )
        return out
    return None


def _expand_xyxy(
    xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
    pad_frac: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    px = bw * pad_frac
    py = bh * pad_frac
    nx1 = max(0, int(x1 - px))
    ny1 = max(0, int(y1 - py))
    nx2 = min(width, int(x2 + px))
    ny2 = min(height, int(y2 + py))
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _prepare_predict(
    path: Path,
    weights_path: PathLike,
    conf: float,
    iou: float,
    imgsz: int,
    class_ids: Optional[Sequence[int]],
    class_name_substrings: Optional[Sequence[str]],
) -> Tuple[Optional[YOLO], Optional[Dict[str, Any]], Dict[int, str]]:
    """
    Shared Ultralytics ``predict`` kwargs. Returns ``(None, None, names)`` when
    ``class_ids`` is an empty sequence (nothing to run).
    """
    model = get_yolo(weights_path)
    id_to_name = get_class_names(weights_path)
    classes = _resolve_class_indices(id_to_name, class_ids, class_name_substrings)
    if classes is not None and len(classes) == 0:
        return None, None, id_to_name
    kwargs: Dict[str, Any] = {
        "source": str(path),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if classes is not None:
        kwargs["classes"] = classes
    return model, kwargs, id_to_name


def count_boxes_at_conf(
    image_path: PathLike,
    *,
    weights_path: PathLike = _DEFAULT_WEIGHTS,
    conf: float = 0.001,
    iou: float = 0.45,
    imgsz: int = 640,
    class_ids: Optional[Sequence[int]] = None,
    class_name_substrings: Optional[Sequence[str]] = None,
) -> Tuple[int, float]:
    """
    Run a single forward pass at low ``conf``. Returns ``(num_boxes, max_score)``
    (``max_score`` is ``0.0`` if there are no boxes).
    """
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model, kwargs, _ = _prepare_predict(
        path, weights_path, conf, iou, imgsz, class_ids, class_name_substrings
    )
    if kwargs is None:
        return 0, 0.0
    results = model.predict(**kwargs)
    if not results:
        return 0, 0.0
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return 0, 0.0
    m = float(r0.boxes.conf.cpu().numpy().max())
    return len(r0.boxes), m


def save_prediction_plot(
    image_path: PathLike,
    out_path: PathLike,
    *,
    weights_path: PathLike = _DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    class_ids: Optional[Sequence[int]] = None,
    class_name_substrings: Optional[Sequence[str]] = None,
) -> bool:
    """
    Save an annotated image (boxes + labels) using Ultralytics' plotter.
    Returns ``False`` if inference was skipped (empty ``class_ids`` filter).
    """
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model, kwargs, _ = _prepare_predict(
        path, weights_path, conf, iou, imgsz, class_ids, class_name_substrings
    )
    if kwargs is None:
        return False
    r0 = model.predict(**kwargs)[0]
    bgr = r0.plot()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(out)
    return True


@dataclass(frozen=True)
class PlasticCropResult:
    """One cropped region aligned with a single detection."""

    crop: Image.Image
    class_id: int
    class_name: str
    confidence: float
    detection_xyxy: Tuple[float, float, float, float]
    crop_xyxy: Tuple[int, int, int, int]

    def save(self, path: PathLike) -> Path:
        """Write the crop to disk; returns the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.crop.convert("RGB").save(p)
        return p.resolve()


def crop_plastic_regions(
    image_path: PathLike,
    *,
    weights_path: PathLike = _DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    pad_frac: float = 0.05,
    class_ids: Optional[Sequence[int]] = None,
    class_name_substrings: Optional[Sequence[str]] = None,
    max_detections: Optional[int] = None,
) -> List[PlasticCropResult]:
    """
    Run YOLO on ``image_path``, optionally restrict classes, return padded crops.

    ``class_name_substrings`` matches substrings in each class name (case-insensitive), e.g.
    ``("plastic", "cup", "bottle")``. Raises ``ValueError`` if none match. Pass explicit
    ``class_ids`` to force specific IDs; an empty ``class_ids`` returns no crops without inference.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    model, kwargs, id_to_name = _prepare_predict(
        path, weights_path, conf, iou, imgsz, class_ids, class_name_substrings
    )
    if kwargs is None:
        return []

    results = model.predict(**kwargs)
    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    im = Image.open(path).convert("RGB")
    w, h = im.size

    boxes = r0.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    out: List[PlasticCropResult] = []
    order = confs.argsort()[::-1]
    for idx in order:
        i = int(idx)
        det = (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]))
        cid = int(clss[i])
        cname = id_to_name.get(cid, str(cid))
        cxy = _expand_xyxy(det, w, h, pad_frac)
        crop = im.crop(cxy)
        out.append(
            PlasticCropResult(
                crop=crop,
                class_id=cid,
                class_name=cname,
                confidence=float(confs[i]),
                detection_xyxy=det,
                crop_xyxy=cxy,
            )
        )
        if max_detections is not None and len(out) >= max_detections:
            break

    return out


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Test YOLO crop pipeline: print class names and save crops.")
    p.add_argument("image", type=Path, help="Input image path")
    p.add_argument(
        "--weights",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="YOLO weights (.pt)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("crop_test_out"),
        help="Directory to write crop JPEGs",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Min confidence for crops (try 0.05–0.15 if you get 0 detections)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size (try 1280 for small objects)",
    )
    p.add_argument(
        "--class-substr",
        nargs="*",
        metavar="STR",
        help="Optional: class name substrings (case-insensitive); labels are Others, PC, PE, PET, PP, PS",
    )
    p.add_argument(
        "--preview",
        type=Path,
        metavar="PATH",
        help="Save an annotated full-frame image (boxes/labels) for debugging",
    )
    args = p.parse_args()
    if not args.image.is_file():
        print(f"Not a file: {args.image}", file=sys.stderr)
        sys.exit(1)
    names = get_class_names(args.weights)
    print("Checkpoint classes:", names)
    csub = args.class_substr if args.class_substr else None
    try:
        hits = crop_plastic_regions(
            args.image,
            weights_path=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            class_name_substrings=csub,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    print(f"Detections used for crops: {len(hits)}")
    if args.preview is not None:
        ok = save_prediction_plot(
            args.image,
            args.preview,
            weights_path=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            class_name_substrings=csub,
        )
        if ok:
            print(f"Saved preview: {args.preview.resolve()}")
        else:
            print("Preview skipped (empty class filter).", file=sys.stderr)
    if len(hits) == 0:
        n_low, mx = count_boxes_at_conf(
            args.image,
            weights_path=args.weights,
            conf=0.001,
            imgsz=args.imgsz,
            class_name_substrings=csub,
        )
        if n_low > 0:
            suggest = max(0.01, min(mx * 0.85, 0.99))
            print(
                f"Hint: at conf=0.001 there are {n_low} raw box(es); max score {mx:.4f}. "
                f"Try: --conf {suggest:.3f} (and/or --imgsz 1280)."
            )
        else:
            print(
                "Hint: still 0 boxes at conf=0.001 — this frame may be off-distribution for "
                "the finetuned resin-type detector, or try a different image / weights."
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    for i, h in enumerate(hits):
        out_path = args.out_dir / f"{stem}_crop_{i:02d}_{h.class_id}.jpg"
        h.save(out_path)
        print(
            f"  {out_path}  cls={h.class_name!r}  conf={h.confidence:.3f}  "
            f"det={tuple(round(x, 1) for x in h.detection_xyxy)}"
        )
