"""Etapa 5 do pipeline MLOps: contrato `evaluate` consumido por mlops.compare.

Uso (bundle unico):
    python -m evaluation.evaluate --bundle <dir> --manifest <json> \
        --conf-threshold <float> --json

Saida: JSON puro na ultima linha do stdout, com as chaves map50_95, map50,
precision, recall, n_images. Linhas anteriores (log do Ultralytics) sao
ignoradas por quem chama (mlops.compare le so a ultima linha do stdout).

Uso (lote, rodado dentro de um job Slurm por mlops.evaluate_job):
    python -m evaluation.evaluate --spec <spec.json> --out <out.json>

`spec.json`: lista de itens `{"label": str, "bundle": str, "manifest": str,
"conf_threshold": float, "split": str (opcional, default "val")}`. Cada item
e avaliado em processo (sem subprocess aninhado) e o resultado e gravado em
`out.json` como `{label: metrics}`.

"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

SUPPORTED_ARCHITECTURES = ("yolov8s",)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

MAP_CONF = 0.001
NMS_IOU = 0.70
GT_MATCH_IOU = 0.50
PREDICT_CHUNK_SIZE = 64


class EvaluateError(RuntimeError):
    pass


def load_bundle_metadata(bundle: Path) -> dict:
    meta_path = bundle / "metadata.yaml"
    if not meta_path.exists():
        raise EvaluateError(f"bundle sem metadata.yaml: {bundle}")
    return yaml.safe_load(meta_path.read_text(encoding="utf-8"))


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def resolve_image_path(source_dir: Path, item_path: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = source_dir / "images" / f"{item_path}{ext}"
        if candidate.exists():
            return candidate
    raise EvaluateError(f"imagem nao encontrada para '{item_path}' em {source_dir}")


def resolve_label_path(source_dir: Path, item_path: str) -> Path:
    return source_dir / "labels" / f"{item_path}.txt"


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_label_to_xyxy(
    cls: int, xc: float, yc: float, w: float, h: float, width: int, height: int
) -> tuple[int, tuple[float, float, float, float]]:
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    return cls, (x1, y1, x2, y2)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[int, tuple[float, float, float, float]]],
) -> tuple[int, int, int]:
    """Casa deteccoes com GT por classe, gulosamente por confianca desc. Retorna (tp, fp, fn)."""
    matched_gt: set[int] = set()
    tp = fp = 0

    for cls, _conf, box in sorted(predictions, key=lambda p: p[1], reverse=True):
        best_iou = 0.0
        best_idx = None
        for idx, (gt_cls, gt_box) in enumerate(ground_truths):
            if idx in matched_gt or gt_cls != cls:
                continue
            iou = iou_xyxy(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= GT_MATCH_IOU:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def _write_split_data_yaml(source_dir: Path, class_indices: dict, split: str, out_dir: Path) -> Path:
    names = {index: name for name, index in class_indices.items()}
    split_dir = str(source_dir / "images" / split)
    payload = {"train": split_dir, "val": split_dir, "names": names}
    out = out_dir / "eval_data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return out


def evaluate_yolov8s(
    bundle: Path, metadata: dict, manifest: dict, conf_threshold: float, split: str = "val"
) -> dict:
    from ultralytics import YOLO

    weights = bundle / metadata["weights"]
    if not weights.is_file():
        raise EvaluateError(f"pesos nao encontrados: {weights}")

    imgsz = int(metadata["imgsz"])
    class_indices = metadata["class_indices"]
    source_dir = Path(manifest["source_dir"])

    if split not in manifest["splits"]:
        raise EvaluateError(
            f"manifesto sem split '{split}' (tem: {', '.join(manifest['splits'])})"
        )
    items = manifest["splits"][split]["items"]
    if not items:
        raise EvaluateError(f"manifesto sem itens no split '{split}'")

    pairs = [
        (resolve_image_path(source_dir, item["path"]), resolve_label_path(source_dir, item["path"]))
        for item in items
    ]

    model = YOLO(str(weights))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # map50/map50_95: curva de confianca completa, padrao de framework,
        # independente do limiar pedido (mesma logica de evaluate_yolov8s.py).
        data_yaml = _write_split_data_yaml(source_dir, class_indices, split, tmp_path)
        val_results = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            conf=MAP_CONF,
            iou=NMS_IOU,
            plots=False,
            verbose=False,
            save_json=False,
            project=str(tmp_path),
            name="val",
            batch=16,
        )
        map50_95 = float(val_results.box.map)
        map50 = float(val_results.box.map50)

        # precision/recall no limiar pedido, aplicado igualmente: correspondencia
        # guloso por IoU sobre deteccoes com confianca >= conf_threshold.
        #
        # Fatiado manualmente em lotes pequenos (PREDICT_CHUNK_SIZE): passar
        # a lista inteira de uma vez pro predict() ("source=[...]", mesmo com
        # stream=True e batch=1) faz o Ultralytics empilhar todas as imagens
        # num unico tensor uint8 antes do primeiro ".to(device)" quando elas
        # tem o mesmo tamanho (como patches 640x640) — confirmado na pratica:
        # OOM tentando alocar exatamente altura*largura*3*N_imagens bytes de
        # uma vez, um golden_manifest real facilmente tem dezenas de
        # milhares de itens. `stream=True`/`batch=1` nao evitam isso, so o
        # fatiamento em Python evita.
        total_tp = total_fp = total_fn = 0
        for chunk_start in range(0, len(pairs), PREDICT_CHUNK_SIZE):
            chunk = pairs[chunk_start:chunk_start + PREDICT_CHUNK_SIZE]
            predict_results = model.predict(
                source=[str(image_path) for image_path, _ in chunk],
                imgsz=imgsz,
                conf=conf_threshold,
                iou=NMS_IOU,
                verbose=False,
            )

            for (_image_path, label_path), result in zip(chunk, predict_results):
                height, width = result.orig_shape

                ground_truths = [
                    yolo_label_to_xyxy(cls, xc, yc, w, h, width, height)
                    for cls, xc, yc, w, h in read_yolo_labels(label_path)
                ]

                predictions = []
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    for cls, conf, xyxy in zip(
                        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                    ):
                        predictions.append((int(cls), float(conf), tuple(xyxy)))

                tp, fp, fn = match_predictions(predictions, ground_truths)
                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    return {
        "map50_95": round(map50_95, 6),
        "map50": round(map50, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "n_images": len(pairs),
    }


DISPATCH = {
    "yolov8s": evaluate_yolov8s,
}


def evaluate(
    bundle: Path, manifest_path: Path, conf_threshold: float, split: str = "val"
) -> dict:
    metadata = load_bundle_metadata(bundle)
    architecture = metadata.get("architecture")

    handler = DISPATCH.get(architecture)
    if handler is None:
        raise EvaluateError(
            f"arquitetura '{architecture}' sem avaliador implementado. "
            f"Suportadas: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )

    manifest = load_manifest(manifest_path)
    return handler(bundle, metadata, manifest, conf_threshold, split=split)


def evaluate_spec(spec: list[dict]) -> dict[str, dict]:
    """Avalia cada item do lote em processo. Nao para no primeiro erro:
    um item com problema fica ausente do resultado, mas os demais completam
    (evita perder um job Slurm inteiro por um bundle/manifesto ruim).

    `item["split"]` e opcional, default "val" — permite apontar o
    generalization_manifest pra qualquer dataset sem depender do nome da
    pasta interna (ex: um dataset que so tem "test", nao "val")."""
    results = {}
    for item in spec:
        label = item["label"]
        try:
            results[label] = evaluate(
                Path(item["bundle"]), Path(item["manifest"]), float(item["conf_threshold"]),
                split=item.get("split", "val"),
            )
        except EvaluateError as exc:
            results[label] = {"error": str(exc)}
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avalia um bundle contra um manifesto (etapa 5 do pipeline MLOps)."
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--conf-threshold", type=float)
    parser.add_argument("--split", default="val", help="split do manifesto a avaliar (default: val)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--spec", type=Path, help="lote: JSON com lista de itens")
    parser.add_argument("--out", type=Path, help="lote: onde gravar o resultado")
    args = parser.parse_args(argv)

    if args.spec:
        if not args.out:
            parser.error("--spec requer --out")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        results = evaluate_spec(spec)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results))
        return 0

    if not args.bundle or not args.manifest or args.conf_threshold is None:
        parser.error("--bundle/--manifest/--conf-threshold sao obrigatorios sem --spec")

    metrics = evaluate(args.bundle, args.manifest, args.conf_threshold, split=args.split)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
