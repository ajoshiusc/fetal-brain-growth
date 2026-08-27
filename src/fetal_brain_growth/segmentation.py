"""FetalSynthSeg v1-compatible inference with provenance and checksum checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np

try:
    import monai
    import torch
    from monai.networks.nets import UNet
except ImportError as error:  # pragma: no cover - exercised only without optional extra
    raise ImportError("Install segmentation dependencies with: pip install -e '.[segmentation]'") from error

from .labels import FETA_LABELS
from .radiology import save_radiology_figure


OFFICIAL_CHECKPOINT_SHA256 = "d991edbb1295797f59973867ec3724ba6ebaddb44387f09da38bc5ab98b2fe52"
OFFICIAL_REPOSITORY = "https://github.com/Medical-Image-Analysis-Laboratory/FetalSynthSeg"
OFFICIAL_COMMIT = "03c439edef02fc830e31a38169c5aa09ca98eeb4"


@dataclass
class PreparedImage:
    tensor: torch.Tensor
    original: nib.spatialimages.SpatialImage
    resampled: nib.spatialimages.SpatialImage
    bbox: tuple[slice, slice, slice]
    crop_shape: tuple[int, int, int]
    pad_before: tuple[int, int, int]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model() -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act="leakyrelu",
        norm="instance",
        dropout=0.1,
    )


def load_model(checkpoint: str | Path, device: torch.device, *, verify_checksum: bool = True) -> tuple[UNet, str]:
    checkpoint = Path(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if verify_checksum and checkpoint.name == "KISPI-all_fss.ckpt" and checkpoint_hash != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint checksum mismatch; do not run an unverified pickle checkpoint.")
    # The official Lightning checkpoint is pickle-based. Its exact SHA-256 is
    # checked above before this trusted deserialization step.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("net."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("net.")
    }
    model = build_model()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint_hash


def _foreground_bbox(data: np.ndarray) -> tuple[slice, slice, slice]:
    foreground = np.abs(np.nan_to_num(data, copy=False)) > 0
    if not foreground.any():
        raise ValueError("Input image has no non-zero foreground.")
    coordinates = np.where(foreground)
    return tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)  # type: ignore[return-value]


def prepare_image(
    input_path: str | Path,
    *,
    target_spacing_mm: float = 0.5,
    spatial_size: int = 256,
) -> PreparedImage:
    original = nib.load(str(input_path))
    if len(original.shape) != 3:
        raise ValueError(f"Input MRI must be 3-D; got shape {original.shape}.")
    canonical = nib.as_closest_canonical(original)
    resampled = resample_to_output(
        canonical,
        voxel_sizes=(target_spacing_mm,) * 3,
        order=1,
        mode="constant",
        cval=0.0,
    )
    data = np.nan_to_num(resampled.get_fdata(dtype=np.float32), copy=False)
    bbox = _foreground_bbox(data)
    crop = data[bbox]
    crop_shape = tuple(int(value) for value in crop.shape)
    if any(value > spatial_size for value in crop_shape):
        raise ValueError(
            f"Foreground at {target_spacing_mm:g} mm is {crop_shape}, exceeding {spatial_size}^3. "
            "FetalSynthSeg expects a skull-stripped/cropped SVR brain."
        )
    pad_before = tuple((spatial_size - value) // 2 for value in crop_shape)
    pad_after = tuple(spatial_size - value - before for value, before in zip(crop_shape, pad_before))
    padded = np.pad(crop, tuple(zip(pad_before, pad_after)), mode="constant")
    low, high = float(padded.min()), float(padded.max())
    if high <= low:
        raise ValueError("Input MRI has no usable intensity range.")
    padded = (padded - low) / (high - low)
    tensor = torch.from_numpy(padded.astype(np.float32, copy=False))[None, None]
    return PreparedImage(tensor, original, resampled, bbox, crop_shape, pad_before)


def restore_segmentation(prediction: np.ndarray, prepared: PreparedImage) -> nib.Nifti1Image:
    unpad = tuple(slice(before, before + size) for before, size in zip(prepared.pad_before, prepared.crop_shape))
    cropped_prediction = prediction[unpad]
    resampled_labels = np.zeros(prepared.resampled.shape, dtype=np.int16)
    resampled_labels[prepared.bbox] = cropped_prediction.astype(np.int16, copy=False)
    restored = resample_from_to(
        nib.Nifti1Image(resampled_labels, prepared.resampled.affine),
        (prepared.original.shape, prepared.original.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    labels = np.rint(restored.get_fdata()).astype(np.int16)
    header = prepared.original.header.copy()
    header.set_data_dtype(np.int16)
    return nib.Nifti1Image(labels, prepared.original.affine, header)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def segment_image(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path,
    *,
    device_name: str = "auto",
    qc_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, object]:
    device = resolve_device(device_name)
    started = time.perf_counter()
    prepared = prepare_image(input_path)
    model, checkpoint_hash = load_model(checkpoint, device, verify_checksum=verify_checksum)
    with torch.inference_mode():
        tensor = prepared.tensor.to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(tensor).argmax(dim=1)[0]
        else:
            prediction = model(tensor).argmax(dim=1)[0]
    output_image = restore_segmentation(prediction.cpu().numpy(), prepared)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output_path))
    if qc_path is not None:
        save_radiology_figure(input_path, output_path, qc_path)
    metadata = {
        "input": str(Path(input_path).resolve()),
        "output": str(output_path.resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "labels": FETA_LABELS,
        "intended_use": "Research use only; visual QC and expert review required.",
    }
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official FetalSynthSeg checkpoint.")
    parser.add_argument("--input", required=True, help="Skull-stripped/cropped 3-D T2w SVR NIfTI")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--qc")
    parser.add_argument("--metadata")
    parser.add_argument("--skip-checksum", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = segment_image(
        args.input,
        args.output,
        args.checkpoint,
        device_name=args.device,
        qc_path=args.qc,
        metadata_path=args.metadata,
        verify_checksum=not args.skip_checksum,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
