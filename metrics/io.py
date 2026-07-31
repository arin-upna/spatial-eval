"""Minimal LAS/LAZ input helpers used by the evaluation pipeline."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _resolve_dimension(las, requested: str) -> str:
    """Resolve a LAS dimension name while preserving custom-field casing."""
    available = list(las.point_format.dimension_names)
    available.extend(las.point_format.extra_dimension_names)
    lookup = {name.lower(): name for name in available}

    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise KeyError(
            f"Dimension '{requested}' is not present in the point cloud. "
            f"Available dimensions: {sorted(set(available))}"
        )
    return resolved


def load_point_cloud(
    file_path,
    features: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load selected dimensions from a LAS or LAZ point cloud.

    Coordinates are returned as scaled floating-point values in the coordinate
    reference system units. The distance metrics therefore assume a projected
    CRS whose unit is the meter.

    Args:
        file_path: Input ``.las`` or ``.laz`` path.
        features: Requested fields. ``"coords"`` returns an ``(N, 3)`` XYZ
            array. Other names can be standard LAS dimensions or extra fields,
            such as ``"classification"`` or ``"Prediction"``.

    Returns:
        Dictionary mapping requested feature names to NumPy arrays.
    """
    path = Path(file_path)
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Unsupported point-cloud format: {path.suffix}")

    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "LAS/LAZ evaluation requires laspy. Install the project with "
            "`pip install -e .`."
        ) from exc

    las = laspy.read(path)
    requested_features = list(features or ("coords", "classification"))
    result: Dict[str, np.ndarray] = {}

    for feature in requested_features:
        if feature == "coords":
            result["coords"] = np.column_stack(
                (
                    np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64),
                )
            )
            continue

        resolved = _resolve_dimension(las, feature)
        result[feature] = np.asarray(las[resolved])

    return result
