"""
Smoke test for purkinje-uv on real patient cut-surface meshes (LV/RV OBJ).

This script intentionally favors simple heuristics so we can confirm the
end-to-end pipeline runs on realistic data before running the full workflow.

What it does:
1) Loads LV and RV cut-surface OBJ files.
2) Chooses init/second nodes automatically (PCA-based) unless you provide them.
3) Builds FractalTreeParameters with scale heuristics based on mesh size.
4) Grows trees with shrink-and-retry on "out of domain" errors.
5) Validates outputs (shapes, index ranges, non-empty PMJs, finite coords).
6) Saves LV/RV VTU + (optional) PMJ VTP.

Examples:
python scripts/validate_purkinje_uv_smoke.py --lv "S62_..._LVendo_heart_cut.obj" --rv "S62_..._RVendo_heart_cut.obj" --out out_s62

python scripts/validate_purkinje_uv_smoke.py --lv ... --rv ... --lv-seeds 123 456 --rv-seeds 789 790 --out out_s62
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import os
from pathlib import Path
import time

import numpy as np

from purkinje_uv.fractal_tree_parameters import FractalTreeParameters
from purkinje_uv.fractal_tree import FractalTree
from purkinje_uv.purkinje_tree import PurkinjeTree
from purkinje_uv.mesh import Mesh


LOGGER = logging.getLogger(__name__)


def _resolve_log_level(level: str | int) -> int:
    """Translate a log level string into a logging constant."""
    if isinstance(level, int):
        return level
    name = str(level).upper()
    value = getattr(logging, name, None)
    return value if isinstance(value, int) else logging.INFO


def configure_logging(level: str) -> None:
    """Configure console logging for the smoke test."""
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    resolved = _resolve_log_level(level)
    logging.basicConfig(level=resolved, format=fmt, datefmt=datefmt)
    logging.getLogger().setLevel(resolved)
    if not isinstance(getattr(logging, str(level).upper(), None), int):
        LOGGER.warning("Unknown log level %r; defaulting to INFO.", level)


def require(condition: bool, label: str, message: str) -> None:
    """Raise a ValueError with a consistent prefix if a check fails."""
    if not condition:
        raise ValueError(f"{label}: {message}")


def pca_seed_vertices(verts: np.ndarray, k_nn: int = 30) -> tuple[int, int]:
    """
    Pick (init_node_id, second_node_id) from vertices using PCA axis.

    init = extreme on -PC1 (apex-like)
    second = among k nearest neighbors of init, the one with largest projection along +PC1
    """
    require(
        verts.ndim == 2 and verts.shape[1] == 3,
        "PCA",
        f"verts must be (N, 3), got {verts.shape}",
    )
    v = verts.astype(np.float64)
    c = v.mean(axis=0)
    X = v - c

    # PCA via SVD: use PC1 as the dominant long axis.
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    pc1 = vh[0]  # (3,)

    proj = X @ pc1
    init = int(np.argmin(proj))

    # Nearest neighbors of init (brute force is OK for a smoke test).
    d2 = np.sum((v - v[init]) ** 2, axis=1)
    nn = np.argsort(d2)[: max(k_nn, 3)]

    # Choose second as neighbor "going forward" along PC1.
    nn_proj = proj[nn]
    second = int(nn[int(np.argmax(nn_proj))])

    if second == init:
        # Fallback: take 2nd closest distinct.
        for idx in nn:
            if int(idx) != init:
                second = int(idx)
                break

    return init, second


def mesh_size_diameter(verts: np.ndarray) -> float:
    """Return bounding-box diagonal length (rough scale proxy)."""
    require(
        verts.ndim == 2 and verts.shape[1] == 3,
        "MESH",
        f"verts must be (N, 3), got {verts.shape}",
    )
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def validate_seeds(label: str, seeds: tuple[int, int], n_verts: int) -> None:
    """Ensure seed indices are valid for a given mesh."""
    require(len(seeds) == 2, label, f"expected 2 seeds, got {len(seeds)}")
    init, second = seeds
    require(init != second, label, "init and second seeds must differ")
    for name, idx in (("init", init), ("second", second)):
        require(
            0 <= idx < n_verts,
            label,
            f"{name} seed out of range [0, {n_verts - 1}]: {idx}",
        )


def build_params(
    meshfile: Path, seeds: tuple[int, int], diag: float
) -> FractalTreeParameters:
    """
    Heuristic scale based on mesh bounding-box diagonal.
    Values are "reasonable starters"; the retry loop shrinks if needed.
    """
    init, second = seeds

    # Scale heuristics: tune later, but good enough for a smoke test.
    init_length = 0.25 * diag
    length = 0.06 * diag
    l_segment = max(0.01 * diag, length / 12.0)

    params = FractalTreeParameters(
        meshfile=str(meshfile),
        init_node_id=init,
        second_node_id=second,
        init_length=float(init_length),
        N_it=20,
        length=float(length),
        branch_angle=0.15,
        w=0.10,
        l_segment=float(l_segment),
        fascicles_angles=[0.02, 0.03],
        fascicles_length=[0.05 * diag, 0.08 * diag],
    )
    return params


def grow_with_retry(
    params: FractalTreeParameters,
    *,
    label: str,
    max_tries: int = 12,
    shrink: float = 0.85,
) -> tuple[FractalTreeParameters, FractalTree]:
    """
    Grow a tree; if it fails with "out of the domain" shrink lengths and retry.
    """
    p = params
    last_err: Exception | None = None

    for i in range(max_tries):
        try:
            LOGGER.info("[%s] Grow attempt %d/%d", label, i + 1, max_tries)
            ft = FractalTree(p)
            ft.grow_tree()
            LOGGER.info(
                "[%s] Growth succeeded (nodes=%d edges=%d pmj=%d)",
                label,
                len(ft.nodes_xyz),
                len(ft.connectivity),
                len(ft.end_nodes),
            )
            return p, ft  # success
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Legacy algorithm often throws "out of the domain".
            if "out of the domain" in msg or "domain" in msg:
                LOGGER.warning(
                    "[%s] Grow failed (%s); shrinking lengths by %.2f and retrying.",
                    label,
                    e,
                    shrink,
                )
                # Shrink geometric step scales.
                p = replace(
                    p,
                    init_length=float(p.init_length * shrink),
                    length=float(p.length * shrink),
                    l_segment=float(p.l_segment * shrink),
                    fascicles_length=[float(x * shrink) for x in p.fascicles_length],
                )
                LOGGER.debug(
                    "[%s] Shrunk params: init_length=%.6g length=%.6g l_segment=%.6g",
                    label,
                    p.init_length,
                    p.length,
                    p.l_segment,
                )
                continue

            # Unknown error -> rethrow.
            LOGGER.exception("[%s] Grow failed with unexpected error.", label)
            raise

    raise RuntimeError(
        f"Failed to grow tree after {max_tries} attempts. Last error: {last_err}"
    )


def tree_arrays(ft: FractalTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a grown FractalTree into numpy arrays."""
    nodes = np.asarray(ft.nodes_xyz, dtype=np.float64)
    edges = np.asarray(ft.connectivity, dtype=np.int64)
    pmj = np.asarray(ft.end_nodes, dtype=np.int64)
    return nodes, edges, pmj


def validate_outputs(
    nodes: np.ndarray, edges: np.ndarray, pmj: np.ndarray, label: str
) -> None:
    """Basic sanity checks to catch gross failures early."""
    require(
        nodes.ndim == 2 and nodes.shape[1] == 3,
        label,
        f"nodes shape invalid: {nodes.shape}",
    )
    require(
        edges.ndim == 2 and edges.shape[1] == 2,
        label,
        f"edges shape invalid: {edges.shape}",
    )
    require(nodes.shape[0] > 10, label, f"too few nodes: {nodes.shape[0]}")
    require(edges.shape[0] > 10, label, f"too few edges: {edges.shape[0]}")
    require(pmj.size > 0, label, "PMJs empty")

    require(np.isfinite(nodes).all(), label, "NaN/Inf in nodes")
    require(
        np.all(edges >= 0) and np.all(edges < nodes.shape[0]),
        label,
        "edge index out of range",
    )
    require(
        np.all(pmj >= 0) and np.all(pmj < nodes.shape[0]),
        label,
        "PMJ index out of range",
    )


def _pv_faces(connectivity: np.ndarray) -> np.ndarray:
    """Build PyVista face array from triangle connectivity."""
    conn = np.asarray(connectivity, dtype=np.int64)
    require(
        conn.ndim == 2 and conn.shape[1] == 3, "VIS", "tri connectivity must be (T, 3)"
    )
    faces = np.hstack([np.full((conn.shape[0], 1), 3, dtype=np.int64), conn])
    return faces.ravel()


def _pv_lines(edges: np.ndarray) -> np.ndarray:
    """Build PyVista line array from edge connectivity."""
    conn = np.asarray(edges, dtype=np.int64)
    require(
        conn.ndim == 2 and conn.shape[1] == 2, "VIS", "edge connectivity must be (E, 2)"
    )
    lines = np.hstack([np.full((conn.shape[0], 1), 2, dtype=np.int64), conn])
    return lines.ravel()


def visualize_results(
    *,
    lv_mesh: Mesh,
    rv_mesh: Mesh,
    lv_nodes: np.ndarray,
    lv_edges: np.ndarray,
    lv_pmj: np.ndarray,
    rv_nodes: np.ndarray,
    rv_edges: np.ndarray,
    rv_pmj: np.ndarray,
    show: bool,
    screenshot: Path | None,
    include_surface: bool,
) -> None:
    """Render LV/RV surfaces and Purkinje trees using PyVista."""
    if not show and screenshot is None:
        return

    try:
        import pyvista as pv
    except Exception as exc:
        LOGGER.warning("Visualization skipped: PyVista unavailable (%s).", exc)
        return

    off_screen = not show
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    if include_surface:
        lv_surface = pv.PolyData(lv_mesh.verts, _pv_faces(lv_mesh.connectivity))
        rv_surface = pv.PolyData(rv_mesh.verts, _pv_faces(rv_mesh.connectivity))
        plotter.add_mesh(lv_surface, color="tomato", opacity=0.15, name="LV surface")
        plotter.add_mesh(rv_surface, color="steelblue", opacity=0.15, name="RV surface")

    lv_tree = pv.PolyData(lv_nodes)
    lv_tree.lines = _pv_lines(lv_edges)
    rv_tree = pv.PolyData(rv_nodes)
    rv_tree.lines = _pv_lines(rv_edges)
    plotter.add_mesh(lv_tree, color="red", line_width=2, name="LV tree")
    plotter.add_mesh(rv_tree, color="blue", line_width=2, name="RV tree")

    if lv_pmj.size:
        lv_pmj_pts = pv.PolyData(lv_nodes[lv_pmj])
        plotter.add_mesh(
            lv_pmj_pts,
            color="gold",
            point_size=6,
            render_points_as_spheres=True,
            name="LV PMJ",
        )
    if rv_pmj.size:
        rv_pmj_pts = pv.PolyData(rv_nodes[rv_pmj])
        plotter.add_mesh(
            rv_pmj_pts,
            color="limegreen",
            point_size=6,
            render_points_as_spheres=True,
            name="RV PMJ",
        )

    legend_items = [
        ["LV tree", "red"],
        ["RV tree", "blue"],
        ["LV PMJ", "gold"],
        ["RV PMJ", "limegreen"],
    ]
    if include_surface:
        legend_items = [
            ["LV surface", "tomato"],
            ["RV surface", "steelblue"],
        ] + legend_items
    plotter.add_legend(legend_items)

    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            plotter.show(screenshot=str(screenshot))
            LOGGER.info("Visualization saved: %s", screenshot)
        except Exception as exc:
            LOGGER.warning("Visualization screenshot failed: %s", exc)
    elif show:
        plotter.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smoke test for purkinje-uv on LV/RV cut-surface OBJ meshes.",
    )
    ap.add_argument("--lv", type=str, required=True, help="Path to LV cut-surface .obj")
    ap.add_argument("--rv", type=str, required=True, help="Path to RV cut-surface .obj")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--lv-seeds", type=int, nargs=2, default=None, metavar=("INIT", "SECOND")
    )
    ap.add_argument(
        "--rv-seeds", type=int, nargs=2, default=None, metavar=("INIT", "SECOND")
    )
    ap.add_argument(
        "--max-tries",
        type=int,
        default=12,
        help="Max retries when growth exits the domain [default: 12]",
    )
    ap.add_argument(
        "--shrink",
        type=float,
        default=0.85,
        help="Shrink factor applied to lengths on retry [default: 0.85]",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("PURKINJE_UV_LOGLEVEL", "INFO"),
        help="Logging level (e.g., INFO, DEBUG) [default: PURKINJE_UV_LOGLEVEL or INFO]",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive PyVista visualization of the results.",
    )
    ap.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Write a PNG screenshot of the visualization to this path.",
    )
    ap.add_argument(
        "--no-surface",
        action="store_true",
        help="Hide LV/RV surface meshes in the visualization.",
    )
    args = ap.parse_args()

    configure_logging(args.log_level)
    LOGGER.info("Starting purkinje-uv smoke test")
    start_time = time.perf_counter()

    require(
        args.max_tries > 0,
        "CONFIG",
        f"max-tries must be > 0, got {args.max_tries}",
    )
    require(
        0.0 < args.shrink <= 1.0,
        "CONFIG",
        f"shrink must be in (0, 1], got {args.shrink}",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lv_path = Path(args.lv)
    rv_path = Path(args.rv)
    if not lv_path.is_file():
        raise FileNotFoundError(f"LV mesh not found: {lv_path}")
    if not rv_path.is_file():
        raise FileNotFoundError(f"RV mesh not found: {rv_path}")

    # Load meshes (verts only) via purkinje_uv.Mesh to match internal indexing.
    lv_mesh = Mesh(filename=str(lv_path))
    rv_mesh = Mesh(filename=str(rv_path))
    LOGGER.info(
        "[LV] verts=%d tris=%d",
        lv_mesh.verts.shape[0],
        lv_mesh.connectivity.shape[0],
    )
    LOGGER.info(
        "[RV] verts=%d tris=%d",
        rv_mesh.verts.shape[0],
        rv_mesh.connectivity.shape[0],
    )

    lv_diag = mesh_size_diameter(lv_mesh.verts)
    rv_diag = mesh_size_diameter(rv_mesh.verts)
    require(lv_diag > 0.0, "LV", "mesh diagonal is zero")
    require(rv_diag > 0.0, "RV", "mesh diagonal is zero")

    if args.lv_seeds is not None:
        lv_seeds = tuple(args.lv_seeds)
        lv_seed_source = "cli"
    else:
        lv_seeds = pca_seed_vertices(lv_mesh.verts)
        lv_seed_source = "pca"

    if args.rv_seeds is not None:
        rv_seeds = tuple(args.rv_seeds)
        rv_seed_source = "cli"
    else:
        rv_seeds = pca_seed_vertices(rv_mesh.verts)
        rv_seed_source = "pca"

    validate_seeds("LV", lv_seeds, lv_mesh.verts.shape[0])
    validate_seeds("RV", rv_seeds, rv_mesh.verts.shape[0])

    LOGGER.info(
        "[LV] diag=%.3f seeds=%s source=%s",
        lv_diag,
        lv_seeds,
        lv_seed_source,
    )
    LOGGER.info(
        "[RV] diag=%.3f seeds=%s source=%s",
        rv_diag,
        rv_seeds,
        rv_seed_source,
    )

    # Build params (start) + retry shrink if needed.
    p_lv0 = build_params(lv_path, lv_seeds, lv_diag)
    p_rv0 = build_params(rv_path, rv_seeds, rv_diag)

    LOGGER.info(
        "[LV] params: init_length=%.6g length=%.6g l_segment=%.6g N_it=%d",
        p_lv0.init_length,
        p_lv0.length,
        p_lv0.l_segment,
        p_lv0.N_it,
    )
    LOGGER.info(
        "[RV] params: init_length=%.6g length=%.6g l_segment=%.6g N_it=%d",
        p_rv0.init_length,
        p_rv0.length,
        p_rv0.l_segment,
        p_rv0.N_it,
    )

    p_lv, ft_lv = grow_with_retry(
        p_lv0,
        label="LV",
        max_tries=args.max_tries,
        shrink=args.shrink,
    )
    p_rv, ft_rv = grow_with_retry(
        p_rv0,
        label="RV",
        max_tries=args.max_tries,
        shrink=args.shrink,
    )

    LOGGER.info(
        "[LV] final params: init_length=%.6g length=%.6g l_segment=%.6g N_it=%d",
        p_lv.init_length,
        p_lv.length,
        p_lv.l_segment,
        p_lv.N_it,
    )
    LOGGER.info(
        "[RV] final params: init_length=%.6g length=%.6g l_segment=%.6g N_it=%d",
        p_rv.init_length,
        p_rv.length,
        p_rv.l_segment,
        p_rv.N_it,
    )

    # Convert grown trees to arrays for validation/output.
    lv_nodes, lv_edges, lv_pmj = tree_arrays(ft_lv)
    rv_nodes, rv_edges, rv_pmj = tree_arrays(ft_rv)

    validate_outputs(lv_nodes, lv_edges, lv_pmj, "LV")
    validate_outputs(rv_nodes, rv_edges, rv_pmj, "RV")

    LOGGER.info(
        "[LV] nodes=%d edges=%d pmj=%d",
        lv_nodes.shape[0],
        lv_edges.shape[0],
        lv_pmj.size,
    )
    LOGGER.info(
        "[RV] nodes=%d edges=%d pmj=%d",
        rv_nodes.shape[0],
        rv_edges.shape[0],
        rv_pmj.size,
    )

    # Save VTU.
    lv_tree = PurkinjeTree(lv_nodes, lv_edges, lv_pmj)
    rv_tree = PurkinjeTree(rv_nodes, rv_edges, rv_pmj)

    lv_vtu = out / "S62_LV_purkinje.vtu"
    rv_vtu = out / "S62_RV_purkinje.vtu"
    lv_tree.save(str(lv_vtu))
    rv_tree.save(str(rv_vtu))
    LOGGER.info("Saved VTU: %s", lv_vtu)
    LOGGER.info("Saved VTU: %s", rv_vtu)

    # Optional: PMJs only (best-effort).
    pmj_paths: list[Path] = []
    try:
        lv_pmj_path = out / "S62_LV_pmj.vtp"
        rv_pmj_path = out / "S62_RV_pmj.vtp"
        lv_tree.save_pmjs(str(lv_pmj_path))
        rv_tree.save_pmjs(str(rv_pmj_path))
        LOGGER.info("Saved PMJ VTP: %s", lv_pmj_path)
        LOGGER.info("Saved PMJ VTP: %s", rv_pmj_path)
        pmj_paths = [lv_pmj_path, rv_pmj_path]
    except Exception as e:
        LOGGER.warning("PMJ save skipped: %s", e)

    screenshot_path = Path(args.screenshot) if args.screenshot else None
    visualize_results(
        lv_mesh=lv_mesh,
        rv_mesh=rv_mesh,
        lv_nodes=lv_nodes,
        lv_edges=lv_edges,
        lv_pmj=lv_pmj,
        rv_nodes=rv_nodes,
        rv_edges=rv_edges,
        rv_pmj=rv_pmj,
        show=args.show,
        screenshot=screenshot_path,
        include_surface=not args.no_surface,
    )

    elapsed = time.perf_counter() - start_time
    LOGGER.info("Saved outputs to: %s", out.resolve())
    if pmj_paths:
        LOGGER.info("PMJ outputs: %s, %s", pmj_paths[0], pmj_paths[1])
    LOGGER.info("Run complete in %.2fs", elapsed)


if __name__ == "__main__":
    main()
