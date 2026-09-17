"""
05_deepforest_itd.py
================================================================================
AI-BASED TREE DETECTION (DeepForest) -- CROSS-VALIDATION AGAINST 03
================================================================================

WHY THIS SCRIPT EXISTS
-----------------------
`03_individual_tree_detection.py` finds trees from HEIGHT alone (a CHM +
local-maxima + watershed). This script finds trees a completely different
way: from a true-color IMAGE, using DeepForest (Weinstein et al. 2019), a
pretrained deep-learning object detector (RetinaNet) that recognizes tree
crowns the way a human aerial-photo interpreter would -- from color and
texture, not height.

Two independent detection methods that (mostly) agree on the same trees is
much stronger evidence than either method alone. Where they DISAGREE tells
you exactly which trees are worth double-checking by eye. That cross-check
is the actual point of this script -- it is a complement to `03`, not a
replacement for it (DeepForest itself has no concept of height at all; this
script borrows `03`'s CHM just to look up a height for each DeepForest box).

WHY THIS IS A SEPARATE SCRIPT FROM 03 (NOT BOLTED ON)
-------------------------------------------------------
1. DEPENDENCY WEIGHT: DeepForest requires PyTorch + torchvision, which is
   roughly 1-5 GB depending on CPU/GPU build -- utterly unlike every other
   dependency in this toolkit (scipy/numpy/rasterio/skimage/CSF are all
   lightweight by comparison). Forcing that install on everyone who just
   wants CHM-based detection would be a bad trade. Keeping it in its own
   script means you only pay that cost if you actually want to try this.
2. DIFFERENT WORKFLOW SHAPE: DeepForest takes an image path and returns 2D
   pixel boxes directly -- no CHM, no watershed, no CSF needed for
   detection itself. That's a fundamentally different pipeline shape than
   `03`, even though it ends up answering a similar question.
3. Matches the existing pattern in this toolkit: `04` reads `01`/`02`'s
   *outputs* rather than being folded into them. This script does the same
   with `03`.

HONEST CAVEATS -- READ BEFORE RUNNING
---------------------------------------
1. **Only works if your .las files have REAL RGB color data.** DeepForest
   needs a genuine image. If your TLS/UAV scans are height+intensity only
   (the common case for un-colorized TLS), there is no image to build and
   this script will skip that plot with a clear message -- not a crash,
   but also not a workaround. Confirmed by `has_real_rgb()` (from
   `tls_core.py`) per file, same check `03` already uses.
2. **Domain mismatch is a real, open question, not a hypothetical.**
   DeepForest's pretrained model was trained on NEON (National Ecological
   Observatory Network) sites -- US ecological research forests. MyDiv is
   young, densely-planted, mixed-species SAPLING plots in Germany. That's
   a meaningfully different forest type, age, and planting pattern than
   what the model learned from. It may perform fine, or it may perform
   poorly -- this script's whole purpose is to let you SEE which, on your
   actual data, rather than assume either way.
3. **Reconstructed imagery vs. a real photo.** The true-color raster this
   script builds (via `tls_core.render_photorealistic_topview()`, the same
   function `03` uses for `trees_bboxes2.png`) is assembled from point
   cloud color via nearest-neighbor gap-filling and smoothing -- it can
   look convincing to a person but may still differ from real camera
   texture in ways that matter to a model trained on real photos.
4. **This script does NOT prove which method is more "correct".** Without
   an independent field-measured stem map, agreement/disagreement between
   DeepForest and the CHM-watershed method tells you where the two methods
   diverge, not which one is right.

WHAT THIS SCRIPT ACTUALLY DOES, STEP BY STEP
-----------------------------------------------
For each plot number x sensor (same batch pattern as 02/03):
  1. Check the .las file has real RGB (skip with a clear message if not)
  2. Build ground classification + CHM (reused from tls_core, same
     pipeline as 03/get_chm(), needed to look up each tree's height later)
  3. Build a high-resolution true-color raster from the point cloud's own
     RGB (tls_core.render_photorealistic_topview(), same function 03 uses)
  4. Save that raster as a PNG (what DeepForest actually reads) and a
     GeoTIFF (for opening in GIS software)
  5. Load DeepForest's pretrained model (cached across plots in one run --
     the weights only need to load once) and run it on the PNG
  6. Convert DeepForest's pixel-coordinate boxes to real-world (meter)
     coordinates using the raster's known transform
  7. Look up each detected box's height from the CHM (DeepForest itself
     has no height information -- this step borrows it from `03`'s method)
  8. If `03`'s batch results exist for the same plot+sensor
     (`detected_trees.csv` in its batch output folder), match DeepForest's
     boxes against them by IoU (intersection-over-union) and categorize:
     matched by both methods / DeepForest-only / watershed-only
  9. Save a comparison image (both detection sets overlaid, different
     colors) and a per-plot CSV
After all plots: a summary CSV across every plot/sensor, and (if
`reportlab` is available) a short PDF report with the comparison table,
images, and a written discussion.

INSTALLING THE DEPENDENCY
---------------------------
    pip install deepforest
This pulls in PyTorch + torchvision automatically. Expect a large,
possibly slow download (multiple GB) and check you have several GB of
free disk space first. GPU is not required -- CPU inference works, just
slower (the images here are small, ~500x500px, so this should still be
fast per plot even on CPU).

Run:
    python 05_deepforest_itd.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rasterio.transform import from_origin

import tls_core as tc

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    # Same file-finding convention as 02/03, so all three scripts can point
    # at the same folder without any translation.
    "plot_numbers": [43, 58, 70],
    "tls_dir": ".",
    "uav_dir": ".",
    "tls_name_pattern": "Plot{n}_MyDiv.las",
    "uav_name_pattern": "MyDiv_{n}_L1.las",
    "epsg_code": 25833,

    "output_dir": "outputs/deepforest_itd",

    # --- ground classification / CHM (needed only to look up each
    # DeepForest box's height -- same parameters as 03's defaults) ---
    "csf_scene": "relief",
    "csf_cloth_resolution": 1.0,
    "csf_max_iterations": 600,
    "csf_class_threshold": 0.5,
    "csf_downsample_voxel_m": 0.02,   # see tls_core.voxel_downsample() -- avoids
                                       # an OOM crash on large real TLS scans
    "chm_res": 0.05,

    # --- true-color image built for DeepForest ---
    "topview2_res": 0.02,      # meters/pixel -- see tls_core docstring;
                                # smaller = sharper image, larger file
    "topview2_smooth_px": 1.5,  # softening blur, same idea as in 03

    # --- DeepForest itself ---
    "deepforest_confidence_threshold": 0.3,   # drop detections below this score
                                               # (DeepForest's own default is
                                               # already applied internally;
                                               # this is an ADDITIONAL filter
                                               # on top, 0-1, higher = stricter)

    # --- cross-validation against 03's results ---
    # Where to find 03's batch output (Plot{n}_{sensor}_detected_trees.csv
    # inside Plot{n}_{sensor}/ subfolders) -- set this to whatever
    # `batch_output_dir` you used when you ran 03. If not found, this
    # script still runs and just skips the comparison step for that plot.
    "watershed_batch_dir": "outputs/tree_detection_batch",
    "iou_match_threshold": 0.3,   # how much two boxes must overlap (0-1) to
                                   # count as "the same tree" between methods

    "generate_pdf_report": True,
    "report_filename": "DeepForest_Comparison_Report.pdf",
}


# ================================================================================
# DEPENDENCY CHECK
# ================================================================================

def check_deepforest_available():
    """
    Tries to import deepforest, with a clear, actionable message instead of
    a raw traceback if it's missing. Import is deferred to here (not done
    at the top of the file) so the rest of this script's logic can still
    be inspected/tested without the dependency installed.
    """
    try:
        from deepforest import main as deepforest_main
        return deepforest_main
    except ImportError as e:
        print("=" * 78)
        print("  DeepForest is not installed.")
        print("=" * 78)
        print("  This script needs the 'deepforest' package, which pulls in")
        print("  PyTorch + torchvision automatically -- expect a large,")
        print("  possibly slow download (multiple GB). Install with:")
        print()
        print("      pip install deepforest")
        print()
        print("  Then re-run this script. (Original error: "
              f"{type(e).__name__}: {e})")
        print("=" * 78)
        raise SystemExit(1)


_MODEL_CACHE = {}


def get_deepforest_model(cfg):
    """
    Loads DeepForest's pretrained model ONCE and reuses it across every
    plot/sensor in a batch run -- reloading the weights per file would be
    slow and pointless, they never change between plots.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]

    deepforest_main = check_deepforest_available()
    print("[DeepForest] Loading pretrained model (first run may also "
          "download weights, ~100-200 MB)...")
    model = deepforest_main.deepforest()
    # Different deepforest versions expose the pretrained-weights loader
    # under different method names -- try both for compatibility.
    try:
        model.use_release()
    except AttributeError:
        model.load_model("weecology/deepforest-tree")
    print("[DeepForest] Model loaded.")
    _MODEL_CACHE["model"] = model
    return model


# ================================================================================
# STEP 1-4: BUILD CHM + TRUE-COLOR RASTER FOR ONE FILE
# ================================================================================

def build_chm_and_truecolor(las_path, cfg, out_dir, file_tag):
    """
    Runs the ground-classification + CHM + true-color pipeline for one
    .las file. Returns None (with a printed reason) if the file has no
    real RGB data -- there's nothing for DeepForest to look at in that
    case. Otherwise returns a dict with everything downstream steps need.
    """
    print(f"  [1/4] Reading {las_path} ...")
    xyz, las = tc.read_las(las_path)
    print(f"        {len(xyz):,} points loaded")

    if not tc.has_real_rgb(las):
        print("  ! No real RGB color data in this file -- DeepForest needs an "
              "actual image to look at, so this plot/sensor is skipped. "
              "(This is expected for un-colorized TLS scans -- see the "
              "script header for why.)")
        return None
    print("        real RGB color data found -- proceeding")

    if cfg["epsg_code"]:
        tc.assign_crs(las, cfg["epsg_code"])

    print("  [2/4] Ground classification + CHM (for height lookup later)...")
    xyz_clean, keep_mask = tc.sor_filter(xyz, k=8, std_ratio=1.0)
    print(f"        SOR filter: kept {len(xyz_clean):,} / {len(xyz):,} points")

    rgb01_clean = tc.extract_rgb01(las)[keep_mask]

    if cfg.get("csf_downsample_voxel_m"):
        xyz_for_csf, _ = tc.voxel_downsample(xyz_clean, cfg["csf_downsample_voxel_m"])
        print(f"        downsampled {len(xyz_clean):,} -> {len(xyz_for_csf):,} "
              f"points for CSF ground classification")
    else:
        xyz_for_csf = xyz_clean

    ground_mask = tc.classify_ground_csf(
        xyz_for_csf, scene=cfg["csf_scene"],
        cloth_resolution=cfg["csf_cloth_resolution"],
        max_iterations=cfg["csf_max_iterations"],
        class_threshold=cfg["csf_class_threshold"],
    )
    ground_xyz = xyz_for_csf[ground_mask]
    print(f"        ground points: {ground_mask.sum():,} / {len(xyz_for_csf):,}")

    if len(ground_xyz) > 300_000:
        ground_xyz, _ = tc.voxel_downsample(ground_xyz, cfg["chm_res"])
        print(f"        thinned ground points to {len(ground_xyz):,} for faster TIN")

    _, _, _, ground_interp = tc.rasterize_terrain_tin(ground_xyz, res=cfg["chm_res"])
    xyz_norm = tc.normalize_height(xyz_clean, ground_interp)
    height_mask = xyz_norm[:, 2] >= 0
    xyz_norm = xyz_norm[height_mask]
    rgb01_norm = rgb01_clean[height_mask]
    print(f"        height-normalized: {len(xyz_norm):,} points (Z>=0)")

    chm, chm_transform, _ = tc.rasterize_canopy_p2r(xyz_norm, res=cfg["chm_res"], fill_gaps=True)
    print(f"        CHM built: shape={chm.shape}, mean height={np.nanmean(chm):.2f} m")

    print("  [3/4] Building true-color raster for DeepForest...")
    true_color_rgb, tc_transform, tc_shape = tc.render_photorealistic_topview(
        xyz_norm, rgb01_norm, chm_transform, chm.shape, cfg
    )
    print(f"        true-color raster: {tc_shape[1]}x{tc_shape[0]} px "
          f"at {cfg['topview2_res']} m/pixel")

    print("  [4/4] Saving true-color image + GeoTIFF...")
    png_path = os.path.join(out_dir, f"{file_tag}_truecolor.png")
    _save_rgb_png(true_color_rgb, png_path)
    geotiff_path = os.path.join(out_dir, f"{file_tag}_truecolor.tif")
    tc.save_true_color_geotiff(true_color_rgb, tc_transform, geotiff_path)
    print(f"        saved {png_path}")
    print(f"        saved {geotiff_path}")

    return {
        "chm": chm, "chm_transform": chm_transform,
        "true_color_rgb": true_color_rgb, "tc_transform": tc_transform, "tc_shape": tc_shape,
        "png_path": png_path,
    }


def _save_rgb_png(rgb01_image, path):
    """Saves a [0,1]-range HxWx3 array as a standard 8-bit PNG (what DeepForest reads)."""
    from PIL import Image as PILImage
    arr255 = (np.clip(rgb01_image, 0, 1) * 255).astype(np.uint8)
    PILImage.fromarray(arr255).save(path)


# ================================================================================
# STEP 5-6: RUN DEEPFOREST + CONVERT TO REAL-WORLD COORDINATES
# ================================================================================

def run_deepforest_on_image(png_path, cfg):
    """
    Runs DeepForest's pretrained model on one image, returns the raw
    prediction DataFrame (pixel coordinates: xmin, ymin, xmax, ymax,
    label, score), filtered by `deepforest_confidence_threshold`.

    VERSION ROBUSTNESS: DeepForest's predict_image() signature changed
    between versions -- versions before 2.0 accepted (and effectively
    required) a `return_plot` keyword; version 2.0+ removed it entirely
    (predict_image now always returns a dataframe; plotting was split
    into a separate visualize.plot_results() function). Calling with the
    wrong style raises "unexpected keyword argument 'return_plot'" on 2.0+
    or behaves unexpectedly on older versions if omitted. Rather than
    hardcode one style, this checks the installed version's actual
    function signature first and calls it the way THAT version expects.
    """
    model = get_deepforest_model(cfg)
    print(f"  [DeepForest] Running inference on {png_path} ...")

    import inspect
    sig = inspect.signature(model.predict_image)
    if "return_plot" in sig.parameters:
        boxes = model.predict_image(path=png_path, return_plot=False)
    else:
        boxes = model.predict_image(path=png_path)

    if boxes is None or len(boxes) == 0:
        print("  [DeepForest] No trees detected in this image.")
        return None

    # Some newer versions expose a shapely 'geometry' column instead of
    # (or alongside) plain xmin/ymin/xmax/ymax columns -- extract bounds
    # from it if the plain columns aren't there.
    if not all(c in boxes.columns for c in ("xmin", "ymin", "xmax", "ymax")):
        if "geometry" in boxes.columns:
            print("  [DeepForest] Converting 'geometry' column to xmin/ymin/xmax/ymax...")
            bounds = boxes["geometry"].apply(lambda g: g.bounds)
            boxes["xmin"] = bounds.apply(lambda b: b[0])
            boxes["ymin"] = bounds.apply(lambda b: b[1])
            boxes["xmax"] = bounds.apply(lambda b: b[2])
            boxes["ymax"] = bounds.apply(lambda b: b[3])
        else:
            raise ValueError(
                f"Unrecognized DeepForest output format -- columns found: "
                f"{list(boxes.columns)}. This likely means your installed "
                f"deepforest version's predict_image() output changed again -- "
                f"check `boxes.columns` yourself and adjust this function."
            )

    # DeepForest's score column has been named "score" in newer versions
    # and "scores" in some older ones -- handle both.
    score_col = "score" if "score" in boxes.columns else "scores"
    n_before = len(boxes)
    boxes = boxes[boxes[score_col] >= cfg["deepforest_confidence_threshold"]].reset_index(drop=True)
    print(f"  [DeepForest] {len(boxes)} / {n_before} detections kept "
          f"(confidence >= {cfg['deepforest_confidence_threshold']})")
    boxes = boxes.rename(columns={score_col: "score"})
    return boxes


def deepforest_boxes_to_world(boxes_px, transform):
    """
    Converts DeepForest's pixel-coordinate boxes (origin top-left, as in a
    normal image) into real-world meter coordinates, using the same
    transform math used throughout this toolkit.
    """
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f

    rows = []
    for _, r in boxes_px.iterrows():
        x_min = xmin_world + r["xmin"] * res
        x_max = xmin_world + r["xmax"] * res
        y_max = ymax_world - r["ymin"] * res
        y_min = ymax_world - r["ymax"] * res
        rows.append({
            "bbox_xmin": x_min, "bbox_xmax": x_max,
            "bbox_ymin": y_min, "bbox_ymax": y_max,
            "x_center": (x_min + x_max) / 2, "y_center": (y_min + y_max) / 2,
            "score": r["score"],
        })
    return rows


# ================================================================================
# STEP 7: LOOK UP HEIGHT FROM THE CHM FOR EACH DEEPFOREST BOX
# ================================================================================

def lookup_height_for_boxes(boxes_world, chm, chm_transform):
    """
    DeepForest itself has no height information -- it only sees a flat
    image. For each detected box, samples the CHM (built in step 2) at the
    box's center to get an estimated tree height, so DeepForest's
    detections end up with the same kind of height field as 03's.
    """
    res = chm_transform.a
    xmin_world, ymax_world = chm_transform.c, chm_transform.f
    nrows, ncols = chm.shape

    for b in boxes_world:
        col = int(np.clip((b["x_center"] - xmin_world) / res, 0, ncols - 1))
        row = int(np.clip((ymax_world - b["y_center"]) / res, 0, nrows - 1))
        h = chm[row, col]
        b["height_m"] = float(h) if not np.isnan(h) else None
    return boxes_world


# ================================================================================
# STEP 8: CROSS-VALIDATE AGAINST 03's WATERSHED DETECTIONS
# ================================================================================

def load_watershed_detections(plot_n, sensor, cfg):
    """
    Loads 03's Plot{n}_{sensor}_detected_trees.csv from its batch output,
    if it exists. Returns None (with a printed message) if not found --
    this script still runs and produces DeepForest-only results in that
    case, just skips the comparison step.
    """
    path = os.path.join(cfg["watershed_batch_dir"], f"Plot{plot_n}_{sensor}",
                         f"Plot{plot_n}_{sensor}_detected_trees.csv")
    if not os.path.exists(path):
        print(f"  ! No 03 results found at {path} -- run 03 in batch mode "
              f"first for a cross-validated comparison. Continuing with "
              f"DeepForest-only results for this plot.")
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "bbox_xmin": float(r["bbox_xmin"]), "bbox_xmax": float(r["bbox_xmax"]),
                "bbox_ymin": float(r["bbox_ymin"]), "bbox_ymax": float(r["bbox_ymax"]),
                "height_m": float(r["height_m"]),
            })
    print(f"  loaded {len(rows)} watershed (03) detections for comparison")
    return rows


def iou(box_a, box_b):
    """Standard intersection-over-union between two axis-aligned boxes."""
    ax0, ax1 = box_a["bbox_xmin"], box_a["bbox_xmax"]
    ay0, ay1 = box_a["bbox_ymin"], box_a["bbox_ymax"]
    bx0, bx1 = box_b["bbox_xmin"], box_b["bbox_xmax"]
    by0, by1 = box_b["bbox_ymin"], box_b["bbox_ymax"]

    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(deepforest_boxes, watershed_boxes, iou_threshold):
    """
    Greedy IoU matching between two sets of boxes: for every DeepForest
    box, find the watershed box with the highest IoU; if that IoU clears
    `iou_threshold`, count them as the same tree (matched). Anything left
    over on either side is method-specific ("only DeepForest found this"
    / "only watershed found this").
    Returns (n_matched, n_deepforest_only, n_watershed_only).
    """
    if watershed_boxes is None:
        return None, len(deepforest_boxes), None

    matched_watershed = set()
    n_matched = 0
    for db in deepforest_boxes:
        best_iou, best_j = 0.0, None
        for j, wb in enumerate(watershed_boxes):
            if j in matched_watershed:
                continue
            v = iou(db, wb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold:
            matched_watershed.add(best_j)
            n_matched += 1

    n_deepforest_only = len(deepforest_boxes) - n_matched
    n_watershed_only = len(watershed_boxes) - n_matched
    return n_matched, n_deepforest_only, n_watershed_only


# ================================================================================
# STEP 9: VISUALIZATION + PER-PLOT CSV
# ================================================================================

def plot_comparison(true_color_rgb, tc_transform, tc_shape, deepforest_boxes,
                     watershed_boxes, out_path, title):
    """Overlays both detection sets (different colors) on the true-color raster."""
    res = tc_transform.a
    xmin_world, ymax_world = tc_transform.c, tc_transform.f
    extent = (xmin_world, xmin_world + tc_shape[1] * res,
              ymax_world - tc_shape[0] * res, ymax_world)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(true_color_rgb, extent=extent, origin="upper")

    for b in deepforest_boxes:
        w = b["bbox_xmax"] - b["bbox_xmin"]
        h = b["bbox_ymax"] - b["bbox_ymin"]
        ax.add_patch(mpatches.Rectangle((b["bbox_xmin"], b["bbox_ymin"]), w, h,
                                         linewidth=1.8, edgecolor="cyan", facecolor="none"))
    if watershed_boxes:
        for b in watershed_boxes:
            w = b["bbox_xmax"] - b["bbox_xmin"]
            h = b["bbox_ymax"] - b["bbox_ymin"]
            ax.add_patch(mpatches.Rectangle((b["bbox_xmin"], b["bbox_ymin"]), w, h,
                                             linewidth=1.2, edgecolor="yellow",
                                             facecolor="none", linestyle="dashed"))

    handles = [mpatches.Patch(edgecolor="cyan", facecolor="none", label="DeepForest (AI)")]
    if watershed_boxes:
        handles.append(mpatches.Patch(edgecolor="yellow", facecolor="none",
                                       label="CHM-watershed (03)", linestyle="dashed"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  saved {out_path}")


def save_boxes_csv(boxes, path):
    if not boxes:
        return
    keys = list(boxes[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for b in boxes:
            f.write(",".join(str(b.get(k, "")) for k in keys) + "\n")
    print(f"  saved {path}")


# ================================================================================
# BATCH ORCHESTRATION
# ================================================================================

def process_one_plot_sensor(n, sensor, cfg):
    print(f"\n{'=' * 78}\nDeepForest ITD: Plot {n} - {sensor}\n{'=' * 78}")

    name_pattern = cfg["tls_name_pattern"] if sensor == "TLS" else cfg["uav_name_pattern"]
    dir_key = "tls_dir" if sensor == "TLS" else "uav_dir"
    las_path = os.path.join(cfg[dir_key], name_pattern.format(n=n))

    if not os.path.exists(las_path):
        print(f"  ! skipped -- file not found: {las_path}")
        return None

    file_tag = f"Plot{n}_{sensor}"
    out_dir = os.path.join(cfg["output_dir"], file_tag)
    os.makedirs(out_dir, exist_ok=True)

    built = build_chm_and_truecolor(las_path, cfg, out_dir, file_tag)
    if built is None:
        return {"plot_number": n, "sensor": sensor, "skipped_reason": "no_rgb",
                "n_deepforest": 0, "n_matched": None, "n_deepforest_only": None,
                "n_watershed_only": None}

    df_boxes_px = run_deepforest_on_image(built["png_path"], cfg)
    if df_boxes_px is None:
        deepforest_boxes = []
    else:
        deepforest_boxes = deepforest_boxes_to_world(df_boxes_px, built["tc_transform"])
        deepforest_boxes = lookup_height_for_boxes(deepforest_boxes, built["chm"], built["chm_transform"])

    save_boxes_csv(deepforest_boxes, os.path.join(out_dir, f"{file_tag}_deepforest_trees.csv"))

    watershed_boxes = load_watershed_detections(n, sensor, cfg)
    n_matched, n_deepforest_only, n_watershed_only = match_detections(
        deepforest_boxes, watershed_boxes, cfg["iou_match_threshold"]
    )

    comparison_png = os.path.join(out_dir, f"{file_tag}_comparison.png")
    plot_comparison(
        built["true_color_rgb"], built["tc_transform"], built["tc_shape"],
        deepforest_boxes, watershed_boxes, comparison_png,
        f"Plot {n} {sensor}: DeepForest vs. CHM-watershed"
    )

    print(f"\n  Plot {n} {sensor} summary: DeepForest={len(deepforest_boxes)} trees, "
          f"watershed={'n/a' if watershed_boxes is None else len(watershed_boxes)} trees, "
          f"matched={n_matched if n_matched is not None else 'n/a'}")

    return {
        "plot_number": n, "sensor": sensor, "skipped_reason": None,
        "n_deepforest": len(deepforest_boxes),
        "n_watershed": None if watershed_boxes is None else len(watershed_boxes),
        "n_matched": n_matched, "n_deepforest_only": n_deepforest_only,
        "n_watershed_only": n_watershed_only,
        "comparison_png": comparison_png,
    }


def save_summary_csv(rows, path):
    if not rows:
        print("  ! no results to summarize")
        return
    keys = ["plot_number", "sensor", "skipped_reason", "n_deepforest", "n_watershed",
            "n_matched", "n_deepforest_only", "n_watershed_only"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"saved {path}")


def build_pdf_report(rows, cfg, path):
    """A short PDF: summary table, comparison images, and a written discussion."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         PageBreak, Image as RLImage, Table, TableStyle)
        from reportlab.lib import colors
    except ImportError:
        print("  ! reportlab not installed -- skipping PDF report "
              "(pip install reportlab to enable it)")
        return

    def img_flowable(p, width=15 * cm, max_h=20 * cm):
        if not p or not os.path.exists(p):
            return None
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            w, h = im.size
        aspect = w / h
        height = width / aspect
        if height > max_h:
            height, width = max_h, max_h * aspect
        flow = RLImage(p, width=width, height=height)
        flow.hAlign = "CENTER"
        return flow

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="T", fontSize=18, alignment=TA_CENTER,
                               spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="S", fontSize=14, spaceBefore=14, spaceAfter=8,
                               fontName="Helvetica-Bold", textColor=colors.HexColor("#1a4d2e")))
    styles.add(ParagraphStyle(name="B", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="C", fontSize=8.5, alignment=TA_CENTER,
                               textColor=colors.grey, spaceAfter=12))

    story = [Spacer(1, 1 * cm),
             Paragraph("DeepForest (AI) vs. CHM-Watershed Tree Detection", styles["T"]),
             Spacer(1, 0.5 * cm),
             Paragraph(
                 "DeepForest is a deep-learning tree detector pretrained on NEON (US) forest "
                 "imagery -- a different forest type/age/country than these plots. This report "
                 "is an honest empirical check of how well it transfers to this data, not an "
                 "assumption that it will. Read the discussion at the end before drawing "
                 "conclusions from the numbers alone.", styles["B"]),
             PageBreak()]

    header = ["Plot", "Sensor", "Status", "DeepForest", "Watershed (03)", "Matched",
              "DF only", "WS only"]
    table_rows = []
    for r in rows:
        status = "no RGB" if r.get("skipped_reason") == "no_rgb" else "OK"
        table_rows.append([
            f"Plot{r['plot_number']}", r["sensor"], status,
            str(r.get("n_deepforest", "")), str(r.get("n_watershed", "n/a")),
            str(r.get("n_matched", "n/a")), str(r.get("n_deepforest_only", "n/a")),
            str(r.get("n_watershed_only", "n/a")),
        ])
    t = Table([header] + table_rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f3")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    for r in rows:
        img = img_flowable(r.get("comparison_png"))
        if img:
            story.append(img)
            story.append(Paragraph(f"Plot {r['plot_number']} {r['sensor']}: cyan = DeepForest, "
                                    f"dashed yellow = CHM-watershed (03)", styles["C"]))
    story.append(PageBreak())

    story.append(Paragraph("Discussion", styles["S"]))
    story.append(Paragraph(
        "Where the two methods agree (\"Matched\"), that's a tree detected by two "
        "independent signals -- height shape AND visual appearance -- meaningfully "
        "stronger evidence than either alone. \"DF only\" trees were seen by "
        "DeepForest's visual model but not by the CHM-watershed method -- worth checking "
        "whether these are real trees the height-based method under-segmented/merged, or "
        "false positives from the AI model. \"WS only\" trees were found by height/shape "
        "but not recognized visually by DeepForest -- could be small/young trees not well "
        "represented in DeepForest's NEON training data, or genuine height-based "
        "detections with no clear visual signature in the reconstructed point-cloud image.",
        styles["B"]))
    story.append(Paragraph(
        "If DeepForest's counts are dramatically lower or higher than the watershed "
        "method's across most plots, that's a signal of the domain-mismatch concern "
        "discussed in this script's header -- the pretrained model may simply not "
        "transfer well to young MyDiv sapling plots. If the two methods track each other "
        "reasonably closely, that's encouraging evidence both are capturing something real.",
        styles["B"]))

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    doc.build(story)
    print(f"saved {path}")


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("=" * 78)
    print("  DEEPFOREST AI TREE DETECTION -- CROSS-VALIDATION AGAINST 03")
    print("=" * 78)
    print(f"  Plots: {cfg['plot_numbers']}   Output: {cfg['output_dir']}")
    print("  See this script's header comment for the honest caveats before")
    print("  trusting these results (domain mismatch, RGB requirement, etc.)")

    rows = []
    for n in cfg["plot_numbers"]:
        for sensor in ["TLS", "UAV"]:
            try:
                result = process_one_plot_sensor(n, sensor, cfg)
                if result:
                    rows.append(result)
            except Exception as e:
                print(f"  ! FAILED Plot{n} {sensor}: {e}")

    save_summary_csv(rows, os.path.join(cfg["output_dir"], "deepforest_comparison_summary.csv"))

    if cfg.get("generate_pdf_report", True):
        build_pdf_report(rows, cfg, os.path.join(cfg["output_dir"], cfg["report_filename"]))

    print(f"\nDone. {len(rows)} plot/sensor combinations processed. See: {cfg['output_dir']}")
    return rows


if __name__ == "__main__":
    main(CONFIG)