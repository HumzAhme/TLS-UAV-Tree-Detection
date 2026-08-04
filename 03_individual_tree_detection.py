"""
03_individual_tree_detection.py
================================================================================
INDIVIDUAL TREE DETECTION (ITD) / INDIVIDUAL TREE CROWN (ITC) DELINEATION
================================================================================

GOAL
----
Take a canopy height model (CHM) derived from your .las point cloud and
automatically detect each individual tree, drawing a bounding box (like the
cyan/orange squares in your reference image) and a crown outline around it
— i.e. the same task shown on the right side of your screenshot, but run
directly on your own TLS-derived data instead of an aerial RGB orthomosaic.

================================================================================
LITERATURE REVIEW: HOW THIS IS DONE (methods, papers, and software products)
================================================================================

Individual tree detection has ~25 years of remote-sensing literature behind
it. Every method boils down to two steps: (1) find a "seed" point for every
tree (usually its treetop), then (2) grow/delineate the area around each
seed that belongs to that tree (the crown). Below is what's actually used
in practice, roughly in order of how often you'll encounter it:

--------------------------------------------------------------------------
(A) CHM-BASED LOCAL-MAXIMA + WATERSHED  <-- WHAT THIS SCRIPT IMPLEMENTS
--------------------------------------------------------------------------
The dominant, most widely validated family of methods. Rasterize the point
cloud into a CHM (you already did this in `tls_pipeline.py`), then:

  Step 1 - Seed detection ("treetop" finding):
    - Popescu & Wynne (2004), "Seeing the trees in the forest": introduced
      the Variable Window Filter (VWF) -- the search window used to decide
      whether a pixel is a local maximum scales with that pixel's height,
      because taller trees have wider crowns than shorter trees. This is
      why the code below supports a variable window, not just a fixed one.
    - A simpler, still very common alternative is a FIXED window (same
      search radius everywhere) -- faster, and fine if your trees are all
      roughly the same size (e.g. a young, even-aged plantation plot like
      MyDiv saplings).

  Step 2 - Crown delineation (growing the region around each seed):
    - Chen, Baldocchi, Gong & Kelly (2006): first applied classic image
      "marker-controlled watershed" segmentation to a CHM, using the
      treetops as markers. Conceptually: turn the CHM upside down (so
      peaks become basins) and "flood" outward from every treetop
      simultaneously; each pixel gets assigned to whichever treetop's
      flood reaches it first. Watershed lines (where two floods meet)
      become the crown boundaries.
    - Dalponte & Coomes (2016): refined watershed with two extra rules to
      stop crowns from over-growing into their neighbors or into the
      ground/understory: (i) a pixel can only belong to a tree if its
      height is at least `th_seed` x the seed's own height (a relative
      height floor), and (ii) a maximum crown radius cap. This is the
      "dalponte2016()" algorithm in the R `lidR` package, and the one this
      script's constraints are modeled on.
    - Silva et al. (2016): a related watershed variant ("silva2016()" in
      lidR) with a similar exclusion-distance idea but simpler tuning.
    - Li et al. (2012): a fully point-based method (no CHM/raster needed at
      all) that spatially sorts points top-down and assigns each point to
      an existing tree or starts a new one based on 3D distance thresholds.
      Good for cases where rasterizing loses too much detail, but has more
      parameters to tune and is noticeably slower on dense TLS data.

  Software that implements this exact family of methods:
    - R package `lidR`               -> locate_trees() + segment_trees()
                                         with lmf(), li2012(), silva2016(),
                                         dalponte2016(), watershed()
    - FUSION/LDV (US Forest Service)  -> CanopyMaxima, TreeSeg
    - ESRI ArcGIS Pro                 -> "Segment Individual Trees" tool
    - Green Valley LiDAR360           -> "Individual Tree Segmentation"
    - eCognition (Trimble)            -> object-based image analysis (OBIA)
                                         watershed/multiresolution segmentation

--------------------------------------------------------------------------
(B) 3D POINT-BASED CLUSTERING (no CHM/rasterization at all)
--------------------------------------------------------------------------
    - Ferraz et al. (2016): adaptive mean-shift 3D clustering directly on
      the point cloud, letting cluster "bandwidth" shrink with height --
      handles overlapping/multi-layered crowns better than 2D CHM methods,
      at the cost of being much slower and having more parameters.
    - Region growing seeded by detected treetops in full 3D, using a
      cone/inverted-cone shaped search volume that widens as you move down
      from the treetop (approximating a natural crown shape) -- this is
      the *simplified* idea offered as `assign_points_3d_conical()` below,
      as a lightweight approximation of Ferraz's approach, more robust than
      a strict 2D CHM-column lookup when crowns interlock or TLS occlusion
      creates CHM gaps.

--------------------------------------------------------------------------
(C) DEEP LEARNING ON RGB IMAGERY  <-- what your reference screenshot is
--------------------------------------------------------------------------
    - Weinstein et al. (2019), "DeepForest": a RetinaNet object-detection
      model pre-trained on the NEON forest RGB benchmark, fine-tunable on
      your own site. Output = bounding boxes with confidence scores, drawn
      directly on an RGB orthomosaic -- almost certainly what generated
      your reference image (that cyan/orange bounding-box style is
      DeepForest's signature look).
      pip install deepforest  ->  from deepforest import main; m = main.deepforest();
      m.use_release(); boxes = m.predict_image(path="orthomosaic.tif")
    - IMPORTANT CAVEAT FOR YOUR DATA: DeepForest needs an RGB (or
      near-RGB) raster image, not a bare point cloud. It only applies here
      if you *also* have a drone/orthophoto of the same plot (or if your
      LAS points are colorized with RGB from photogrammetry, in which case
      you could rasterize a top-down RGB image from the points and feed
      that in). A bare-intensity TLS point cloud has no color channels for
      this method to use. This script does NOT implement DeepForest for
      that reason, but the snippet above is everything you need if you
      later get RGB orthomosaic imagery of the same plots.

--------------------------------------------------------------------------
(D) TLS-SPECIFIC STEM-BASED METHODS (most relevant to *your* sensor)
--------------------------------------------------------------------------
Everything above (A-C) was originally designed for AIRBORNE data (drone/
plane), which looks straight down onto an unbroken canopy surface. Your
data is TERRESTRIAL (ground-based) -- the scanner looks up and outward
THROUGH the canopy, so:
  - Individual stems/trunks are actually visible near the ground (airborne
    sensors essentially never see trunks) -- this opens up an entirely
    different, often more reliable strategy: detect trunks directly
    (e.g. by slicing the point cloud into a thin horizontal band around
    "breast height", ~1.3 m above the ground, then clustering the resulting
    trunk cross-sections and fitting circles to each cluster to get exact
    trunk positions and diameters), then assign the rest of each tree's
    crown points to whichever detected trunk is closest.
  - Dedicated software for exactly this: the R package `TreeLS` (stem
    mapping, `stemPoints()`+`stemSegmentation()`+circle fitting), and
    QSM (Quantitative Structure Model) tools like `TreeQSM`/`AdQSM`, which
    reconstruct full 3D tree skeletons from TLS data.
  - Trade-off: stem detection is more accurate for mature, well-separated
    trunks with a clear view at breast height, but fails on very young
    saplings (no clear trunk yet) or extremely occluded/dense understories
    -- for a young, dense MyDiv sapling plot, the CHM-based approach (A)
    implemented below is usually the more practical starting point; stem
    detection is a natural "phase 2" once you have canopy-based tree
    locations to validate/refine against trunk positions.

--------------------------------------------------------------------------
WHY THIS SCRIPT USES METHOD (A)
--------------------------------------------------------------------------
It directly reuses the CHM you already built in the earlier scripts, has
by far the best-documented literature/parameter guidance, is what nearly
all of the commercial/open-source tools above are actually built on, and
does not require RGB imagery you may not have. Method (D) is flagged above
as the natural next step specifically because your data is TLS.

================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours
from skimage.morphology import h_maxima
import rasterio
from rasterio.transform import from_origin

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, Table, TableStyle
)
from reportlab.lib import colors

import tls_core as tc

# ================================================================================
# CONFIGURATION -- every parameter that can change the classification result,
# with its meaning, valid/typical range, and what happens if you turn it up
# or down. Start with the defaults, run once, look at the overlay PNG, then
# adjust ONE parameter at a time and re-run -- that's the fastest way to
# learn how each one behaves on your specific plot.
# ================================================================================

CONFIG = {
    # ---------------------------------------------------------------
    # INPUT: give EITHER a raw .las file (the script builds the CHM for
    # you using tls_core, same as tls_pipeline.py) OR a path to an
    # already-computed CHM GeoTIFF (e.g. the CHM.tif from tls_pipeline.py
    # or from 01_tls_foliage_density_pipeline.py) to skip recomputing it.
    # ---------------------------------------------------------------
    "input_las": "MyDiv_43_L1.las",
    "input_chm_tif": None,     # e.g. "/mnt/user-data/outputs/CHM.tif" -- if set, input_las is ignored
    "output_dir": "outputs/tree_detection",

    # ---------------------------------------------------------------
    # If building the CHM from a raw LAS file, these mirror tls_pipeline.py.
    # ---------------------------------------------------------------
    "csf_scene": "relief",             # "steep_slope" | "relief" | "flat"
    "csf_cloth_resolution": 1.0,
    "csf_max_iterations": 600,
    "csf_class_threshold": 0.5,
    "chm_res": 0.05,                   # meters/pixel. Finer = more detail but noisier
                                        # local maxima; coarser = smoother but can merge
                                        # trees that are close together. Typical: 0.05-0.25 m
                                        # for TLS, 0.15-1.0 m for UAV/ALS.

    # ---------------------------------------------------------------
    # CHM SMOOTHING (applied before treetop search)
    # ---------------------------------------------------------------
    # A Gaussian blur applied to the CHM before searching for treetops.
    # WHY: raw point-to-raster CHMs are noisy -- individual leaves/branches
    # create lots of tiny spurious local bumps that would otherwise be
    # (wrongly) detected as separate trees.
    #   smoothing_sigma_m = 0        -> no smoothing (fastest, noisiest,
    #                                    tends to OVER-segment: too many
    #                                    trees detected from noise bumps)
    #   smoothing_sigma_m = 0.05-0.15 -> light smoothing, RECOMMENDED for
    #                                    small/young trees planted close
    #                                    together (e.g. MyDiv saplings) --
    #                                    heavier smoothing can flatten away
    #                                    the shallow dip between two
    #                                    neighboring crowns and merge them
    #                                    into one detection (see
    #                                    "peak_detection_method" below if
    #                                    you're seeing this problem)
    #   smoothing_sigma_m = 0.3-1.0  -> heavy smoothing, only appropriate
    #                                    for mature, widely-spaced trees;
    #                                    HIGH RISK of UNDER-segmentation
    #                                    (merging several real neighboring
    #                                    trees into one blob) on densely
    #                                    packed canopies
    "smoothing_sigma_m": 0.15,

    # ---------------------------------------------------------------
    # MINIMUM TREE HEIGHT (m above ground)
    # ---------------------------------------------------------------
    # Pixels below this height are never considered a treetop and are
    # excluded from the watershed entirely (they'll get tree_id = 0,
    # "ground/understory/unclassified"). Set this to roughly the height
    # of your shortest real tree, to exclude grass/shrubs/noise.
    #   Young MyDiv saplings: try 0.5-1.5 m
    #   Mature forest plots:  try 2-5 m
    "min_height_m": 0.5,

    # ---------------------------------------------------------------
    # TREETOP SEARCH / PEAK DETECTION METHOD
    # ---------------------------------------------------------------
    # This is THE parameter to change if you're seeing several real,
    # adjacent trees being merged into one detection (under-segmentation)
    # -- e.g. a dense, same-species, similarly-tall group of saplings where
    # only a shallow dip separates one crown from the next.
    #
    #   "fixed"    -> constant search radius everywhere (see
    #                 fixed_window_radius_m below). Simple, but a radius
    #                 large enough to avoid noise will also merge any two
    #                 real trees closer together than that radius.
    #   "variable" -> Popescu & Wynne (2004) height-scaled radius (see vwf_*
    #                 below). Same fundamental limitation as "fixed": it
    #                 can only ever separate two peaks if they are farther
    #                 apart than the (height-dependent) radius, REGARDLESS
    #                 of how deep/shallow the dip between them actually is.
    #   "h_maxima" -> RECOMMENDED FOR YOUR CASE. Instead of a fixed spacing
    #                 rule, this asks a much more direct question at every
    #                 candidate peak: "does this bump stick up by at least
    #                 `h_maxima_prominence_m` meters above the lowest
    #                 saddle/dip connecting it to any taller neighboring
    #                 peak?" (the "extended maxima transform" /
    #                 morphological H-maxima method). Two trees standing
    #                 right next to each other, even just centimeters
    #                 apart, will correctly come out as TWO peaks as long
    #                 as there is a real dip of at least that many meters
    #                 between their tops -- exactly the "recognize the
    #                 shadow/slight boundary between trees" behavior you
    #                 asked for, because it directly measures the boundary
    #                 depth instead of guessing a fixed distance.
    "peak_detection_method": "h_maxima",

    # --- if peak_detection_method == "h_maxima" ---
    # Minimum prominence (m) a bump must have above its connecting saddle
    # to count as its own tree.
    #   Lower (e.g. 0.05-0.15) -> MUCH more sensitive to subtle boundaries;
    #                              will split tightly-packed, similarly-tall
    #                              trees correctly, but if set too low will
    #                              also start picking up noise bumps/branch
    #                              clusters as fake extra "trees"
    #                              (over-segmentation). START HERE for
    #                              dense young sapling plots.
    #   Higher (e.g. 0.5-1.5)  -> only very obviously separate, tall-vs-
    #                              short peaks get split; subtle boundaries
    #                              between same-height neighboring crowns
    #                              will be merged again (this is what was
    #                              happening implicitly with the old
    #                              "fixed"/"variable" methods on your data)
    # Good practice: try 0.1, 0.2, 0.3, 0.5 in turn and look at how the
    # tree count / bounding boxes change -- there is usually a "sweet
    # spot" where real trees are separated but noise is not.
    "h_maxima_prominence_m": 0.1,

    # ---------------------------------------------------------------
    # TREETOP SEARCH WINDOW (only used if peak_detection_method is
    # "fixed" or "variable" -- ignored for "h_maxima")
    # ---------------------------------------------------------------
    "window_mode": "variable",   # "fixed" or "variable"

    # --- if window_mode == "fixed" ---
    # A single, constant search radius (m) used everywhere on the plot.
    # Smaller radius -> MORE trees detected (risk of one real tree split
    # into multiple detections). Larger radius -> FEWER trees detected
    # (risk of merging two real neighboring trees into one detection).
    #   Typical for young/small trees:  0.5 - 1.5 m
    #   Typical for mature forest:      2.0 - 4.0 m
    "fixed_window_radius_m": 1.0,

    # --- if window_mode == "variable" (Popescu & Wynne 2004 style) ---
    # radius(height) = clip(a + b * height, min_radius_m, max_radius_m)
    # i.e. taller pixels get a wider "no other treetop nearby" search
    # radius, because tall trees really do tend to have wider crowns.
    # `a` and `b` should be calibrated to YOUR species/site if possible
    # (e.g. by measuring a few known crown radii vs. heights in the field
    # or in CloudCompare, then fitting a line) -- the defaults below are a
    # generic, conservative starting point, not a universal truth.
    #   Increasing `b` -> tall trees get proportionally much wider search
    #                     radii than short ones (more height-dependent).
    #   Increasing `a` -> shifts ALL radii up/down by a constant, regardless
    #                      of height.
    "vwf_a": 0.5,           # intercept (m)
    "vwf_b": 0.15,          # slope (m of radius per m of height)
    "vwf_min_radius_m": 0.3,
    "vwf_max_radius_m": 3.0,

    # ---------------------------------------------------------------
    # CROWN DELINEATION (WATERSHED) CONSTRAINTS
    # Modeled on Dalponte & Coomes (2016). These stop the "flood" from a
    # treetop spilling too far into a neighboring tree or into the ground.
    # ---------------------------------------------------------------
    # A pixel can only be assigned to a tree if its height is at least
    # this FRACTION of that tree's own seed (treetop) height.
    #   Lower (e.g. 0.2)  -> crowns grow bigger/lower down the tree (risk:
    #                         swallowing neighboring low vegetation)
    #   Higher (e.g. 0.7) -> crowns stay small, close to the very top only
    #                         (risk: missing the true, wider crown extent)
    #   Typical starting point: 0.35 - 0.5
    "th_seed": 0.40,

    # Hard cap (m) on how far any single tree's crown can extend outward
    # from its own treetop, regardless of what the watershed flood would
    # otherwise do. Prevents one dominant tall tree from "eating" all its
    # smaller neighbors.
    #   Typical: roughly 1.5x your expected mature crown radius
    "max_crown_radius_m": 0.5,

    # ---------------------------------------------------------------
    # 3D POINT-TO-TREE ASSIGNMENT (after 2D crown segmentation)
    # ---------------------------------------------------------------
    # "columnar"  -> every point is assigned to whichever tree occupies
    #                the CHM pixel directly above/below it (fast, simple,
    #                standard approach, works well for airborne data with
    #                mostly non-overlapping crowns seen from above)
    # "conical"   -> approximates Ferraz et al. (2016): the search radius
    #                around each treetop SHRINKS as you go down in height,
    #                like an upside-down cone -- more forgiving of TLS
    #                occlusion gaps and slightly overlapping crowns, at
    #                higher compute cost. Recommended if "columnar" leaves
    #                too many points unassigned (tree_id = 0) below the
    #                CHM's clean top surface.
    "point_assignment_mode": "columnar",
    "conical_top_radius_factor": 1.0,   # multiplier on crown radius at treetop height
    "conical_bottom_radius_factor": 0.3,  # multiplier on crown radius near the ground

    # ---------------------------------------------------------------
    # OUTPUT / VISUALIZATION
    # ---------------------------------------------------------------
    # The background image behind the boxes is rendered from the CHM
    # (Canopy Height Model) -- a grid of HEIGHT NUMBERS (meters above
    # ground), not a photo. It has no color channels at all (your TLS
    # .las almost certainly has no RGB field either -- TLS records
    # intensity, not color, unless it was separately colorized from
    # photos). So a literal "photo-like" render isn't possible from height
    # data alone -- BUT a HILLSHADE (simulated sun lighting from the
    # height gradient, the same trick GIS software uses to make bare
    # elevation data look like a textured, almost photographic relief map)
    # gets much closer to that look than a flat grayscale/blurry image,
    # entirely from the height values you already have. This is what
    # `display_style: "hillshade"` below does.
    "display_style": "hillshade",   # "hillshade" (textured, photo-like) or "flat" (plain colormap)
    "display_use_smoothed": False,  # False = use the RAW (unsmoothed) CHM for
                                     # the background image so it looks sharp/
                                     # crisp instead of blurry. Detection still
                                     # always uses the smoothed CHM regardless
                                     # of this setting -- this only changes
                                     # what you SEE, not what gets detected.
    "hillshade_azimuth_deg": 315,    # simulated sun direction (315 = NW, GIS default)
    "hillshade_altitude_deg": 45,    # simulated sun height above horizon (deg)

    "chm_background_cmap": "Greens_r",  # tint color for the shaded height layer;
                                          # try "Greys_r" for classic grayscale or
                                          # "viridis" for a scientific look

    # How to color each tree's box, since a single flat color makes
    # overlapping boxes hard to tell apart:
    #   "height"     -> RECOMMENDED. Continuous colormap by treetop
    #                    height (with a colorbar) -- neighboring trees of
    #                    different heights become visually distinct even
    #                    when their boxes overlap.
    #   "tree_id"    -> Each tree cycles through a 20-color categorical
    #                    palette by ID -- maximizes contrast between ANY
    #                    two adjacent boxes (doesn't encode any real
    #                    quantity, just "these are different trees").
    #   "confidence" -> the original behavior: cyan = normal, orange =
    #                    crown area below low_confidence_crown_area_m2
    #                    (only 2 colors -- can look messy with many
    #                    overlapping boxes of the same color).
    "bbox_color_by": "height",

    # Any detected tree with a crown area smaller than this (m^2) is
    # flagged as a small/low-confidence detection (used by "confidence"
    # mode above, and always noted with a dashed instead of solid box
    # outline regardless of color mode, so you can still spot likely
    # false positives even when coloring by height/tree_id).
    "low_confidence_crown_area_m2": 0.3,

    "save_per_tree_las": False,   # True = also save one .las file per detected
                                  # tree (can be slow/disk-heavy for many trees)

    # ---------------------------------------------------------------
    # trees_bboxes2.png / trees_crown_outlines2.png -- dedicated
    # high-quality true-color top view
    # ---------------------------------------------------------------
    # Only produced if your .las has REAL RGB color data (see
    # has_real_rgb()) -- if it doesn't, there is no real color to render
    # and these files are skipped with an explanation printed to the
    # console rather than faking a color that was never actually measured.
    "topview2_res": 0.02,   # meters/pixel for the true-color render --
                             # DELIBERATELY INDEPENDENT of chm_res. This is
                             # the main lever for a sharper/clearer image:
                             # a smaller number = finer native pixel grid
                             # = more real point-color detail actually
                             # makes it into the picture, regardless of
                             # dpi below. Going finer than your TLS scan's
                             # actual point spacing adds no more real
                             # detail (just more interpolated/filled
                             # pixels) and makes the file bigger/slower.
                             # Typical range: 0.005 (very fine, dense
                             # close-range TLS) to 0.05 (matches default
                             # chm_res, coarser/faster). Set to None to
                             # just reuse chm_res (old behavior).
    "topview2_smooth_px": 1.5,   # gaussian blur (pixels) for a softer,
                                  # less pixelated look closer to how
                                  # CloudCompare blends overlapping points
                                  # on screen. 0 = no smoothing (crisper
                                  # but more speckled/harsh). Note this is
                                  # in PIXELS of the topview2_res grid, so
                                  # a finer topview2_res already means each
                                  # pixel covers less real-world distance.
    "topview2_dpi": 300,          # output PNG resolution (higher = larger,
                                  # sharper file; 300 is print-quality)

    # ---------------------------------------------------------------
    # BATCH MODE -- process multiple plots' TLS *and* UAV data in one run,
    # plus a built-in PDF report comparing tree detection results across
    # both sensors. When "batch_mode" is True, input_las/input_chm_tif
    # above are IGNORED -- files are instead found from plot_numbers +
    # the two name patterns below (same convention as
    # 02_uav_tls_chm_comparison.py, so both scripts can point at the same
    # folder of files).
    # ---------------------------------------------------------------
    "batch_mode": True,
    "plot_numbers": [43, 58, 70],
    "tls_dir": ".",
    "uav_dir": ".",
    "tls_name_pattern": "Plot{n}_MyDiv.las",
    "uav_name_pattern": "MyDiv_{n}_L1.las",
    "batch_output_dir": "outputs/tree_detection_batch",
    "generate_pdf_report": True,
    "report_filename": "Tree_Detection_Report.pdf",
    # Shown as a disclosure banner at the top of the PDF report -- set to
    # False once you've confirmed you're running on your own real .las
    # files (matches the same pattern used in 04_generate_report.py).
    "report_data_is_placeholder": True,
}


# ================================================================================
# STEP 1: GET THE CHM (either load an existing one, or build it from raw LAS)
# ================================================================================

def has_real_rgb(las):
    """
    Checks whether a LAS file has MEANINGFUL color data, not just the
    red/green/blue FIELDS existing. Several LAS point formats always
    include red/green/blue dimensions in their file structure even when
    no one ever populated them -- e.g. an uncolorized TLS scan saved in
    point format 3 will still technically "have" red/green/blue fields,
    just all set to 0. Checking only `dimension_names` would wrongly
    report "yes, this has color" for such a file. This function also
    checks that the values aren't degenerately all-zero before trusting
    them.
    """
    dims = las.point_format.dimension_names
    if not all(d in dims for d in ("red", "green", "blue")):
        return False
    r, g, b = np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)
    return not (r.max() == 0 and g.max() == 0 and b.max() == 0)


def extract_rgb01(las):
    """Returns Nx3 RGB in [0, 1], auto-detecting 8-bit vs. 16-bit LAS color storage."""
    r = np.asarray(las.red, dtype=float)
    g = np.asarray(las.green, dtype=float)
    b = np.asarray(las.blue, dtype=float)
    scale = 65535.0 if max(r.max(), g.max(), b.max()) > 255 else 255.0
    return np.stack([r, g, b], axis=1) / scale


def rasterize_true_color(xyz_norm, rgb01, transform, shape):
    """
    Builds a top-down TRUE-COLOR image from the point cloud's own RGB
    values, on the exact same grid as the CHM (so boxes drawn later line
    up correctly). For each output cell, uses the color of whichever
    point is HIGHEST in that cell -- i.e. the color of the visible
    "top of canopy" surface, matching what the CHM itself measures. This
    is the real, original point-cloud color (e.g. actual green foliage),
    not a synthetic height-based tint.
    Returns (rgb_image [rows, cols, 3] in [0,1], has_data_mask [rows,cols]
    -- cells with no point at all are `False` in the mask and left black
    in the image; the caller fills those in).
    """
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f
    nrows, ncols = shape

    x, y, z = xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2]
    col = np.clip(((x - xmin_world) / res).astype(int), 0, ncols - 1)
    row = np.clip(((ymax_world - y) / res).astype(int), 0, nrows - 1)
    flat_idx = row * ncols + col

    order = np.argsort(z)  # ascending -> assigning in this order means the
    flat_idx_sorted = flat_idx[order]  # LAST (highest-z) point written per
    rgb_sorted = rgb01[order]          # cell is the one that "wins"

    flat_rgb = np.zeros((nrows * ncols, 3))
    has_data = np.zeros(nrows * ncols, dtype=bool)
    flat_rgb[flat_idx_sorted] = rgb_sorted
    has_data[flat_idx_sorted] = True

    return flat_rgb.reshape(nrows, ncols, 3), has_data.reshape(nrows, ncols)


def build_finer_transform(transform, shape, target_res):
    """
    Given an existing transform+shape describing a real-world extent (the
    CHM grid), builds a NEW, finer transform+shape covering the exact
    SAME extent at `target_res` meters/pixel. Used to render
    trees_bboxes2.png / trees_crown_outlines2.png at much higher native
    pixel density than the CHM itself uses -- boxes and crown outlines are
    stored in real-world (meter) coordinates, so they still line up
    perfectly on this finer image; only the background image gets sharper.
    """
    res = transform.a
    xmin, ymax = transform.c, transform.f
    height_m = shape[0] * res
    width_m = shape[1] * res
    new_shape = (max(1, int(np.ceil(height_m / target_res))),
                 max(1, int(np.ceil(width_m / target_res))))
    new_transform = from_origin(xmin, ymax, target_res, target_res)
    return new_transform, new_shape


def render_photorealistic_topview(xyz_norm, rgb01, transform, shape, cfg):
    """
    Builds a HIGH-QUALITY top-down true-color render from the point
    cloud's own RGB values -- meant to look close to how CloudCompare
    renders the point cloud from directly above (like your reference
    screenshot), not the coarser/darker hillshade-blended background used
    in trees_bboxes.png.

    IMPORTANT: this renders at `topview2_res` (CONFIG), which is
    INDEPENDENT of `chm_res` -- the CHM only needs to be coarse enough for
    stable tree detection, but this image can (and by default does) use a
    much finer native pixel grid, since TLS point spacing is usually much
    finer than a typical CHM resolution. This is the main lever for
    "higher resolution, clearer" output -- increasing plot dpi alone can't
    add detail that was never in the underlying pixel grid to begin with.

    Three things make this look noticeably better than the raw
    `rasterize_true_color()` output:
      1. A FINER native grid (`topview2_res`, meters/pixel) than the CHM
         uses -- more real point-color detail actually makes it into the
         image instead of being averaged away into large coarse cells.
      2. Gap filling by NEAREST NEIGHBOR (via a distance transform)
         instead of a flat average color -- cells with no point directly
         above them (common between individual leaf clusters, especially
         at fine resolution) get the color of the closest point that DOES
         have data, so the canopy reads as one continuous textured surface
         instead of having flat gray/average speckles in it.
      3. A small Gaussian blur (`topview2_smooth_px`) across the color
         image, approximating how CloudCompare visually blends
         neighboring points together when rendering (real points aren't
         infinitely small dots on screen -- they overlap/blend) -- this
         is what gives the reference image its soft, "fluffy foliage"
         look rather than a harsh, pixelated one.
    All three are purely cosmetic and have no effect on tree detection.
    """
    target_res = cfg.get("topview2_res") or transform.a
    if target_res < transform.a:
        fine_transform, fine_shape = build_finer_transform(transform, shape, target_res)
    else:
        fine_transform, fine_shape = transform, shape

    rgb_image, has_data = rasterize_true_color(xyz_norm, rgb01, fine_transform, fine_shape)

    if (~has_data).any() and has_data.any():
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(~has_data, return_distances=True, return_indices=True)
        rgb_image = rgb_image[indices[0], indices[1]]

    sigma_px = cfg.get("topview2_smooth_px", 1.5)
    if sigma_px > 0:
        rgb_image = ndi.gaussian_filter(rgb_image, sigma=(sigma_px, sigma_px, 0))

    print(f"      true-color top view rendered at {target_res} m/pixel "
          f"({fine_shape[1]}x{fine_shape[0]} px)")
    return np.clip(rgb_image, 0, 1)


def plot_bboxes2(background_rgb, rows, transform, shape, cfg, path):
    """
    Same bounding-box overlay as plot_bboxes(), but drawn on top of the
    high-quality photorealistic true-color render (see
    render_photorealistic_topview()) instead of the CHM/hillshade
    background -- i.e. the "make it look like my reference photo, then
    draw the boxes on it" output, saved separately as trees_bboxes2.png so
    the original CHM-based trees_bboxes.png is completely unaffected.
    """
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f
    extent = (xmin_world, xmin_world + shape[1] * res,
              ymax_world - shape[0] * res, ymax_world)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background_rgb, extent=extent, origin="upper")

    low_conf_thresh = cfg["low_confidence_crown_area_m2"]
    color_by = cfg["bbox_color_by"]

    colorbar_mappable = None
    if color_by == "height" and rows:
        heights = np.array([r["height_m"] for r in rows])
        norm = plt.Normalize(vmin=heights.min(), vmax=heights.max())
        cmap = plt.cm.plasma
        colorbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    elif color_by == "tree_id":
        cmap = plt.cm.tab20

    for r in rows:
        is_low_conf = r["crown_area_m2"] < low_conf_thresh
        if color_by == "height":
            color = cmap(norm(r["height_m"]))
        elif color_by == "tree_id":
            color = cmap(r["tree_id"] % 20)
        else:
            color = "orange" if is_low_conf else "cyan"

        w = r["bbox_xmax"] - r["bbox_xmin"]
        h = r["bbox_ymax"] - r["bbox_ymin"]
        rect = mpatches.Rectangle(
            (r["bbox_xmin"], r["bbox_ymin"]), w, h,
            linewidth=1.5, edgecolor=color, facecolor="none",
            linestyle="dashed" if is_low_conf else "solid",
        )
        ax.add_patch(rect)

    if colorbar_mappable is not None:
        plt.colorbar(colorbar_mappable, ax=ax, fraction=0.046, label="Treetop height (m)")

    ax.set_title(f"Detected trees: {len(rows)}  (true-color top view)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.get("topview2_dpi", 200))
    plt.close()
    print(f"      saved {path}")


def plot_crown_outlines2(background_rgb, labels, rows, transform, shape, cfg, path):
    """
    True-color counterpart of plot_crown_outlines() -- identical red crown
    contours and white treetop markers, drawn on the photorealistic
    true-color top view instead of the CHM/hillshade background. Saved
    separately as trees_crown_outlines2.png; trees_crown_outlines.png is
    completely unaffected.
    """
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f
    extent = (xmin_world, xmin_world + shape[1] * res,
              ymax_world - shape[0] * res, ymax_world)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background_rgb, extent=extent, origin="upper")

    for r in rows:
        tree_id = r["tree_id"]
        mask = (labels == tree_id).astype(float)
        contours = find_contours(mask, level=0.5)
        for contour in contours:
            ys = ymax_world - contour[:, 0] * res
            xs = xmin_world + contour[:, 1] * res
            ax.plot(xs, ys, color="red", linewidth=1.2)
        ax.plot(r["x_top"], r["y_top"], "w+", markersize=8)

    ax.set_title(f"Crown outlines (true-color top view), n={len(rows)}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.get("topview2_dpi", 200))
    plt.close()
    print(f"      saved {path}")


def save_true_color_geotiff(background_rgb, transform, path):
    """
    Saves the photorealistic true-color top view as a 3-band GeoTIFF
    (uint8, 0-255 per channel), on the exact same grid as tree_labels.tif,
    as tree_labels2.tif -- so you have a GIS-loadable true-color image to
    go with the GIS-loadable tree-ID label raster. NOTE: unlike
    tree_labels.tif (which holds the actual tree_id classification data
    used for analysis), tree_labels2.tif holds COLOR, not a classification
    -- tree detection itself is always done from height/CHM data only and
    does not change based on color, so there's no alternate "RGB-based"
    tree_id classification to provide here; this file exists purely so
    every visual/GeoTIFF product has both an original and a true-color
    counterpart, per your naming convention.
    """
    rgb255 = (np.clip(background_rgb, 0, 1) * 255).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=rgb255.shape[0], width=rgb255.shape[1],
        count=3, dtype=np.uint8, crs=None, transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(rgb255[:, :, i], i + 1)
    print(f"      saved {path}")


def get_chm(cfg):
    if cfg["input_chm_tif"]:
        print(f"[1/6] Loading existing CHM: {cfg['input_chm_tif']}")
        with rasterio.open(cfg["input_chm_tif"]) as src:
            chm = src.read(1).astype(float)
            chm[chm == src.nodata] = np.nan
            transform = src.transform
        return chm, transform, None  # no point cloud available -> no true color possible

    print(f"[1/6] Building CHM from raw LAS: {cfg['input_las']}")
    xyz, las = tc.read_las(cfg["input_las"])
    xyz_clean, keep_mask = tc.sor_filter(xyz, k=8, std_ratio=1.0)

    rgb01_clean = None
    if has_real_rgb(las):
        rgb01_clean = extract_rgb01(las)[keep_mask]
        print("      real RGB color data found in this .las -- will additionally "
              "render trees_bboxes2.png using the actual point colors "
              "(trees_bboxes.png / trees_crown_outlines.png are unaffected -- "
              "they always use the CHM/hillshade rendering)")
    else:
        print("      no usable RGB color data in this .las (fields absent or all-zero, "
              "e.g. a raw un-colorized TLS scan) -- trees_bboxes2.png will be skipped; "
              "trees_bboxes.png / trees_crown_outlines.png still use the CHM/hillshade rendering")

    ground_mask = tc.classify_ground_csf(
        xyz_clean, scene=cfg["csf_scene"],
        cloth_resolution=cfg["csf_cloth_resolution"],
        max_iterations=cfg["csf_max_iterations"],
        class_threshold=cfg["csf_class_threshold"],
    )
    ground_xyz = xyz_clean[ground_mask]
    _, _, _, ground_interp = tc.rasterize_terrain_tin(ground_xyz, res=cfg["chm_res"])
    xyz_norm = tc.normalize_height(xyz_clean, ground_interp)
    height_mask = xyz_norm[:, 2] >= 0
    xyz_norm = xyz_norm[height_mask]
    if rgb01_clean is not None:
        rgb01_clean = rgb01_clean[height_mask]

    chm, transform, extent = tc.rasterize_canopy_p2r(xyz_norm, res=cfg["chm_res"], fill_gaps=True)
    print(f"      CHM built: shape={chm.shape}, resolution={cfg['chm_res']} m")

    rgb_info = (xyz_norm, rgb01_clean) if rgb01_clean is not None else None
    return chm, transform, rgb_info


# ================================================================================
# STEP 2: SMOOTH THE CHM
# ================================================================================

def smooth_chm(chm, transform, sigma_m):
    res = transform.a  # pixel size in meters (assumes square pixels)
    sigma_px = sigma_m / res
    chm_filled = np.where(np.isnan(chm), 0, chm)
    if sigma_px > 0:
        chm_smooth = ndi.gaussian_filter(chm_filled, sigma=sigma_px)
    else:
        chm_smooth = chm_filled.copy()
    print(f"[2/6] Smoothed CHM (sigma={sigma_m} m = {sigma_px:.2f} px)")
    return chm_smooth


# ================================================================================
# STEP 3: DETECT TREETOPS (fixed or variable window local maxima)
# ================================================================================

def detect_treetops_hmaxima(chm_smooth, transform, cfg):
    """
    Prominence-based treetop detection (extended maxima / H-maxima
    transform). Directly answers "is this bump at least `prominence_m`
    meters above the saddle connecting it to any taller neighbor?" for
    every candidate peak -- this is what correctly separates tightly
    packed, similarly-tall trees that a fixed/variable search RADIUS
    would otherwise merge into one detection, because it responds to the
    actual depth of the dip between two crowns rather than the distance
    between them.
    """
    prominence_m = cfg["h_maxima_prominence_m"]
    mask = h_maxima(chm_smooth, prominence_m)
    labeled, n = ndi.label(mask)

    coords = []
    for i in range(1, n + 1):
        rr, cc = np.where(labeled == i)
        heights = chm_smooth[rr, cc]
        j = np.argmax(heights)
        r, c = rr[j], cc[j]
        if chm_smooth[r, c] >= cfg["min_height_m"]:
            coords.append((r, c))
    return np.array(coords, dtype=int) if coords else np.empty((0, 2), dtype=int)


def detect_treetops(chm_smooth, transform, cfg):
    res = transform.a
    min_h = cfg["min_height_m"]

    if cfg["peak_detection_method"] == "h_maxima":
        coords = detect_treetops_hmaxima(chm_smooth, transform, cfg)
        print(f"[3/6] Detected {len(coords)} candidate treetops "
              f"(mode=h_maxima, prominence={cfg['h_maxima_prominence_m']} m, "
              f"min_height={min_h} m)")
        return coords

    if cfg["window_mode"] == "fixed":
        min_distance_px = max(1, int(round(cfg["fixed_window_radius_m"] / res)))
        coords = peak_local_max(
            chm_smooth, min_distance=min_distance_px,
            threshold_abs=min_h, exclude_border=False,
        )
    else:
        # Variable Window Filter (Popescu & Wynne 2004 style):
        # 1. Find ALL candidate local maxima with a small min_distance
        #    (just to avoid literally-adjacent duplicate pixels).
        # 2. Sort candidates tallest-first.
        # 3. Greedily keep a candidate only if no already-kept, taller
        #    candidate lies within ITS OWN height-dependent radius.
        candidates = peak_local_max(
            chm_smooth, min_distance=1, threshold_abs=min_h, exclude_border=False,
        )
        if len(candidates) == 0:
            coords = candidates
        else:
            heights = chm_smooth[candidates[:, 0], candidates[:, 1]]
            order = np.argsort(-heights)  # tallest first
            candidates = candidates[order]
            heights = heights[order]

            radii_m = np.clip(
                cfg["vwf_a"] + cfg["vwf_b"] * heights,
                cfg["vwf_min_radius_m"], cfg["vwf_max_radius_m"]
            )
            radii_px = radii_m / res

            kept_rc = []
            kept_radius_px = []
            for i in range(len(candidates)):
                r, c = candidates[i]
                too_close = False
                for (kr, kc), krad in zip(kept_rc, kept_radius_px):
                    dist = np.hypot(r - kr, c - kc)
                    if dist < max(radii_px[i], krad):
                        too_close = True
                        break
                if not too_close:
                    kept_rc.append((r, c))
                    kept_radius_px.append(radii_px[i])
            coords = np.array(kept_rc) if kept_rc else np.empty((0, 2), dtype=int)

    print(f"[3/6] Detected {len(coords)} candidate treetops "
          f"(mode={cfg['window_mode']}, min_height={min_h} m)")
    return coords


# ================================================================================
# STEP 4: WATERSHED CROWN DELINEATION (Dalponte & Coomes 2016 style constraints)
# ================================================================================

def segment_crowns(chm_smooth, treetops_rc, transform, cfg):
    res = transform.a
    mask = chm_smooth >= cfg["min_height_m"]

    markers = np.zeros(chm_smooth.shape, dtype=int)
    for i, (r, c) in enumerate(treetops_rc):
        markers[r, c] = i + 1  # tree IDs start at 1; 0 = background

    # Flood "downhill" from every treetop simultaneously. Using -chm makes
    # each treetop a basin so the flood expands outward/downward from it.
    labels = watershed(-chm_smooth, markers=markers, mask=mask)

    # Apply Dalponte & Coomes (2016)-style constraints on top of the raw
    # watershed result: relative height floor + absolute crown radius cap.
    seed_heights = {i + 1: chm_smooth[r, c] for i, (r, c) in enumerate(treetops_rc)}
    max_radius_px = cfg["max_crown_radius_m"] / res

    final_labels = labels.copy()
    for tree_id, seed_h in seed_heights.items():
        tree_mask = labels == tree_id
        if not tree_mask.any():
            continue
        rr, cc = np.where(tree_mask)
        seed_r, seed_c = treetops_rc[tree_id - 1]

        # constraint 1: relative height floor
        too_low = chm_smooth[rr, cc] < cfg["th_seed"] * seed_h
        # constraint 2: absolute radius cap
        dist_px = np.hypot(rr - seed_r, cc - seed_c)
        too_far = dist_px > max_radius_px

        drop = too_low | too_far
        final_labels[rr[drop], cc[drop]] = 0  # revoke assignment -> background

    n_trees = len(seed_heights)
    print(f"[4/6] Watershed segmentation complete: {n_trees} crowns delineated "
          f"(th_seed={cfg['th_seed']}, max_crown_radius={cfg['max_crown_radius_m']} m)")
    return final_labels


# ================================================================================
# STEP 5: PER-TREE METRICS (bounding box, crown area/diameter, treetop height)
# ================================================================================

def tree_metrics_table(labels, chm_smooth, treetops_rc, transform, res):
    props = regionprops(labels, intensity_image=chm_smooth)
    xmin_world, ymax_world = transform.c, transform.f

    rows = []
    for p in props:
        tree_id = p.label
        seed_r, seed_c = treetops_rc[tree_id - 1]
        top_h = chm_smooth[seed_r, seed_c]

        r0, c0, r1, c1 = p.bbox  # (min_row, min_col, max_row, max_col), exclusive
        x_min = xmin_world + c0 * res
        x_max = xmin_world + c1 * res
        y_max = ymax_world - r0 * res
        y_min = ymax_world - r1 * res

        x_top = xmin_world + seed_c * res
        y_top = ymax_world - seed_r * res

        crown_area_m2 = p.area * res * res
        crown_diameter_m = 2 * np.sqrt(crown_area_m2 / np.pi)  # equivalent circular diameter

        rows.append({
            "tree_id": tree_id,
            "x_top": x_top, "y_top": y_top, "height_m": top_h,
            "bbox_xmin": x_min, "bbox_xmax": x_max,
            "bbox_ymin": y_min, "bbox_ymax": y_max,
            "crown_area_m2": crown_area_m2,
            "crown_diameter_m": crown_diameter_m,
            "n_pixels": p.area,
        })
    print(f"[5/6] Computed metrics for {len(rows)} trees")
    return rows


def save_trees_csv(rows, path):
    if not rows:
        print("      ! no trees to save")
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"      saved {path}")


# ================================================================================
# STEP 6: VISUALIZATION (bounding boxes like your reference image + crown outlines)
# ================================================================================

def compute_hillshade(chm, res, azimuth_deg=315, altitude_deg=45):
    """
    Classic GIS "hillshade": simulates sunlight hitting the height surface
    from a given compass direction (azimuth) and elevation angle
    (altitude), producing a shaded-relief image where slopes facing the
    "sun" are bright and slopes facing away are dark. This is exactly the
    technique used to turn plain bare-earth elevation data into the
    textured, almost-photographic relief maps you see in GIS software and
    topographic maps -- applied here to canopy height instead of terrain,
    it makes individual crown bumps visually "pop" the way they do in a
    real photo, without needing any actual color/RGB data.
    Returns an array of shading values in [0, 1] (0 = fully shaded/dark,
    1 = fully lit), same shape as `chm`.
    """
    chm_filled = np.where(np.isnan(chm), 0, chm)
    dz_dy, dz_dx = np.gradient(chm_filled, res)

    slope = np.pi / 2 - np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)

    azimuth_rad = np.radians(360.0 - azimuth_deg + 90)
    altitude_rad = np.radians(altitude_deg)

    shaded = (np.sin(altitude_rad) * np.sin(slope) +
              np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect))
    return np.clip(shaded, 0, 1)


def render_chm_background(chm_raw, chm_smooth, transform, cfg, true_color=None):
    """
    Builds the RGB background image used by plot_bboxes() and
    plot_crown_outlines().

    If `true_color` is provided (the point cloud's own RGB, only possible
    when built from a raw .las that actually has real color data --
    see `has_real_rgb()`), that REAL color is used as the base image,
    with hillshade multiplied on top purely to add a bit of depth/texture.
    Otherwise (no color data available, the common case for un-colorized
    TLS scans), falls back to a synthetic height-based color ramp +
    hillshade -- see the "display_style"/"chm_background_cmap" CONFIG
    comments.
    """
    res = transform.a

    if true_color is not None:
        rgb_image, has_data = true_color
        base_rgb = rgb_image.copy()
        if (~has_data).any():
            # fill any gaps (cells with no point at all) with the average
            # color of the real data, so gaps don't show up as pure black
            fill_color = rgb_image[has_data].mean(axis=0) if has_data.any() else np.array([0.1, 0.3, 0.1])
            base_rgb[~has_data] = fill_color
    else:
        base = chm_smooth if cfg["display_use_smoothed"] else chm_raw
        base = np.where(np.isnan(base), 0, base)
        cmap = plt.get_cmap(cfg["chm_background_cmap"])
        norm = plt.Normalize(vmin=np.nanpercentile(base, 2), vmax=np.nanpercentile(base, 98))
        base_rgb = cmap(norm(base))[:, :, :3]

    if cfg["display_style"] == "hillshade":
        shade_source = chm_smooth if cfg["display_use_smoothed"] else chm_raw
        shade_source = np.where(np.isnan(shade_source), 0, shade_source)
        shade = compute_hillshade(shade_source, res, cfg["hillshade_azimuth_deg"], cfg["hillshade_altitude_deg"])
        shade = 0.5 + 0.5 * shade  # gentler blend so real colors aren't washed out
        rgb = base_rgb * shade[:, :, None]
    else:
        rgb = base_rgb

    return np.clip(rgb, 0, 1)


def plot_bboxes(chm_raw, chm_smooth, rows, transform, cfg, path, true_color=None):
    """
    Draws the bounding-box overlay image (the squares-on-a-canopy-render
    picture you've been looking at).

    WHAT'S IN THE BACKGROUND:
    Built from the CHM (Canopy Height Model) -- a grid of HEIGHT NUMBERS
    (meters above ground), not a photo. It is NOT the point cloud and has
    no "original green/RGB color" to preserve -- a CHM cell only ever
    holds one number, never a color (your TLS .las almost certainly has
    no RGB field either). `display_style: "hillshade"` (the default)
    simulates sunlight on that height surface to produce a textured,
    shaded-relief look much closer to a "real photo" appearance than a
    flat colormap, entirely from the height values alone -- see
    `compute_hillshade()`. `display_use_smoothed: False` (the default)
    renders from the RAW, unsmoothed CHM so the background looks sharp
    rather than blurry -- detection itself always still runs on the
    smoothed CHM regardless of this display setting, so changing it never
    affects which trees get detected, only what the picture looks like.

    WHAT THE BOXES MEAN:
    Every row in `rows` (one per detected tree) becomes one rectangle,
    positioned at that tree's `bbox_xmin/xmax/ymin/ymax` (the tight
    axis-aligned box around its segmented crown pixels -- see
    `tree_metrics_table()`). `bbox_color_by` in CONFIG controls how each
    box is colored (see the CONFIG comments above this function for the
    3 modes: "height", "tree_id", "confidence") -- this is also purely
    cosmetic, it recolors the same boxes, it doesn't change them.
    A dashed line style (instead of solid) always marks a
    low-confidence/likely-false-positive detection (crown area below
    `low_confidence_crown_area_m2`), regardless of color mode.

    IF YOUR BOXES OVERLAP A LOT:
    That's not a plotting problem -- it means the detected crowns
    genuinely overlap that much given your current `max_crown_radius_m`
    /`th_seed` settings. On a densely-planted plot (e.g. young saplings a
    meter or so apart), try lowering `max_crown_radius_m` (e.g. to
    0.5-1.0 m) so each box only extends to a realistic crown size --
    this usually cleans up the picture far more than any color change.
    Check the console line printed by segment_crowns() ("Watershed
    segmentation complete... max_crown_radius=X m") to confirm the value
    you set is actually the one that was used in that run.
    """
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f
    extent = (xmin_world, xmin_world + chm_smooth.shape[1] * res,
              ymax_world - chm_smooth.shape[0] * res, ymax_world)

    background_rgb = render_chm_background(chm_raw, chm_smooth, transform, cfg, true_color=true_color)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(background_rgb, extent=extent, origin="upper")

    low_conf_thresh = cfg["low_confidence_crown_area_m2"]
    color_by = cfg["bbox_color_by"]

    colorbar_mappable = None
    if color_by == "height" and rows:
        heights = np.array([r["height_m"] for r in rows])
        norm = plt.Normalize(vmin=heights.min(), vmax=heights.max())
        cmap = plt.cm.plasma
        colorbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    elif color_by == "tree_id":
        cmap = plt.cm.tab20

    for r in rows:
        is_low_conf = r["crown_area_m2"] < low_conf_thresh

        if color_by == "height":
            color = cmap(norm(r["height_m"]))
        elif color_by == "tree_id":
            color = cmap(r["tree_id"] % 20)
        else:  # "confidence"
            color = "orange" if is_low_conf else "cyan"

        w = r["bbox_xmax"] - r["bbox_xmin"]
        h = r["bbox_ymax"] - r["bbox_ymin"]
        rect = mpatches.Rectangle(
            (r["bbox_xmin"], r["bbox_ymin"]), w, h,
            linewidth=1.5, edgecolor=color, facecolor="none",
            linestyle="dashed" if is_low_conf else "solid",
        )
        ax.add_patch(rect)

    if colorbar_mappable is not None:
        plt.colorbar(colorbar_mappable, ax=ax, fraction=0.046, label="Treetop height (m)")

    subtitle = {
        "height": "box color = treetop height",
        "tree_id": "box color = tree ID (arbitrary, for contrast only)",
        "confidence": f"orange = crown area < {low_conf_thresh} m2",
    }[color_by]
    ax.set_title(f"Detected trees: {len(rows)}  ({subtitle})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[6/6] saved {path}")


def plot_crown_outlines(chm_raw, chm_smooth, labels, rows, transform, cfg, path, true_color=None):
    res = transform.a
    xmin_world, ymax_world = transform.c, transform.f
    extent = (xmin_world, xmin_world + chm_smooth.shape[1] * res,
              ymax_world - chm_smooth.shape[0] * res, ymax_world)

    background_rgb = render_chm_background(chm_raw, chm_smooth, transform, cfg, true_color=true_color)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(background_rgb, extent=extent, origin="upper")

    for r in rows:
        tree_id = r["tree_id"]
        mask = (labels == tree_id).astype(float)
        contours = find_contours(mask, level=0.5)
        for contour in contours:
            ys = ymax_world - contour[:, 0] * res
            xs = xmin_world + contour[:, 1] * res
            ax.plot(xs, ys, color="red", linewidth=1)
        ax.plot(r["x_top"], r["y_top"], "w+", markersize=8)

    ax.set_title(f"Crown outlines (actual segmented shape), n={len(rows)}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"      saved {path}")


def save_label_raster(labels, transform, path):
    tc.save_geotiff(labels.astype(float), transform, path, nodata=-1)
    print(f"      saved {path}")


# ================================================================================
# 3D POINT -> TREE ID ASSIGNMENT
# ================================================================================

def assign_points_columnar(xyz_norm, labels, transform, res):
    """Assign each 3D point to the tree occupying its CHM column (fast, standard)."""
    xmin_world, ymax_world = transform.c, transform.f
    col = np.clip(((xyz_norm[:, 0] - xmin_world) / res).astype(int), 0, labels.shape[1] - 1)
    row = np.clip(((ymax_world - xyz_norm[:, 1]) / res).astype(int), 0, labels.shape[0] - 1)
    tree_ids = labels[row, col]
    return tree_ids


def assign_points_conical(xyz_norm, rows, cfg):
    """
    Approximate 3D conical-crown assignment (see Ferraz et al. 2016 discussion
    in the literature review above): search radius around each treetop
    SHRINKS as height decreases, mimicking a natural crown taper. Slower
    (O(n_points x n_trees)) but more forgiving of TLS occlusion gaps.
    """
    tree_ids = np.zeros(len(xyz_norm), dtype=int)
    best_dist = np.full(len(xyz_norm), np.inf)

    for r in rows:
        tid = r["tree_id"]
        top_h = r["height_m"]
        crown_r = r["crown_diameter_m"] / 2

        dz = np.clip(xyz_norm[:, 2] / top_h, 0, 1) if top_h > 0 else np.zeros(len(xyz_norm))
        # radius allowed at this point's height: interpolate between
        # bottom-factor (near ground) and top-factor (at treetop height)
        allowed_r = crown_r * (
            cfg["conical_bottom_radius_factor"]
            + dz * (cfg["conical_top_radius_factor"] - cfg["conical_bottom_radius_factor"])
        )
        horiz_dist = np.hypot(xyz_norm[:, 0] - r["x_top"], xyz_norm[:, 1] - r["y_top"])
        within = horiz_dist <= allowed_r
        better = within & (horiz_dist < best_dist)
        tree_ids[better] = tid
        best_dist[better] = horiz_dist[better]

    return tree_ids


def save_classified_las(xyz, tree_ids, path, template_las=None):
    import laspy
    if template_las is not None:
        header = laspy.LasHeader(point_format=template_las.header.point_format,
                                  version=template_las.header.version)
    else:
        header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las_out.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32))
    las_out.tree_id = tree_ids.astype(np.int32)
    las_out.write(path)
    print(f"      saved {path} (open in CloudCompare, color by 'tree_id' scalar field)")


# ================================================================================
# INTERACTIVE PARAMETER WIZARD
# ================================================================================
# Walks you through every tunable parameter in CONFIG above, one at a
# time, grouped into the same sections as the CONFIG comments. For each
# parameter you see what it does, its typical/valid range, and its
# current default -- press Enter to keep the default, or type a new value.
#
# Run:
#   python 03_individual_tree_detection.py             -> launches this wizard
#   python 03_individual_tree_detection.py --defaults   -> skips the wizard,
#                                                           runs CONFIG as-is
#
# At the end you get a summary table of everything you chose, and a chance
# to jump back and edit any single parameter by name before it actually runs.
# ================================================================================

# Each entry: (key, type, short description, range/choices or None, unit)
#   type is one of: "str", "str_or_none", "float", "float_or_none", "int",
#                    "bool", "choice"
# ---------------------------------------------------------------
# Two alternative "how do I find my input files" sections -- exactly one
# of these is used, depending on the answer to the batch_mode question
# asked first in run_wizard() below. Everything in COMMON_PARAM_SCHEMA
# applies identically either way (the same detection settings are used
# for every file in a batch run).
# ---------------------------------------------------------------
BATCH_PARAM_SCHEMA = [
    ("BATCH / MULTI-FILE SETUP (processing multiple plots' TLS + UAV files)", [
        ("plot_numbers", "int_list",
         "Which plot numbers to process, as a comma-separated list, e.g. "
         "'43,58,70' processes exactly those 3 plots (both their TLS and "
         "UAV file each) -- 6 files total for 3 plot numbers. Every plot "
         "number listed here must have BOTH a matching TLS file and a "
         "matching UAV file findable via the two filename patterns below, "
         "or that file is skipped with a printed warning (not a crash).",
         None, ""),
        ("tls_dir", "str",
         "Folder containing your TLS .las files (e.g. Plot43_MyDiv.las). "
         "'.' means the same folder this script is run from.", None, ""),
        ("uav_dir", "str",
         "Folder containing your UAV-LiDAR .las files (e.g. "
         "MyDiv_43_L1.las). Often the same folder as tls_dir -- '.' means "
         "the same folder this script is run from.", None, ""),
        ("tls_name_pattern", "str",
         "Filename pattern for TLS files, with {n} standing in for the "
         "plot number -- e.g. 'Plot{n}_MyDiv.las' matches Plot43_MyDiv.las, "
         "Plot58_MyDiv.las, etc. for plot_numbers 43, 58. Must contain "
         "the literal text '{n}' somewhere.", None, ""),
        ("uav_name_pattern", "str",
         "Same idea as tls_name_pattern but for the UAV files -- e.g. "
         "'MyDiv_{n}_L1.las' matches MyDiv_43_L1.las, MyDiv_58_L1.las, etc.",
         None, ""),
        ("batch_output_dir", "str",
         "Folder where ALL batch results go. A subfolder is created per "
         "plot+sensor combination, e.g. batch_output_dir/Plot43_TLS/, "
         "Plot43_UAV/, Plot58_TLS/, ... each containing that file's full "
         "normal output set (detected_trees.csv, trees_bboxes.png, etc.), "
         "plus a combined summary CSV and PDF report directly inside "
         "batch_output_dir itself.", None, ""),
        ("generate_pdf_report", "bool",
         "Build a combined PDF report at the end, comparing tree counts/"
         "heights/crown sizes across all processed plots and sensors, with "
         "a summary table, TLS-vs-UAV comparison charts, all detection "
         "images, and a written discussion of expected differences. If "
         "False, you still get the per-file outputs and the summary CSV, "
         "just no PDF.", None, ""),
        ("report_filename", "str",
         "Filename (not full path) for the PDF report, saved inside "
         "batch_output_dir.", None, ""),
        ("report_data_is_placeholder", "bool",
         "Show a disclosure banner at the top of the PDF stating the "
         "input data may be synthetic/placeholder rather than real scans. "
         "Set to No once you've confirmed you're running on your actual "
         "real .las files, to remove the banner.", None, ""),
    ]),
]

SINGLE_FILE_PARAM_SCHEMA = [
    ("INPUT / OUTPUT (single file)", [
        ("input_las", "str",
         "Path to the raw TLS/UAV .las file to process", None, ""),
        ("input_chm_tif", "str_or_none",
         "Path to an existing CHM GeoTIFF to reuse instead of building one "
         "from input_las (type 'none' to build from input_las instead)", None, ""),
        ("output_dir", "str",
         "Folder where all results are written", None, ""),
    ]),
]

PARAM_SCHEMA = [
    ("GROUND CLASSIFICATION (CSF)", [
        ("csf_scene", "choice",
         "Terrain roughness preset for the Cloth Simulation Filter",
         ["steep_slope", "relief", "flat"], ""),
        ("csf_cloth_resolution", "float",
         "Cloth grid spacing -- smaller follows finer terrain detail but is "
         "slower / more easily caught by low vegetation", (0.1, 5.0), "m"),
        ("csf_max_iterations", "int",
         "Max cloth simulation steps -- more lets the cloth fully settle "
         "onto the terrain", (50, 1000), ""),
        ("csf_class_threshold", "float",
         "Distance below the settled cloth within which a point counts as "
         "ground", (0.05, 2.0), "m"),
    ]),
    ("RASTERIZATION", [
        ("chm_res", "float",
         "CHM grid resolution used for tree DETECTION (not the true-color "
         "image -- see topview2_res below for that)", (0.01, 1.0), "m/px"),
    ]),
    ("CHM SMOOTHING", [
        ("smoothing_sigma_m", "float",
         "Gaussian blur applied before treetop search -- lower risks "
         "over-segmentation (noise treated as trees), higher risks merging "
         "close real trees together", (0.0, 1.5), "m"),
    ]),
    ("MINIMUM TREE HEIGHT", [
        ("min_height_m", "float",
         "Pixels below this height are never treated as a treetop "
         "(excludes grass/shrubs/noise)", (0.0, 10.0), "m"),
    ]),
    ("PEAK DETECTION METHOD", [
        ("peak_detection_method", "choice",
         "How treetops are found. h_maxima recommended for tightly-packed "
         "trees with only a shallow dip between crowns",
         ["fixed", "variable", "h_maxima"], ""),
        ("h_maxima_prominence_m", "float",
         "(h_maxima only) Minimum bump height above its connecting saddle "
         "to count as its own tree -- THE key knob for splitting merged "
         "close-together trees", (0.02, 2.0), "m"),
    ]),
    ("TREETOP SEARCH WINDOW (only used if peak_detection_method is fixed/variable)", [
        ("window_mode", "choice",
         "Constant search radius (fixed) or height-scaled radius (variable)",
         ["fixed", "variable"], ""),
        ("fixed_window_radius_m", "float",
         "(fixed mode) constant minimum spacing enforced between treetops",
         (0.1, 5.0), "m"),
        ("vwf_a", "float",
         "(variable mode) radius intercept: radius = a + b*height", (0.0, 3.0), "m"),
        ("vwf_b", "float",
         "(variable mode) radius slope per meter of height", (0.0, 1.0), ""),
        ("vwf_min_radius_m", "float",
         "(variable mode) minimum clamp on the computed radius", (0.05, 3.0), "m"),
        ("vwf_max_radius_m", "float",
         "(variable mode) maximum clamp on the computed radius", (0.5, 10.0), "m"),
    ]),
    ("CROWN DELINEATION (WATERSHED)", [
        ("th_seed", "float",
         "Minimum height fraction of a tree's own seed height a pixel must "
         "have to join that crown -- higher = smaller, tighter crowns",
         (0.05, 0.95), "fraction"),
        ("max_crown_radius_m", "float",
         "Hard cap on how far any single tree's crown can extend outward "
         "from its treetop -- prevents one tree 'eating' its neighbors",
         (0.1, 15.0), "m"),
    ]),
    ("3D POINT-TO-TREE ASSIGNMENT", [
        ("point_assignment_mode", "choice",
         "How individual 3D points get assigned to a tree ID",
         ["columnar", "conical"], ""),
        ("conical_top_radius_factor", "float",
         "(conical only) crown-radius multiplier applied at treetop height",
         (0.1, 3.0), ""),
        ("conical_bottom_radius_factor", "float",
         "(conical only) crown-radius multiplier applied near the ground",
         (0.0, 2.0), ""),
    ]),
    ("VISUALIZATION -- CHM/hillshade background (trees_bboxes.png / trees_crown_outlines.png)", [
        ("display_style", "choice",
         "Background rendering style for these two files",
         ["hillshade", "flat"], ""),
        ("display_use_smoothed", "bool",
         "Use the smoothed CHM (blurrier) instead of the raw CHM for the "
         "background image (does not affect detection either way)", None, ""),
        ("hillshade_azimuth_deg", "float",
         "Simulated sun compass direction for the hillshade effect", (0, 360), "deg"),
        ("hillshade_altitude_deg", "float",
         "Simulated sun height above the horizon", (0, 90), "deg"),
        ("chm_background_cmap", "str",
         "Matplotlib colormap name for the height tint, e.g. Greens_r, "
         "Greys_r, viridis", None, ""),
    ]),
    ("VISUALIZATION -- box coloring", [
        ("bbox_color_by", "choice",
         "How each detected tree's bounding box is colored",
         ["height", "tree_id", "confidence"], ""),
        ("low_confidence_crown_area_m2", "float",
         "Crown area below this is flagged as low-confidence (dashed box "
         "outline)", (0.01, 5.0), "m^2"),
    ]),
    ("OUTPUT EXTRAS", [
        ("save_per_tree_las", "bool",
         "Also save one separate .las file per detected tree "
         "(slower/disk-heavy for many trees)", None, ""),
    ]),
    ("TRUE-COLOR TOP VIEW (trees_bboxes2.png etc -- only if your .las has real RGB)", [
        ("topview2_res", "float_or_none",
         "Native pixel resolution of the true-color render, INDEPENDENT of "
         "chm_res -- smaller = sharper/more real detail, larger file. Type "
         "'none' to just reuse chm_res", (0.002, 0.2), "m/px"),
        ("topview2_smooth_px", "float",
         "Gaussian blur (in pixels of the topview2_res grid) for a softer, "
         "less pixelated look", (0.0, 5.0), "px"),
        ("topview2_dpi", "int",
         "Output PNG resolution -- 300 is print-quality", (72, 600), "dpi"),
    ]),
]


def _fmt(value):
    return "none" if value is None else str(value)


def _wrap(text, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _prompt_param(key, ptype, desc, constraint, unit, default):
    unit_str = f" {unit}" if unit else ""
    print(f"\n  * {key}")
    for line in _wrap(desc, 74):
        print(f"      {line}")

    if ptype == "choice":
        for i, choice in enumerate(constraint, 1):
            marker = "  <- default" if choice == default else ""
            print(f"      [{i}] {choice}{marker}")
    elif ptype == "bool":
        print(f"      options: yes / no")
    elif ptype in ("float", "int") and constraint:
        lo, hi = constraint
        print(f"      typical range: {lo}{unit_str} to {hi}{unit_str}")
    elif ptype == "float_or_none" and constraint:
        lo, hi = constraint
        print(f"      typical range: {lo}{unit_str} to {hi}{unit_str}  (or 'none' to disable)")
    elif ptype == "int_list":
        print("      format: comma-separated numbers, e.g. 43,58,70")

    while True:
        raw = input(f"      > value [{_fmt(default)}]: ").strip()
        if raw == "":
            return default
        try:
            if ptype == "choice":
                if raw.isdigit() and 1 <= int(raw) <= len(constraint):
                    return constraint[int(raw) - 1]
                if raw in constraint:
                    return raw
                raise ValueError
            elif ptype == "bool":
                if raw.lower() in ("y", "yes", "true", "1"):
                    return True
                if raw.lower() in ("n", "no", "false", "0"):
                    return False
                raise ValueError
            elif ptype == "int":
                return int(raw)
            elif ptype == "int_list":
                return [int(x.strip()) for x in raw.split(",") if x.strip()]
            elif ptype == "float":
                return float(raw)
            elif ptype == "float_or_none":
                if raw.lower() in ("none", "null", "-"):
                    return None
                return float(raw)
            elif ptype == "str_or_none":
                if raw.lower() in ("none", "null", "-"):
                    return None
                return raw
            else:  # "str"
                return raw
        except ValueError:
            print("      ! invalid input -- try again (or press Enter to keep the default)")


def _print_summary(cfg, schema=None):
    schema = schema if schema is not None else PARAM_SCHEMA
    print("\n" + "=" * 82)
    print("  SUMMARY OF SELECTED PARAMETERS")
    print("=" * 82)
    print(f"\n  PROCESSING MODE\n    {'batch_mode':32s} = {_fmt(cfg.get('batch_mode'))}")
    for section_title, params in schema:
        print(f"\n  {section_title}")
        for key, *_ in params:
            print(f"    {key:32s} = {_fmt(cfg[key])}")
    print("\n" + "=" * 82)


def run_wizard(base_cfg=None):
    cfg = dict(base_cfg or CONFIG)

    print("=" * 82)
    print("  INDIVIDUAL TREE DETECTION -- INTERACTIVE PARAMETER SETUP")
    print("=" * 82)
    print("  For each parameter: press Enter to keep the default shown in [brackets],")
    print("  or type a new value. Choices can be entered as their number or exact name.")

    print("\n" + "-" * 82)
    print("  PROCESSING MODE")
    print("-" * 82)
    cfg["batch_mode"] = _prompt_param(
        "batch_mode", "bool",
        "Process MULTIPLE plots' TLS + UAV files in one run (batch mode -- "
        "e.g. 6 files for 3 plot numbers, each TLS+UAV pair), or just ONE "
        "single .las file? Choose 'yes' for batch -- you'll then be asked "
        "for the plot numbers and file-naming patterns; the detection "
        "settings you set afterward (ground classification, peak detection, "
        "crown delineation, etc.) are applied IDENTICALLY to every file in "
        "the batch, since there's normally no reason to tune them "
        "differently per plot/sensor. Choose 'no' to process just one "
        "specific .las file (or one pre-built CHM GeoTIFF) instead.",
        None, "", cfg.get("batch_mode", True)
    )

    mode_schema = BATCH_PARAM_SCHEMA if cfg["batch_mode"] else SINGLE_FILE_PARAM_SCHEMA
    full_schema = mode_schema + PARAM_SCHEMA

    for section_title, params in full_schema:
        print("\n" + "-" * 82)
        print(f"  SECTION: {section_title}")
        print("-" * 82)
        for key, ptype, desc, constraint, unit in params:
            cfg[key] = _prompt_param(key, ptype, desc, constraint, unit, cfg.get(key))

    _print_summary(cfg, full_schema)

    all_known_schema = BATCH_PARAM_SCHEMA + SINGLE_FILE_PARAM_SCHEMA + PARAM_SCHEMA

    while True:
        choice = input(
            "\nProceed with these settings? [Enter/y = run, n = quit, "
            "or type a parameter name to edit it]: "
        ).strip()
        if choice == "" or choice.lower() in ("y", "yes"):
            return cfg
        if choice.lower() in ("n", "no", "q", "quit"):
            print("Aborted -- no files were generated.")
            raise SystemExit(0)

        key = choice
        found = False
        for section_title, params in all_known_schema:
            for pkey, ptype, desc, constraint, unit in params:
                if pkey == key:
                    cfg[key] = _prompt_param(key, ptype, desc, constraint, unit, cfg[key])
                    found = True
        if found:
            _print_summary(cfg, full_schema)
        else:
            print(f"  ! unknown parameter name '{key}' -- check spelling against the table above")


# ================================================================================
# MAIN
# ================================================================================

def process_single_file(cfg, filename_prefix=""):
    """
    Runs the full single-file individual tree detection pipeline (exactly
    what `main()` used to do end-to-end) and additionally returns a
    summary dict, so this can be called repeatedly across many files by
    `run_batch()` below. Behavior for a single, non-batch run is
    unchanged from before (filename_prefix defaults to "", i.e. no prefix).

    `filename_prefix` is prepended to every output FILENAME (not just the
    folder) -- e.g. "Plot43_TLS_" turns "trees_bboxes.png" into
    "Plot43_TLS_trees_bboxes.png". Used by run_batch() so that every file
    is self-identifying by plot number and sensor even if copied out of
    its per-run subfolder into one flat folder later -- belt-and-suspenders
    on top of the subfolder structure itself, which already prevents any
    actual overwriting between runs.
    """
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    fp = filename_prefix  # short alias, used a lot below

    chm, transform, rgb_info = get_chm(cfg)
    chm_smooth = smooth_chm(chm, transform, cfg["smoothing_sigma_m"])
    treetops_rc = detect_treetops(chm_smooth, transform, cfg)

    if len(treetops_rc) == 0:
        print("! No treetops detected -- try lowering 'min_height_m' or "
              "'smoothing_sigma_m', or check the CHM looks correct.")
        return {"n_trees": 0, "mean_height_m": np.nan, "max_height_m": np.nan,
                "min_height_m": np.nan, "mean_crown_area_m2": np.nan,
                "mean_crown_diameter_m": np.nan, "bboxes_png": None,
                "bboxes2_png": None}

    labels = segment_crowns(chm_smooth, treetops_rc, transform, cfg)
    res = transform.a
    rows = tree_metrics_table(labels, chm_smooth, treetops_rc, transform, res)

    # trees_bboxes.png / trees_crown_outlines.png / tree_labels.tif ALWAYS
    # use the original CHM/hillshade-based approach, regardless of whether
    # real RGB is available -- deliberately unchanged from before RGB
    # support was added. The true-color versions (when RGB is available)
    # are ADDITIONAL files with a "2" suffix -- both versions exist side
    # by side, one never replaces the other.
    save_trees_csv(rows, os.path.join(out_dir, f"{fp}detected_trees.csv"))
    bboxes_png = os.path.join(out_dir, f"{fp}trees_bboxes.png")
    plot_bboxes(chm, chm_smooth, rows, transform, cfg, bboxes_png)
    plot_crown_outlines(chm, chm_smooth, labels, rows, transform, cfg,
                         os.path.join(out_dir, f"{fp}trees_crown_outlines.png"))
    save_label_raster(labels, transform, os.path.join(out_dir, f"{fp}tree_labels.tif"))

    bboxes2_png = None
    if rgb_info is not None:
        xyz_norm_rgb, rgb01 = rgb_info
        background_rgb2 = render_photorealistic_topview(xyz_norm_rgb, rgb01, transform, chm.shape, cfg)
        bboxes2_png = os.path.join(out_dir, f"{fp}trees_bboxes2.png")
        plot_bboxes2(background_rgb2, rows, transform, chm.shape, cfg, bboxes2_png)
        plot_crown_outlines2(background_rgb2, labels, rows, transform, chm.shape, cfg,
                              os.path.join(out_dir, f"{fp}trees_crown_outlines2.png"))
        save_true_color_geotiff(background_rgb2, transform, os.path.join(out_dir, f"{fp}tree_labels2.tif"))
    else:
        print(f"      {fp}trees_bboxes2.png / {fp}trees_crown_outlines2.png / "
              f"{fp}tree_labels2.tif skipped -- no real RGB color data in "
              f"this .las (see the message printed in step [1/6] above)")

    # 3D point -> tree assignment (only possible if we built the CHM from a
    # raw LAS ourselves, since we need the actual normalized point cloud)
    if cfg["input_las"] and not cfg["input_chm_tif"]:
        xyz, las = tc.read_las(cfg["input_las"])
        xyz_clean, _ = tc.sor_filter(xyz, k=8, std_ratio=1.0)
        ground_mask = tc.classify_ground_csf(
            xyz_clean, scene=cfg["csf_scene"],
            cloth_resolution=cfg["csf_cloth_resolution"],
            max_iterations=cfg["csf_max_iterations"],
            class_threshold=cfg["csf_class_threshold"],
        )
        _, _, _, ground_interp = tc.rasterize_terrain_tin(xyz_clean[ground_mask], res=cfg["chm_res"])
        xyz_norm = tc.normalize_height(xyz_clean, ground_interp)
        xyz_norm = xyz_norm[xyz_norm[:, 2] >= 0]

        if cfg["point_assignment_mode"] == "columnar":
            tree_ids = assign_points_columnar(xyz_norm, labels, transform, res)
        else:
            tree_ids = assign_points_conical(xyz_norm, rows, cfg)

        save_classified_las(xyz_norm, tree_ids,
                             os.path.join(out_dir, f"{fp}points_with_tree_id.las"))

        if cfg["save_per_tree_las"]:
            per_tree_dir = os.path.join(out_dir, "per_tree_las")
            os.makedirs(per_tree_dir, exist_ok=True)
            for r in rows:
                tid = r["tree_id"]
                mask = tree_ids == tid
                if mask.sum() > 0:
                    save_classified_las(
                        xyz_norm[mask], tree_ids[mask],
                        os.path.join(per_tree_dir, f"{fp}tree_{tid:03d}.las")
                    )

    heights = [r["height_m"] for r in rows]
    print(f"\nDone. {len(rows)} trees detected. Outputs in: {out_dir}")
    print(f"  height range: {min(heights):.2f} - {max(heights):.2f} m "
          f"(mean {np.mean(heights):.2f} m)")

    return {
        "n_trees": len(rows),
        "mean_height_m": float(np.mean(heights)),
        "max_height_m": float(np.max(heights)),
        "min_height_m": float(np.min(heights)),
        "mean_crown_area_m2": float(np.mean([r["crown_area_m2"] for r in rows])),
        "mean_crown_diameter_m": float(np.mean([r["crown_diameter_m"] for r in rows])),
        "bboxes_png": bboxes_png,
        "bboxes2_png": bboxes2_png,
    }


# ================================================================================
# BATCH MODE: process multiple plots' TLS + UAV data, then build a PDF report
# ================================================================================

def run_batch(cfg):
    """
    Runs process_single_file() once per (plot_number x sensor) combination
    -- e.g. for plot_numbers=[43,58,70], that's 6 runs: Plot43 TLS,
    Plot43 UAV, Plot58 TLS, Plot58 UAV, Plot70 TLS, Plot70 UAV. Each run's
    full output (detected_trees.csv, trees_bboxes.png, etc.) is saved into
    its own subfolder of `batch_output_dir`, exactly as a single-file run
    would produce, so nothing about the per-file outputs changes -- this
    just orchestrates many single-file runs and collects a summary.
    """
    base_out = cfg["batch_output_dir"]
    os.makedirs(base_out, exist_ok=True)

    summary_rows = []
    for n in cfg["plot_numbers"]:
        for sensor, name_pattern, dir_key in [
            ("TLS", cfg["tls_name_pattern"], "tls_dir"),
            ("UAV", cfg["uav_name_pattern"], "uav_dir"),
        ]:
            las_path = os.path.join(cfg[dir_key], name_pattern.format(n=n))
            print(f"\n{'=' * 70}\nBATCH: Plot {n} - {sensor}  ({las_path})\n{'=' * 70}")
            if not os.path.exists(las_path):
                print(f"  ! skipped -- file not found: {las_path}")
                continue

            sub_cfg = dict(cfg)
            sub_cfg["input_las"] = las_path
            sub_cfg["input_chm_tif"] = None
            sub_out_dir = os.path.join(base_out, f"Plot{n}_{sensor}")
            sub_cfg["output_dir"] = sub_out_dir

            try:
                prefix = f"Plot{n}_{sensor}_"
                summary = process_single_file(sub_cfg, filename_prefix=prefix)
                summary["plot_number"] = n
                summary["sensor"] = sensor
                summary_rows.append(summary)
            except Exception as e:
                print(f"  ! FAILED Plot{n} {sensor}: {e}")

    save_batch_summary_csv(summary_rows, os.path.join(base_out, "tree_detection_summary.csv"))

    if cfg.get("generate_pdf_report", True):
        build_tree_detection_report(
            summary_rows, cfg, os.path.join(base_out, cfg.get("report_filename", "Tree_Detection_Report.pdf"))
        )

    print(f"\nBatch done. {len(summary_rows)} plot/sensor runs completed. See: {base_out}")
    return summary_rows


def save_batch_summary_csv(rows, path):
    if not rows:
        print("      ! no batch results to save")
        return
    keys = ["plot_number", "sensor", "n_trees", "mean_height_m", "max_height_m",
            "min_height_m", "mean_crown_area_m2", "mean_crown_diameter_m"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
    print(f"      saved {path}")


def _img_flowable(path, width=15 * cm, max_height=22 * cm):
    """
    Embeds an image at the given target width, computing height from the
    image's ACTUAL pixel aspect ratio (read via PIL) so it is never
    stretched/distorted regardless of the source figure's real shape.
    """
    if not path or not os.path.exists(path):
        return None
    from PIL import Image as PILImage
    with PILImage.open(path) as pil_img:
        px_w, px_h = pil_img.size
    aspect = px_w / px_h
    height = width / aspect
    if height > max_height:
        height = max_height
        width = height * aspect
    img = RLImage(path, width=width, height=height)
    img.hAlign = "CENTER"
    return img


def _make_batch_comparison_charts(rows, out_dir):
    """Builds 2 summary bar charts (n_trees, mean height) comparing TLS vs UAV per plot."""
    import collections
    by_plot = collections.defaultdict(dict)
    for r in rows:
        by_plot[r["plot_number"]][r["sensor"]] = r
    plot_numbers = sorted(by_plot.keys())

    def _bar_chart(metric_key, ylabel, filename, title):
        tls_vals = [by_plot[n].get("TLS", {}).get(metric_key, np.nan) for n in plot_numbers]
        uav_vals = [by_plot[n].get("UAV", {}).get(metric_key, np.nan) for n in plot_numbers]
        x = np.arange(len(plot_numbers))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, tls_vals, width, label="TLS", color="tab:green")
        ax.bar(x + width / 2, uav_vals, width, label="UAV", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Plot{n}" for n in plot_numbers])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=250)
        plt.close()
        return path

    n_trees_path = _bar_chart("n_trees", "Number of trees detected",
                               "compare_n_trees.png", "Detected tree count: TLS vs UAV")
    height_path = _bar_chart("mean_height_m", "Mean detected tree height (m)",
                              "compare_mean_height.png", "Mean detected tree height: TLS vs UAV")
    return n_trees_path, height_path


def build_tree_detection_report(rows, cfg, path):
    """
    Compiles a PDF report summarizing individual tree detection results
    across all processed plot/sensor combinations: a summary table,
    TLS-vs-UAV comparison charts, one bounding-box overlay image per
    plot/sensor for visual reference, and a discussion of expected
    TLS-vs-UAV differences in tree detection specifically (distinct from,
    but consistent with, the CHM-level discussion in
    02_uav_tls_chm_comparison.py's own report).
    """
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RTitle", fontSize=20, leading=24, alignment=TA_CENTER,
                               spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="RSubtitle", fontSize=11, leading=14, alignment=TA_CENTER,
                               textColor=colors.grey, spaceAfter=20))
    styles.add(ParagraphStyle(name="RSection", fontSize=15, leading=18, spaceBefore=18,
                               spaceAfter=8, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#1a4d2e")))
    styles.add(ParagraphStyle(name="RBody", fontSize=10, leading=14.5, spaceAfter=8))
    styles.add(ParagraphStyle(name="RCaption", fontSize=8.5, leading=11, textColor=colors.grey,
                               alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(name="RDisclosure", fontSize=9.5, leading=13,
                               textColor=colors.HexColor("#7a1f1f"), spaceAfter=10,
                               backColor=colors.HexColor("#fdf0f0")))

    story = []
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Individual Tree Detection Report -- TLS vs. UAV-LiDAR", styles["RTitle"]))
    plot_list = ", ".join(f"Plot{n}" for n in cfg["plot_numbers"])
    story.append(Paragraph(f"Plots analyzed: {plot_list}", styles["RSubtitle"]))

    if cfg.get("report_data_is_placeholder", True):
        story.append(Paragraph(
            "<b>Data disclosure:</b> if the .las files in this run are synthetic/placeholder "
            "stand-ins rather than your real scans, treat the tree counts and sizes below as "
            "illustrative only. Set <b>report_data_is_placeholder: False</b> in CONFIG once "
            "you've confirmed these are your real files, to remove this banner.",
            styles["RDisclosure"]
        ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Individual tree detection (CHM-based local-maxima + watershed crown delineation, "
        "see the script's own header for the full method) was run independently on both the "
        "TLS and UAV-LiDAR point cloud for each plot, using each sensor's own resolution and "
        "the same detection parameters otherwise. This report compares what each sensor "
        "detects for the same physical plots.",
        styles["RBody"]
    ))
    story.append(PageBreak())

    # ---------------- Summary table ----------------
    story.append(Paragraph("Summary: Detected Trees per Plot and Sensor", styles["RSection"]))
    header = ["Plot", "Sensor", "N trees", "Mean height (m)", "Max height (m)",
              "Mean crown area (m2)", "Mean crown diam. (m)"]
    table_rows = []
    for r in sorted(rows, key=lambda r: (r["plot_number"], r["sensor"])):
        table_rows.append([
            f"Plot{r['plot_number']}", r["sensor"], str(r["n_trees"]),
            f"{r['mean_height_m']:.2f}" if not np.isnan(r["mean_height_m"]) else "n/a",
            f"{r['max_height_m']:.2f}" if not np.isnan(r["max_height_m"]) else "n/a",
            f"{r['mean_crown_area_m2']:.2f}" if not np.isnan(r["mean_crown_area_m2"]) else "n/a",
            f"{r['mean_crown_diameter_m']:.2f}" if not np.isnan(r["mean_crown_diameter_m"]) else "n/a",
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

    # ---------------- Comparison charts ----------------
    out_dir = os.path.dirname(path)
    n_trees_chart, height_chart = _make_batch_comparison_charts(rows, out_dir)
    img = _img_flowable(n_trees_chart, width=13 * cm)
    if img:
        story.append(img)
        story.append(Paragraph("Detected tree count per plot, TLS vs. UAV", styles["RCaption"]))
    img = _img_flowable(height_chart, width=13 * cm)
    if img:
        story.append(img)
        story.append(Paragraph("Mean detected tree height per plot, TLS vs. UAV", styles["RCaption"]))
    story.append(PageBreak())

    # ---------------- Per-plot bbox images ----------------
    story.append(Paragraph("Detected Trees by Plot and Sensor", styles["RSection"]))
    for n in cfg["plot_numbers"]:
        for r in rows:
            if r["plot_number"] == n:
                img = _img_flowable(r.get("bboxes_png"), width=15 * cm)
                if img:
                    story.append(img)
                    story.append(Paragraph(
                        f"Plot {n} - {r['sensor']} ({r['n_trees']} trees detected)", styles["RCaption"]
                    ))
    story.append(PageBreak())

    # ---------------- Discussion ----------------
    story.append(Paragraph("Discussion: TLS vs. UAV-LiDAR Tree Detection Differences", styles["RSection"]))

    import collections
    by_plot = collections.defaultdict(dict)
    for r in rows:
        by_plot[r["plot_number"]][r["sensor"]] = r
    diffs = []
    for n, d in by_plot.items():
        if "TLS" in d and "UAV" in d:
            diffs.append(d["TLS"]["n_trees"] - d["UAV"]["n_trees"])
    mean_diff = np.mean(diffs) if diffs else float("nan")

    story.append(Paragraph(
        f"On average across these plots, TLS detected {mean_diff:+.1f} more trees per plot than "
        f"UAV (positive = TLS detects more). At least three factors likely explain any "
        f"systematic difference in detected tree count and size between the two sensors:",
        styles["RBody"]
    ))
    story.append(Paragraph(
        "<b>Resolution and CHM smoothing:</b> TLS is typically rasterized at a much finer "
        "resolution than UAV (e.g. 0.05 m vs. 0.15 m here). A finer CHM preserves more small, "
        "closely-spaced local maxima -- meaning TLS can resolve individual small/young trees "
        "that a coarser, smoother UAV CHM merges into one detection (under-segmentation on "
        "the UAV side), especially in a densely-planted young sapling plot.", styles["RBody"]
    ))
    story.append(Paragraph(
        "<b>Occlusion and noise near the scanner:</b> TLS point density is very high close to "
        "the scan position and drops off with distance/occlusion -- this can create both "
        "false positives (noise-driven small bumps mistaken for extra trees close to the "
        "scanner) and false negatives (real but occluded/distant trees never registering a "
        "clear treetop). UAV's more spatially uniform point density avoids this particular "
        "bias, at the cost of being unable to distinguish very close, small crowns.",
        styles["RBody"]
    ))
    story.append(Paragraph(
        "<b>True canopy-top visibility:</b> UAV-LiDAR looks straight down and reliably captures "
        "the true top of each crown; TLS looks up/outward from the ground and can "
        "systematically underestimate treetop height due to occlusion (consistent with "
        "02_uav_tls_chm_comparison.py's own finding that UAV CHM height exceeds TLS CHM "
        "height at every plot in that comparison) -- so even where both sensors agree on "
        "\"there is a tree here,\" the TLS-estimated height for that same tree is expected to "
        "run lower than the UAV estimate.", styles["RBody"]
    ))
    story.append(Paragraph(
        "As with the CHM-level comparison, disentangling genuine detection differences from "
        "pure sensor-geometry/resolution artifacts would require independent field-measured "
        "tree positions (a stem map) to check which sensor's detections are actually correct "
        "-- without that, this comparison shows where the two sensors agree/disagree, not "
        "which one is more accurate.", styles["RBody"]
    ))

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    doc.build(story)
    print(f"      saved {path}")





def main(cfg=CONFIG):
    if cfg.get("batch_mode"):
        return run_batch(cfg)
    else:
        return process_single_file(cfg)


if __name__ == "__main__":
    import sys
    if "--defaults" in sys.argv:
        main(CONFIG)
    else:
        chosen_cfg = run_wizard(CONFIG)
        main(chosen_cfg)