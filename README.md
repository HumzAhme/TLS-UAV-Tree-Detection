# TLS / UAV-LiDAR Point Cloud Analysis Toolkit

A set of Python scripts for processing terrestrial laser scanning (TLS) and
UAV-LiDAR point clouds of forest/tree plots: preprocessing, ground
classification, terrain/canopy models, foliage-density metrics, TLS-vs-UAV
comparison, and individual tree detection & crown delineation.

## What's in here

| File | What it does |
|---|---|
| `tls_pipeline.py` | Standalone: clean a single `.las` → DTM/DSM/CHM + ground mesh |
| `tls_core.py` | Shared library (ground classification, TIN/DTM, height normalization, CHM, voxelization, LAI parsing) — never run directly, imported by `01`/`02` |
| `01_tls_foliage_density_pipeline.py` | Batch TLS processing → foliage-density metrics (FHD, RH, canopy volume) + LAI comparison |
| `02_uav_tls_chm_comparison.py` | Compares TLS-derived CHMs against UAV-LiDAR CHMs across plots |
| `03_individual_tree_detection.py` | **Individual tree detection & crown delineation** — the main, most feature-complete script (see below) |
| `03_individual_tree_detection_v1_no_rgb.py` | A lighter snapshot of `03` from before RGB/true-color support was added — same detection logic, no color rendering, no wizard |
| `point_cloud_glossary.md` | Full glossary of every term used across all scripts (TLS, CSF, DTM/DSM/CHM, FHD, LAI, hillshade, watershed, etc.) |
| `OUTPUTS_EXPLAINED.md` | File-by-file, column-by-column explanation of everything each script generates |
| `requirements.txt` | `pip install -r requirements.txt` |

Install once:
```bash
pip install -r requirements.txt
```

---

## 1. `tls_pipeline.py` — basic preprocessing

Takes one raw `.las` file and produces a cleaned point cloud, ground/
vegetation split, DTM, DSM, CHM, and a ground mesh. Edit the `CONFIG` dict
at the top (`input_las`, `output_dir`), then:
```bash
python tls_pipeline.py
```
See `point_cloud_glossary.md` for what every one of these outputs means.

## 2. `01_tls_foliage_density_pipeline.py` — foliage density + LAI

Batch-processes every `Plot*_MyDiv.las` in a folder: ground classification
→ DTM → height normalization → CHM → voxelization → foliage-density
metrics (FHD, RH25/50/75/98, canopy volume, occupancy index) → joins with
LI-COR `.TXT` LAI files by plot number → correlation plots. Edit `CONFIG`
(`las_dir`, `lai_dir`, `epsg_code`), then:
```bash
python 01_tls_foliage_density_pipeline.py
```

## 3. `02_uav_tls_chm_comparison.py` — TLS vs. UAV-LiDAR

For each plot, builds a TLS CHM and a UAV-LiDAR CHM (using the UAV file's
existing ground classification), resamples them onto a common grid, and
compares them (difference raster, SD comparison, height-distribution
boxplots). Only useful if you have **both** a TLS and a UAV scan of the
same plot — skip this one if you only have TLS data. Edit `CONFIG`
(`tls_dir`, `uav_dir`, `plot_numbers`), then:
```bash
python 02_uav_tls_chm_comparison.py
```

---

## 4. `03_individual_tree_detection.py` — individual tree detection (main script)

Detects individual trees in a TLS/UAV point cloud and draws a bounding box
+ crown outline around each one — the "squares around each tree" task.
Implements the CHM-based local-maxima + marker-controlled watershed family
of methods (Popescu & Wynne 2004, Chen et al. 2006, Dalponte & Coomes
2016), with an added H-maxima (prominence-based) peak detector for
correctly splitting tightly-packed, similarly-tall trees that simpler
fixed/variable-radius methods merge together. The script's own header
comment has a full literature review and a comparison of what commercial/
open-source tools (lidR, FUSION, ArcGIS Pro, LiDAR360, DeepForest, TreeLS)
use — read that for the full "why," this README is the "how."

### Running it

```bash
python 03_individual_tree_detection.py              # interactive wizard (recommended)
python 03_individual_tree_detection.py --defaults    # skip the wizard, just use CONFIG as-is
```

**The wizard** walks you through all ~34 tunable parameters, grouped into
13 sections, one at a time — shows what each one does, its typical range
(or numbered choices), and its current default in `[brackets]`. Press
Enter to keep the default, or type a new value. At the end you get a full
summary and a chance to jump back and edit any single parameter by name
before anything actually runs:
```
Proceed with these settings? [Enter/y = run, n = quit, or type a parameter name to edit it]:
```
If you'd rather skip all the prompts, edit the `CONFIG` dict directly at
the top of the script and run with `--defaults`.

### Parameter reference (quick summary — full detail is in the script itself)

| Section | Key parameters | What they control |
|---|---|---|
| Input/Output | `input_las`, `input_chm_tif`, `output_dir` | Source file and where results go |
| Ground classification | `csf_scene`, `csf_cloth_resolution`, `csf_max_iterations`, `csf_class_threshold` | How ground points are separated from vegetation (CSF algorithm) |
| Rasterization | `chm_res` | CHM grid resolution used for **detection** |
| CHM smoothing | `smoothing_sigma_m` | Noise reduction before treetop search |
| Minimum height | `min_height_m` | Excludes grass/shrubs from being treated as trees |
| Peak detection | `peak_detection_method`, `h_maxima_prominence_m` | **The #1 knob for split/merge issues** — `h_maxima` + low `h_maxima_prominence_m` splits close trees; higher values merge them |
| Legacy search window | `window_mode`, `fixed_window_radius_m`, `vwf_*` | Older fixed/variable-radius alternative to h_maxima |
| Crown delineation | `th_seed`, `max_crown_radius_m` | How far each tree's crown is allowed to grow — lower `max_crown_radius_m` fixes overlapping/oversized boxes |
| Point-to-tree assignment | `point_assignment_mode`, `conical_*` | How 3D points get labeled with a tree ID |
| CHM/hillshade visuals | `display_style`, `display_use_smoothed`, `hillshade_*`, `chm_background_cmap` | Look of `trees_bboxes.png` / `trees_crown_outlines.png` |
| Box coloring | `bbox_color_by`, `low_confidence_crown_area_m2` | How boxes are colored so overlapping ones stay distinguishable |
| True-color view | `topview2_res`, `topview2_smooth_px`, `topview2_dpi` | Sharpness/resolution of the `*2` true-color outputs (only if your `.las` has real RGB) |

### Outputs

Every visual/GeoTIFF product comes in **two versions**: the original
(CHM/height-based, always generated) and, only if your `.las` has real
color data, a `2`-suffixed true-color version. One never replaces the
other — you get both.

| Original (always) | True-color counterpart (`2`, only if RGB exists) |
|---|---|
| `trees_bboxes.png` | `trees_bboxes2.png` |
| `trees_crown_outlines.png` | `trees_crown_outlines2.png` |
| `tree_labels.tif` | `tree_labels2.tif` |

Plus: `detected_trees.csv` (per-tree stats — position, height, crown
area/diameter, bounding box), `points_with_tree_id.las` (your point cloud
with a `tree_id` scalar field, open in CloudCompare), and optionally
`per_tree_las/tree_XXX.las` per tree if `save_per_tree_las: True`.

See `OUTPUTS_EXPLAINED.md` for a column-by-column breakdown of the CSV
and every other file this whole toolkit produces.

### Troubleshooting

| Symptom | Likely fix |
|---|---|
| Several real trees detected as one | Lower `h_maxima_prominence_m` (try 0.05–0.1) |
| Too many fake tiny trees appear | Raise `h_maxima_prominence_m` (try 0.3–0.5) |
| Boxes massively overlap / look chaotic | Lower `max_crown_radius_m` to a realistic crown radius for your trees |
| Background looks gray/flat | That's the CHM (a height grid, not a photo) — see `chm_background_cmap`/`display_style`, or check if `trees_bboxes2.png` was generated (needs real RGB in your `.las`) |
| `trees_bboxes2.png` etc. missing | Your `.las` has no real RGB color data — check the console message from step `[1/6]`, which says explicitly whether color was found |
| `trees_bboxes2.png` looks low-res/blurry | Lower `topview2_res` (e.g. 0.01 or 0.005) and/or raise `topview2_dpi` |
| Script re-run doesn't reflect your edits | Restart your Python session/kernel — a stale cached module import can silently keep running old code; running via `python script.py` fresh each time avoids this |

---

## Recommended workflow

1. Run `tls_pipeline.py` once on a single plot to sanity-check ground
   classification and get a first CHM.
2. Run `03_individual_tree_detection.py` (wizard mode) on that same plot,
   tune `h_maxima_prominence_m` and `max_crown_radius_m` against what you
   see in `trees_bboxes.png` until the tree count/boxes look right.
3. If you need vertical structure / LAI comparison across all plots, run
   `01_tls_foliage_density_pipeline.py`.
4. If you also have matching UAV-LiDAR scans, run
   `02_uav_tls_chm_comparison.py`.

---

## What could be improved next

Ideas for extending this toolkit further, roughly in order of likely
impact for a TLS sapling-plot use case like MyDiv:

**Detection accuracy**
- **Stem/trunk detection** (TLS-specific): slice the point cloud at
  breast height (~1.3 m), cluster and circle-fit trunk cross-sections to
  get exact stem positions/diameters, then assign crowns to the nearest
  stem instead of relying purely on a top-down CHM. More reliable than
  CHM-watershed once trees have a clear trunk, and gives you DBH for free.
  (See the `TreeLS` R package / QSM tools mentioned in the script's
  literature review.)
- **Auto-tuning `h_maxima_prominence_m` / `max_crown_radius_m`**: instead
  of manual trial-and-error, sweep a range of values and pick the one that
  best matches a known ground-truth tree count (if you have field-counted
  stems per plot) or that minimizes a segmentation-quality heuristic.
- **3D point-based clustering** (Ferraz et al. 2016-style adaptive mean
  shift) as a genuine alternative to CHM+watershed, better suited to
  overlapping/interlocking crowns than a 2D top-down projection.

**Validation**
- Compare `detected_trees.csv` tree counts/positions against manually
  counted or field-surveyed stems per plot to quantify precision/recall,
  not just visually inspect the boxes.
- Cross-check detected tree heights against field-measured heights (if
  available) to calibrate for the systematic under-estimation that CHM
  smoothing introduces (documented in the script's own comments).

**Species / ecological richness**
- If species-level data exists per plot (MyDiv is a tree-diversity
  experiment), join `detected_trees.csv` with known planting maps to
  check whether detected tree counts/positions match expected species
  composition and stocking density per plot.
- If a hyperspectral or multispectral image of the same plots becomes
  available, fuse it with the LiDAR-derived crown segments (use crown
  polygons as zonal boundaries) for per-tree species classification.

**Performance / scale**
- Vectorize/parallelize the batch loop in `01`/`02` across all 12 plots
  (currently sequential) using `multiprocessing` or `concurrent.futures`.
- For very large or many-plot point clouds, switch the point-to-tree
  assignment step from the current O(points) pass to a KD-tree/spatial
  index for faster nearest-crown lookups, especially for `"conical"` mode.

**Output / integration**
- Export `detected_trees.csv` bounding boxes / crown polygons as an actual
  GIS vector format (Shapefile/GeoJSON) instead of just a CSV + raster, so
  results can be loaded directly into QGIS alongside other layers.
- Add a confidence/uncertainty score per tree (e.g. based on crown
  compactness, point density, or agreement between `"columnar"` and
  `"conical"` point assignment) rather than the current binary
  low-confidence-by-area flag.
- If DeepForest becomes usable (i.e. you obtain an RGB drone orthomosaic
  of the same plots), add it as a genuine alternative detection method and
  cross-validate its boxes against the CHM-watershed results.

**Multi-temporal**
- If TLS scans of the same plots exist at multiple dates, track individual
  trees across scans (matching by nearest position + similar height) to
  estimate growth rates per tree, not just a single-date snapshot.
