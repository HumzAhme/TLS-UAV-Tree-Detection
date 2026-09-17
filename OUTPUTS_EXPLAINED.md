# What Every Generated File & Column Means

You ran `tls_pipeline.py` (Part 1) and `01_tls_foliage_density_pipeline.py`
(Part 2, foliage-density + LAI). This file walks through **every single
output** they produce — what it is, what it's used for, and how to read it.

> **Bug fix note**: if you saw a file named `Plot431_...` instead of
> `Plot43_...`, that was a naming bug in `01_tls_foliage_density_pipeline.py`
> — it was concatenating *every* digit found anywhere in the filename
> (`MyDiv_43_L1.las` → digits `4`,`3`,`1` → `"431"`) instead of reading the
> plot number as one group. This has been fixed (re-download the script) —
> it now correctly reads `MyDiv_43_L1.las` → `Plot43`. Just re-run the
> script and your files will be named `Plot43_...` from now on.

---

## Part 1 outputs — `tls_pipeline.py`

These all land directly in your `output_dir`.

| File | What it is |
|---|---|
| **`01_sor_filtered.las`** | Your original point cloud, minus the outlier/noise points removed by SOR filtering (see "SOR" below). This is the "cleaned" cloud everything downstream is built from. |
| **`02_ground_points.las`** | Only the points the CSF algorithm classified as **ground** (bare soil), tagged with LAS classification code `2` (the official ASPRS code for ground). Open this in CloudCompare/QGIS to visually check whether the ground classification looks right (should look like a thin, continuous surface layer, no floating clumps). |
| **`03_nonground_vegetation_points.las`** | Everything else — trunks, branches, leaves, understory — tagged with classification code `1` (unclassified/other). This is your "vegetation" cloud. |
| **`DTM.tif`** | Digital Terrain Model — a raster (grid image) where every cell holds the **bare-ground elevation** at that location, built only from `02_ground_points.las`. |
| **`DTM_preview.png`** | A quick-look colored image of the DTM so you can eyeball it without GIS software. Should look like a smooth, gently varying surface (the actual ground shape of your plot). |
| **`DSM.tif`** | Digital Surface Model — a raster where every cell holds the elevation of the **highest thing present** (treetop, branch, or bare ground if nothing is above it). Built from *all* points, not just ground. |
| **`CHM.tif`** | Canopy Height Model = `DSM − DTM`. This cancels out the ground's own elevation/slope and leaves just **vegetation height above the ground** at every grid cell — the main output you actually want for "how tall is the canopy here?" |
| **`CHM_preview.png`** / **`DSM_preview.png`** | Same idea as the DTM preview — quick colored images so you can see the shape of the canopy/surface without opening GIS software. In the CHM preview, bright/warm colors = tall vegetation, dark = ground level (~0 m). |
| **`ground_mesh.obj`** | A 3D mesh (triangulated surface) built by connecting the ground points into triangles, instead of leaving them as disconnected dots. You can open this in Blender, MeshLab, or CloudCompare to inspect the ground surface as a continuous shape — useful for spotting holes or spikes that indicate a bad ground classification. |
| **`ground_mesh_preview.png`** | A rendered image of that mesh, viewed in 3D, so you don't need separate 3D software just to check what it looks like. |

---

## Part 2 outputs — `01_tls_foliage_density_pipeline.py`

Everything lands in `tls_results/` (or whatever you set `out_dir` to).

### Per-plot files

| File | What it is |
|---|---|
| **`dtm/Plot43_DTM.tif`** | Same concept as Part 1's `DTM.tif`, but for this specific plot (here, using TIN interpolation instead of grid-binning — slightly more accurate, see glossary). |
| **`chm/Plot43_CHM.tif`** | Same concept as Part 1's `CHM.tif` — canopy height above ground, for this plot. |
| **`Plot43_vertical_profile.csv`** | The **vertical foliage profile**: how much vegetation structure exists at each height slice (default 3 m-thick slabs) through the canopy. This is the data behind the FHD calculation. Columns explained below. |

### Summary tables (one row per plot)

| File | What it is |
|---|---|
| **`tls_metrics.csv`** | The master table — one row per plot, every structural metric computed from the point cloud (no LAI yet). |
| **`LAI_data.csv`** | One row per LI-COR `.TXT` file you had in your `lai_dir`, with the parsed `LAI` value and the `plot_id` it was matched to. |
| **`TLS_LAI_comparison.csv`** | `tls_metrics.csv` + the matched `LAI` value + species-richness group (`SR`) joined in — this is the table you'd actually use for the correlation analysis and to answer the assignment questions. |
| **`LAI_vs_FHD.png`, `LAI_vs_max_height.png`, `LAI_vs_canopy_volume_per_ground_area.png`, `LAI_vs_chm_mean.png`, `LAI_vs_RH98.png`** | Scatter plots of field-measured LAI (y-axis) against each TLS-derived structural metric (x-axis), with a fitted trend line and the Pearson correlation `r` in the title — visual answer to "which metric best predicts LAI?" (Only generated if you have ≥3 plots with matched LAI data.) |
| **`TLS_assignment_answers.md`** | Full written answers to the 5 assignment questions from the handout. |

### Column-by-column meaning of `tls_metrics.csv` / `TLS_LAI_comparison.csv`

| Column | Meaning |
|---|---|
| `plot_id` | Which plot this row belongs to (e.g. `Plot43`) |
| `las_file` | The original filename processed |
| `n_points` | Total number of points in the raw file |
| `n_ground_points` | How many of those were classified as ground by CSF |
| `FHD` | **Foliage Height Diversity** (raw Shannon entropy) — how evenly foliage is spread across height layers. Higher = more vertically complex/layered canopy. Lower = foliage concentrated in one narrow height band. |
| `FHD_std` | Same idea, but rescaled to always fall between 0 and 1, so it's comparable even if plots have different numbers of occupied height layers. |
| `chm_mean` | Average canopy height (m) across the whole plot's CHM raster |
| `chm_sd` | Standard deviation of canopy height (m) — how *variable* the canopy height is across the plot. A high value means a mix of tall and short areas (gaps, uneven canopy); a low value means a uniform canopy height. |
| `chm_max` | Tallest point in the CHM raster (m) — essentially the tallest tree/canopy point detected. |
| `n_occupied_voxels` | How many 3D voxels (see "voxelization" below) actually contain at least one laser point — a raw measure of how much 3D space has *any* detected structure in it. |
| `n_possible_voxels_fixed_volume` | The total number of voxels that *could* exist within a fixed reference volume (your configured plot width × length × height, divided into voxel-sized cells) — the denominator used to normalize `n_occupied_voxels` into a density index. |
| `occupancy_index_fixed_volume` | `n_occupied_voxels / n_possible_voxels_fixed_volume` — the fraction of the reference volume that's actually "filled" with vegetation structure. A density index that's comparable across plots regardless of their absolute canopy volume. |
| `canopy_volume_m3` | Total 3D volume (m³) occupied by vegetation = `n_occupied_voxels × voxel volume`. |
| `canopy_volume_per_ground_area` | `canopy_volume_m3 ÷ plot footprint area` — canopy volume normalized by plot size, so plots of different footprints are comparable. This is usually the metric most strongly correlated with field LAI, since both describe "how much plant material is packed in," just measured in 3D vs. as a 2D index. |
| `max_height` | Tallest occupied voxel's height (m) — a voxel-based alternative to `chm_max`. |
| `mean_height` | Average height (m) of all occupied voxels — note this is different from `chm_mean` (which averages a 2D raster of max heights per cell); this one averages the height of every occupied 3D cell, so it's more influenced by how much material sits at each height, not just the top surface. |
| `RH25`, `RH50`, `RH75`, `RH98` | **Relative Height percentiles.** RH50 = the height below which 50% of the cumulative vertical foliage/plant-area profile sits (like a median height of "where the plant material is"); RH98 = a robust proxy for "canopy top height" (more stable than the raw max, which can be thrown off by one stray high point). Read them together: if RH25/RH50/RH75/RH98 are all clustered close together, most foliage sits in a narrow height band near the top (simple canopy); if they're spread far apart, foliage is distributed across many heights (complex, layered canopy) — this is the numeric version of what FHD summarizes into one number. |
| `LAI` | The field-measured Leaf Area Index from the matched LI-COR `.TXT` file (only in `TLS_LAI_comparison.csv`) |
| `SR` | Species-richness group of the plot (`1SR`/`2SR`/`4SR`), from the hardcoded `SR_GROUPS` mapping in the script — edit this dictionary if your plot list differs |

### Column-by-column meaning of `Plot43_vertical_profile.csv`

This is the raw data behind the FHD number — one row per height layer (default 3 m-thick slabs) that contains at least one occupied voxel.

| Column | Meaning |
|---|---|
| `layer_id` | Which height slab this row is (0 = ground-level slab, 1 = next slab up, etc.) |
| `z_low` | Bottom height (m) of this slab |
| `z_mid` | Middle height (m) of this slab — useful as the x-axis value when plotting the profile |
| `z_high` | Top height (m) of this slab |
| `PAI_layer` | Plant-Area-Index-like quantity for this slab = (number of occupied voxels in this slab) × (voxel height). A proxy for "how much plant material sits in this height band." Bigger = more foliage/wood detected at that height. |
| `n_occupied_voxels` | Raw count of occupied 3D voxels in this slab (before multiplying by voxel height to get PAI_layer) |

If you plot `PAI_layer` (y-axis) against `z_mid` (x-axis) as a bar chart, you get the "vertical canopy occupancy profile" — a picture of where in the canopy column the vegetation actually is. A single tall spike near the top = simple, single-layered canopy (low FHD). Several bars of similar height spread across many `z_mid` values = a multi-layered canopy (high FHD).

---

## Key concepts referenced above (quick definitions)

- **SOR (Statistical Outlier Removal)**: for every point, look at its nearest neighbors; if a point sits unusually far from everything else compared to the rest of the cloud, it's considered noise and removed.
- **CSF (Cloth Simulation Filter)**: the ground-classification algorithm — simulates a piece of cloth "falling" onto the (inverted) point cloud; wherever it settles becomes the estimated ground surface.
- **Voxelization**: chopping 3D space into small cubic cells (voxels, e.g. 0.5 × 0.5 × 0.5 m) and checking which ones contain at least one point ("occupied"), instead of working with millions of raw points directly.
- **TIN (Triangular Irregular Network)**: a way of interpolating elevation between ground points by connecting them into triangles (rather than a simple grid), used to build the DTM.

For the *full* explanations (including the "why" behind each concept, worked examples, and how CloudCompare's manual workflow maps onto this code), see `point_cloud_glossary.md` — this file is meant as a quick, output-focused companion to that one.

---

## Part 3 outputs — `03_individual_tree_detection.py` (individual tree detection)

Lands in `tree_detection/` (or whatever you set `output_dir` to).

| File | What it is |
|---|---|
| **`detected_trees.csv`** | One row per detected tree: `tree_id`, treetop coordinates (`x_top`, `y_top`) and height (`height_m`), its bounding box (`bbox_xmin/xmax/ymin/ymax` — the "square" corners), `crown_area_m2`, `crown_diameter_m` (equivalent circular diameter from the crown's pixel area), and `n_pixels` (how many CHM cells belong to that tree). |
| **`trees_bboxes.png`** | The CHM with a rectangle drawn around every detected tree — directly equivalent to the cyan/orange squares in your reference image. Orange = crown area below `low_confidence_crown_area_m2` (flagged as a small/uncertain detection worth checking by eye); cyan = normal-confidence detection. |
| **`trees_crown_outlines.png`** | A more accurate alternative view: instead of a rectangular box, this traces the *actual* segmented crown shape (irregular outline) for each tree, plus a white `+` marking each treetop. Bounding boxes are a simplification — this image shows what the algorithm really "thinks" each crown's shape is. |
| **`tree_labels.tif`** | A GeoTIFF raster where every cell's value is the `tree_id` it was assigned to (0 = not part of any tree — ground/understory/too short). Open this in QGIS to overlay tree boundaries on other data. |
| **`points_with_tree_id.las`** | Your original point cloud, height-normalized, with a new extra field `tree_id` on every point. Open in CloudCompare, switch the scalar field display to `tree_id`, and each tree will show up as a distinct color — lets you visually verify individual trees in full 3D, not just the flat CHM. |
| **`per_tree_las/tree_001.las`, `tree_002.las`, ...** | (Only if `save_per_tree_las: True`) — each detected tree's points saved as its own separate `.las` file, e.g. for computing per-tree volume/DBH/biomass individually. |

### Key parameters you can tune (all explained in-line at the top of the script)

| Parameter | Effect if you increase it |
|---|---|
| `smoothing_sigma_m` | Fewer, smoother detections (risk: merges real neighboring trees); lower = more, noisier detections (risk: splits one tree into several) |
| `min_height_m` | Raises the bar for what counts as a "tree" at all — raise it to ignore shrubs/grass/noise |
| `fixed_window_radius_m` / `vwf_a`, `vwf_b` | Controls minimum spacing between two treetops — bigger radius = fewer, more separated trees detected |
| `th_seed` | How far down (relatively) a crown is allowed to extend from its treetop — higher = smaller, tighter crowns |
| `max_crown_radius_m` | Hard cap on how wide any one tree's crown can grow — prevents one tall tree "eating" its neighbors |
| `point_assignment_mode` | `"columnar"` (fast, standard) vs `"conical"` (slower, more forgiving of TLS occlusion gaps) for assigning 3D points to a tree |

The script's own top-of-file comments include a full literature review (Popescu & Wynne 2004, Chen et al. 2006, Li et al. 2012, Silva et al. 2016, Dalponte & Coomes 2016, Ferraz et al. 2016, DeepForest/Weinstein et al. 2019) and a comparison of what software products (lidR, FUSION, ArcGIS Pro, LiDAR360, eCognition, TreeLS/QSM) use which method — read that first if you want the full "why," not just the "what."

---

## Part 4 outputs — `05_deepforest_itd.py` (AI-based detection, cross-validated against `03`)

Lands in `outputs/deepforest_itd/` (or whatever you set `output_dir` to),
with one subfolder per plot+sensor combination (e.g. `Plot43_TLS/`,
`Plot43_UAV/`), plus two files directly inside `output_dir` summarizing
across all of them.

### Per plot+sensor subfolder (e.g. `Plot43_TLS/`)

| File | What it is |
|---|---|
| **`{tag}_truecolor.png`** | The true-color raster built from the point cloud's own RGB values (reused from `03`'s true-color rendering) — this is the actual image fed into DeepForest. Only produced if the `.las` has real RGB color data; if not, this whole subfolder is skipped and a message explains why. |
| **`{tag}_truecolor.tif`** | The same image as a georeferenced GeoTIFF, for opening in GIS software alongside other layers. |
| **`{tag}_deepforest_trees.csv`** | One row per tree DeepForest detected: its bounding box (`bbox_xmin/xmax/ymin/ymax`), center coordinates (`x_center`, `y_center`), DeepForest's own confidence `score` (0-1), and `height_m` — looked up from the CHM at the box's center, since DeepForest itself has no height information at all (it only sees a flat image). |
| **`{tag}_comparison.png`** | The true-color image with BOTH detection sets overlaid: solid cyan boxes = DeepForest, dashed yellow boxes = `03`'s CHM-watershed detections for the same plot. Boxes that visually overlap closely are usually (but not automatically) the "matched" trees counted in the summary CSV below. |

(`{tag}` is `Plot{n}_{sensor}`, e.g. `Plot43_TLS`.)

### Files directly in `output_dir`

| File | What it is |
|---|---|
| **`deepforest_comparison_summary.csv`** | One row per plot+sensor combination, across the whole batch run — see column meanings below. |
| **`DeepForest_Comparison_Report.pdf`** | A short PDF: the same summary table, every plot's comparison image, and a written discussion of what "matched" vs. "single-method-only" detections likely mean. |

### Column-by-column meaning of `deepforest_comparison_summary.csv`

| Column | Meaning |
|---|---|
| `plot_number` | Which plot this row is for |
| `sensor` | `TLS` or `UAV` |
| `skipped_reason` | `no_rgb` if this plot/sensor was skipped (no real color data to build an image from), otherwise blank |
| `n_deepforest` | How many trees DeepForest detected (after the `deepforest_confidence_threshold` filter) |
| `n_watershed` | How many trees `03`'s CHM-watershed method detected for the same plot+sensor (`n/a` if `03`'s batch results weren't found — see `watershed_batch_dir` in `CONFIG`) |
| `n_matched` | How many trees BOTH methods detected — matched by IoU (intersection-over-union) exceeding `iou_match_threshold`. The highest-confidence subset: two independent signals (height-shape and visual appearance) agreeing. |
| `n_deepforest_only` | Trees DeepForest saw but the watershed method didn't — could be real trees the height-based method under-segmented/merged, or a DeepForest false positive. Worth a manual look, not an automatic error either way. |
| `n_watershed_only` | Trees the watershed method saw but DeepForest didn't — could be small/young trees not well represented in DeepForest's training data, or a genuine height-based detection with no clear visual signature in the reconstructed image. |

### What the numbers actually looked like on the real MyDiv data

Every TLS plot got **`n_deepforest = 0`** — DeepForest detected nothing at
all on any TLS-derived true-color image, across all 3 plots, completely
consistently. UAV plots got real detections (24-58 trees), with roughly
40-55% of the watershed method's own count matched. See `05`'s section in
`README.md` for the full discussion of why (short version: a TLS-derived
true-color image, despite being technically "top-down," looks visually
very different from the real aerial photos DeepForest was trained on —
radial density gradient, under-canopy texture — while UAV's version looks
much closer to genuine aerial imagery). If you see the same `n_deepforest
= 0` pattern on TLS in your own run, that's consistent with this finding,
not a bug.