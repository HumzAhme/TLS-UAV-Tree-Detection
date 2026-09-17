# TLS / UAV-LiDAR Point Cloud Analysis Toolkit

A set of Python scripts for processing terrestrial laser scanning (TLS) and
UAV-LiDAR point clouds of forest/tree plots: preprocessing, ground
classification, terrain/canopy models, foliage-density metrics, TLS-vs-UAV
comparison, and individual tree detection & crown delineation.

## What's in here

| File | What it does |
|---|---|
| `tls_pipeline.py` | Standalone: clean a single `.las` → DTM/DSM/CHM + ground mesh |
| `tls_core.py` | Shared library (ground classification, TIN/DTM, height normalization, CHM, voxelization, LAI parsing, true-color rendering) — never run directly, imported by `01`/`02`/`05` |
| `01_tls_foliage_density_pipeline.py` | Batch TLS processing → foliage-density metrics (FHD, RH, canopy volume) + LAI comparison |
| `02_uav_tls_chm_comparison.py` | Compares TLS-derived CHMs against UAV-LiDAR CHMs across plots |
| `03_individual_tree_detection.py` | **Individual tree detection & crown delineation** — the main, most feature-complete script (see below) |
| `03_individual_tree_detection_v1_no_rgb.py` | A lighter snapshot of `03` from before RGB/true-color support was added — same detection logic, no color rendering, no wizard |
| `04_generate_report.py` | Compiles `01`/`02`'s results into one PDF report (metrics tables, correlation plots, comparison figures, written answers) |
| `05_deepforest_itd.py` | **AI-based tree detection (DeepForest) cross-validated against `03`** — a second, independent detection method (see below) |
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

### Algorithms used, explicitly

- **Treetop detection** (finding candidate tree locations): a **local
  maximum filter** over the CHM, in one of three variants —
  `"fixed"` (constant search radius), `"variable"` (Popescu & Wynne 2004,
  radius scales with height), or `"h_maxima"` (prominence-based extended
  maxima transform — the default, and the one that actually solves
  merged/split-tree problems the other two can't).
- **Crown delineation** (turning a treetop into a crown shape): **marker-
  controlled watershed** segmentation of the CHM, seeded at each detected
  treetop, with Dalponte & Coomes (2016)-style constraints (`th_seed`,
  `max_crown_radius_m`) to stop one tree's flood from swallowing its
  neighbors.
- This is the same algorithmic family used by lidR (R), FUSION, ArcGIS
  Pro's "Segment Individual Trees," and LiDAR360 — a **height-only**
  method (it never looks at color/texture at all, only the CHM). `05`
  adds a second, independent, **image-based** method (DeepForest) for
  comparison — see below.

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

## 5. `05_deepforest_itd.py` — AI-based detection, cross-validated against `03`

A second, independent way to detect trees: DeepForest (Weinstein et al.
2019), a pretrained deep-learning object detector, finds tree crowns from
a true-color *image* using visual appearance (color/texture) — a
completely different signal than `03`'s height-based CHM+watershed
approach. Two independent methods agreeing on a tree is much stronger
evidence than either alone; where they *disagree* tells you exactly which
detections are worth a manual look. Full reasoning, honest caveats, and a
step-by-step breakdown are in the script's own header comment — read that
before trusting the numbers.

**Why this is a separate script from `03`, not a mode of it**: DeepForest
needs PyTorch + torchvision, a genuinely heavy dependency (1–5 GB)
completely unlike anything else in this toolkit. Keeping it separate means
you only pay that cost if you actually want to try it — see "Installing
DeepForest without permanently affecting your environment" below.

### Running it

```bash
pip install deepforest   # see the isolation note below first
python 05_deepforest_itd.py
```
Run `03` in batch mode first if you want the cross-validation (it reads
`03`'s `detected_trees.csv` files) — without them, `05` still runs and
produces DeepForest-only results.

### What it actually does

For each plot × sensor (same `Plot{n}_MyDiv.las` / `MyDiv_{n}_L1.las`
convention as `02`/`03`): checks for real RGB color data (skips cleanly if
none — DeepForest needs an actual image, there's nothing to build one
from height-only data), builds a true-color raster (reusing `03`'s
rendering, now shared via `tls_core.py`), runs DeepForest on it, looks up
each detected tree's height from the CHM (DeepForest itself has no height
information), then matches its boxes against `03`'s by IoU
(intersection-over-union) and reports matched / DeepForest-only /
watershed-only counts. Ends with a summary CSV and a PDF report across
all plots.

### Real result from running this on the MyDiv plots — worth knowing before you run it

| Plot | Sensor | DeepForest trees | `03` (watershed) trees | Matched |
|---|---|---|---|---|
| 43 | TLS | **0** | 178 | 0 |
| 43 | UAV | 42 | 58 | 28 |
| 58 | TLS | **0** | 147 | 0 |
| 58 | UAV | 58 | 63 | 31 |
| 70 | TLS | **0** | 136 | 0 |
| 70 | UAV | 24 | 60 | 14 |

A completely consistent pattern: **DeepForest detected zero trees on
every single TLS true-color image, but real, substantial detections on
every UAV one.** This is exactly the kind of thing this script exists to
surface, and it did. The likely reason: a TLS-derived true-color raster,
even though it's technically also a "top-down view," looks visually very
different from a real aerial photo — built from a single ground-based
scan position, it has that characteristic radial density gradient (dense
near the scanner, sparse/gappy toward the edges) and shows a lot of
under-canopy/side texture rather than a clean leafy canopy top. UAV's
true-color raster, built from an actually-airborne point cloud, looks
much more like what DeepForest was trained on (real NEON aerial RGB
photos) — hence it works there, however imperfectly (roughly half of each
method's detections go unmatched even on UAV).

**Practical takeaway**: on this data, DeepForest is only useful for the
UAV side, and even there it should be read as a second opinion, not a
replacement for `03` — the two methods agree on roughly half of what
either finds, so treat "matched" detections as the highest-confidence
subset, and "single-method-only" detections as worth a manual look rather
than as errors.

### Installing DeepForest without permanently affecting your environment

Use a disposable conda environment rather than installing into whatever
you normally use for `01`–`04`:
```bash
conda create --clone your_env_name -n env_deepforest_test
conda activate env_deepforest_test
pip install deepforest
python 05_deepforest_itd.py
conda deactivate
conda env remove -n env_deepforest_test
conda clean --all -y   # optional, reclaims the shared package cache
```
Cloning (rather than a fresh env) means `05` can run immediately with
everything else already installed. Your original environment is never
touched by any of this. See the script's header comment for what to do if
you hit a `predict_image()` argument error — DeepForest's API has changed
across versions, and `05` auto-detects which style your installed version
needs, but it's worth knowing this is a live area of the package.

### What more is possible here

- **Investigate the TLS zero-detection result further**: check DeepForest's
  raw (pre-confidence-filter) output on a TLS image — is it truly finding
  *nothing* (no region proposals at all), or finding weak candidates that
  all fall under the confidence threshold? `deepforest_confidence_threshold`
  in `CONFIG` controls this — try lowering it toward 0 on a TLS image
  specifically to see if there's any signal at all before concluding
  there's none.
- **Try a hillshade-rendered TLS image instead of true-color**: since TLS's
  true-color raster is the likely problem (not TLS data itself), a
  hillshade render (already used as `03`'s CHM background) has a very
  different, more uniformly-textured appearance that might transfer better
  than the radially-gapped true-color version — worth an experiment.
  Requires a small code change to pass a hillshade image to DeepForest
  instead of a true-color one for TLS specifically.
- **Fine-tune DeepForest on your own imagery**: use `03`'s watershed
  detections as approximate/noisy training labels to fine-tune DeepForest
  specifically on TLS-style rasters (a form of semi-supervised
  bootstrapping) — a real project, not a quick change, but the most
  likely way to actually fix the TLS zero-detection result rather than
  just working around it.
- **Ensemble the two methods properly**: rather than just reporting
  matched/unmatched counts, build a combined tree list where matched
  detections get high confidence, single-method detections get flagged
  for review, and (if you get field counts) calibrate which method is
  more reliable in which situation.
- **Tune `iou_match_threshold`**: currently 0.3; since only ~50-65% of
  either method's UAV detections match the other, check whether that's
  genuine disagreement or just strict box-overlap requirements — try
  0.15-0.2 and see how much the matched count changes.
- **Try `predict_tile()` instead of `predict_image()`**: DeepForest's docs
  note whole-image prediction can perform worse on dense/large images than
  its tiled prediction mode (`patch_size`/`patch_overlap`) — untested here,
  worth comparing on the UAV images specifically.

### Other AI models worth trying instead of / alongside DeepForest

DeepForest is one specific, somewhat dated (2019) choice — not the only
option. Current (2025-2026) research points at several genuinely stronger
or differently-shaped alternatives, roughly ordered by how much effort
each would take to try:

- **Detectree2** ([PatBall1/detectree2](https://github.com/PatBall1/detectree2),
  Ball et al. 2023) — the closest drop-in replacement for DeepForest: also
  a pretrained image-based detector with a similar "give it an image, get
  boxes/masks back" API (Mask R-CNN via Detectron2, not RetinaNet), but
  pretrained on more diverse forest data *including tropical sites*,
  rather than DeepForest's US-only NEON training. Recent literature
  (SelvaMask, 2026) explicitly names it "the current state-of-the-art for
  generalized tropical tree crown segmentation" — worth trying first
  simply because it might generalize to non-US forest types (like MyDiv)
  better than DeepForest does. A high-level wrapper exists via the
  `segment-geospatial` package (`samgeo.detectree2`) that simplifies the
  API further. Install is heavier than DeepForest (needs Detectron2 built
  from source), but no architectural changes to `05` would be needed —
  it's a similar "run inference, get pixel boxes back" call.

- **SAM2 / SAM3** (Segment Anything, Meta) — rather than replacing
  DeepForest, use `03`'s own CHM-watershed detections as *box prompts* fed
  into SAM2, which then returns a precise pixel-level crown *mask* instead
  of just a rectangular box — the current state-of-the-art for turning an
  approximate detection into an accurate outline. This is a genuinely
  different, complementary role than DeepForest plays (mask refinement,
  not independent detection), and 2026 research (SelvaMask) found this
  kind of "modular" detector→SAM pipeline can match or beat fully
  fine-tuned end-to-end models *without* any fine-tuning at all. Available
  through Meta's own repo and mirrored on HuggingFace.

- **Mask2Former** ([HuggingFace-native](https://huggingface.co/docs/transformers/model_doc/mask2former),
  `transformers` library) — a heavier, more modern transformer-based
  instance segmentation architecture. 2026 tropical-forest benchmarks
  (SelvaMask) found a Mask2Former–Swin-L backbone achieving 19.7 mAP vs.
  11.1 mAP for a comparable Mask R-CNN — a real, substantial accuracy gap
  in recent published comparisons. No tree-specific pretrained Mask2Former
  checkpoint is publicly available as far as I could confirm, though — you'd
  likely need to fine-tune a general-purpose (COCO-pretrained) checkpoint
  on your own or a public tree-crown dataset first, making this the
  highest-effort option of the three.

- **Feed it height data, not just color** (RSPrompter/DSMPrompter/BalSAM,
  2025-2026 research): several recent papers found that adding DSM
  (elevation) information alongside RGB — either as a 4th input channel or
  as an elevation-based prompt — measurably improves SAM-based crown
  segmentation over RGB alone. This is a particularly good fit for this
  toolkit specifically, since `03` already computes a CHM for every plot —
  the elevation data these papers need is something you already have,
  not something you'd need to newly acquire.

**One more thing worth knowing**: 2026 benchmark research (SelvaMask)
found that off-the-shelf pretrained models — including *both* DeepForest
*and* Detectree2 — score in the *single digits* (mAP) when tested
out-of-domain (their test: tropical forest canopy neither model was
trained on). That's a published, independent confirmation of exactly the
pattern this toolkit found empirically: DeepForest detected zero trees on
every TLS-derived image here. Domain mismatch between a pretrained
model's training data and your actual forest type is a real, common,
and *expected* failure mode in this field — not a bug specific to this
setup.



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

**LiDAR-native AI detection** (answering "can we make our own raster + DeepForest work?")
- **If your `.las` has real RGB**: `render_photorealistic_topview()` already
  builds a true-color raster (used for `trees_bboxes2.png`) -- this could
  be fed into DeepForest's *pretrained* weights directly, since it's real
  photographic color/texture, the same input type DeepForest was trained
  on. The most direct path to a working LiDAR-based DeepForest pipeline.
- **If height-only (no real RGB)**: stack height + intensity + hillshade
  into a synthetic 3-channel raster and *fine-tune* DeepForest (or train a
  small model from scratch) on that, rather than using the pretrained RGB
  weights as-is -- the pretrained model's learned features are
  color/texture-specific and don't reliably transfer to a height-based
  image even after colorizing it. Needs labeled training data (known tree
  positions) to fine-tune against -- a real project, not a quick add-on.

**Additional derived forestry metrics**
- **DBH (diameter at breast height)**: once stem/trunk detection exists
  (see "Detection accuracy" above), fit circles to a horizontal point
  cloud slice at ~1.3 m per detected stem to get an actual trunk diameter
  -- a standard forestry metric this pipeline doesn't currently produce.
- **Biomass / carbon estimation**: combine detected height + DBH (or
  crown metrics alone) with published allometric equations to estimate
  per-tree biomass/carbon storage, then aggregate to plot level --
  directly relevant to MyDiv's ecological research questions.

**Data quality**
- **Multi-scan-position TLS registration**: if multiple scan positions per
  plot are available (rather than the single-scan data used here),
  registering and merging them before processing would substantially
  reduce the occlusion gaps visible in the current single-scan CHMs and
  crown detections (the "shadow wedge" pattern discussed throughout this
  project).
- **Use intensity / multi-return fields**: detection currently uses height
  (CHM) only; `intensity` and `return_number`/`number_of_returns` (present
  in most LAS files but unused here) could help distinguish live foliage
  from bare branches, or improve ground/vegetation separation beyond CSF
  alone.

**Performance**
- **Separate detection from visualization**: currently, changing a purely
  cosmetic parameter (`bbox_color_by`, `topview2_dpi`, `chm_background_cmap`)
  still re-runs the entire pipeline from raw points. Caching the
  CHM/watershed/tree-table results and only re-running the plotting step
  when only visualization parameters change would make style iteration
  much faster.
- **Parameter sensitivity / uncertainty analysis**: systematically sweep
  `h_maxima_prominence_m`, `max_crown_radius_m`, and `th_seed` across a
  range and report how stable the detected tree count/positions are --
  turns manual trial-and-error tuning into a data-driven confidence
  measure.

**Cross-script integration**
- Join `detected_trees.csv` (individual tree count/size from `03`) with
  `tls_metrics.csv` (plot-level FHD/LAI from `01`) to check whether
  individual tree density/size explains plot-level foliage-density
  patterns -- currently these two analyses don't talk to each other.

**Multi-temporal**
- If TLS scans of the same plots exist at multiple dates, track individual
  trees across scans (matching by nearest position + similar height) to
  estimate growth rates per tree, not just a single-date snapshot.

---

## Key terms (tree detection specific)

The full glossary (`point_cloud_glossary.md`) covers point-cloud and
CHM/DTM terminology in depth. These are the additional terms specific to
individual tree detection (`03_individual_tree_detection.py`):

- **ITD (Individual Tree Detection)** — the general task this whole script
  performs: locating each individual tree in a plot (as opposed to just
  describing the canopy in aggregate, like a CHM does). "ITC" (Individual
  Tree Crown) delineation is the closely related task of also outlining
  each tree's crown shape, not just its location — this script does both.

- **CSF algorithm (Cloth Simulation Filter)** — the ground-classification
  method (Zhang et al. 2016) used throughout this toolkit (`tls_core.py`,
  and every script that needs a DTM). Simulates a piece of cloth falling
  onto an inverted point cloud; wherever it settles approximates the
  ground surface. Full explanation with parameter-by-parameter breakdown
  in `point_cloud_glossary.md`.

- **Local maximum filter** — the simplest treetop-finding method: a pixel
  in the CHM counts as a treetop if it's taller than every other pixel
  within some search radius around it. This script supports a `"fixed"`
  radius, a height-scaled `"variable"` radius (Popescu & Wynne 2004), and
  the more robust `"h_maxima"` prominence-based method (the current
  default) — see `peak_detection_method` in `CONFIG`.

- **Watershed algorithm** — the crown-delineation method used after
  treetops are found: floods outward from every treetop simultaneously
  (like water filling a landscape from multiple sources), and wherever two
  floods meet becomes the boundary between two crowns. Implemented in
  `segment_crowns()`, constrained by `th_seed` and `max_crown_radius_m`
  (Dalponte & Coomes 2016 style).

- **Foliage-density metrics** — quantities describing how much/how
  vertically-structured a canopy's foliage is (FHD, PAI, canopy volume,
  RH percentiles), computed in `01_tls_foliage_density_pipeline.py`. Full
  definitions in `point_cloud_glossary.md` Part 2.

- **DeepForest (AI-based tree detection)** — a deep-learning alternative
  (RetinaNet object detector, Weinstein et al. 2019) that finds tree
  crowns directly from an RGB image, the way an aerial photo interpreter
  would, rather than from a height model. **Not implemented in this
  script** — it needs an RGB drone orthomosaic, which a bare TLS/UAV point
  cloud doesn't provide (see the "possible improvements" discussion
  below). Full comparison against CHM+watershed methods in the script's
  own header comments.

- **LiDAR-to-raster (rasterization)** — converting the scattered 3D point
  cloud into a regular 2D grid of height values (the CHM) so image-based
  methods (local-maximum search, watershed) can be applied at all. Done in
  `get_chm()` / `tls_core.rasterize_canopy_p2r()`.

- **Resolution** — the size of each raster grid cell, in meters
  (`chm_res` for detection, `topview2_res` for the true-color visuals,
  independently configurable). Finer resolution preserves more real
  detail but is slower and noisier; coarser resolution is faster and
  smoother but can merge close trees together. See the CONFIG comments
  and the wizard's explanations for the exact trade-offs.
