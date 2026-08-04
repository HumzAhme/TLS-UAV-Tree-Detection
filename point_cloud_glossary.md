# TLS Point Cloud Processing — Concepts & Glossary

This file explains every term from the CloudCompare (CC) handout, and shows
exactly which line of `tls_pipeline.py` reproduces each manual CC step.
Read it top to bottom once — after that, use it as a reference.

---

## 1. The basics

### What is a 3D point cloud?
A point cloud is a set of points in 3D space, each with at least `X, Y, Z`
coordinates. It's the raw output of a laser scanner (LiDAR) or photogrammetry:
millions of individual "hits" where a laser pulse bounced off a surface
(leaf, branch, soil, rock...) and came back to the sensor. Each point can
also carry extra attributes: intensity (how strong the return was),
return number, RGB color, GPS time, etc. It has no surfaces or connectivity
by default — just a "cloud" of dots.

### What is TLS?
**TLS = Terrestrial Laser Scanning.** A tripod-mounted laser scanner is placed
on the ground and spins around, firing thousands of laser pulses per second
in all directions, building a very dense, high-resolution point cloud of
everything around it (ground, trunks, branches, leaves). This is different
from **ALS** (Airborne Laser Scanning) or **L1** data, which is collected
from a plane/drone and looks "down" on the canopy rather than "into" it —
that's why TLS is so good at capturing understory and stem structure, but
has a limited range/footprint per scan position.

### Why preprocess a 3D point cloud?
Raw point clouds are messy:
- **Noise/outliers**: stray points from reflections, moving objects (leaves
  in wind), multi-path laser returns, or sensor error.
- **Redundancy**: overlapping scans, uneven point density.
- **No semantic labels**: the scanner doesn't know which point is "ground"
  vs. "leaf" vs. "trunk" — you have to derive that.
- **No direct use for analysis**: you can't compute canopy height, biomass,
  or terrain slope from a raw dot cloud; you need cleaned, structured,
  and classified data first.

Preprocessing (cleaning, classifying, rasterizing) turns the raw cloud into
something you can actually measure and compare across plots.

### Why is there no geographic coordinate system in CC? (the "global shift")
LiDAR/TLS coordinates are often huge numbers (e.g. UTM easting/northing in
the millions, like `X=412345.678`). Storing/computing with such large
numbers in single-precision floats (which most 3D software, including
CloudCompare's renderer, uses internally) causes rounding errors ("jittery"
points, visual glitches). To avoid this, CloudCompare detects large
coordinates and asks to apply a **global shift**: it subtracts a large
constant offset from every point so the numbers become small and easy to
work with, and stores that offset internally so the true global position
isn't lost. In practice, many TLS scans (especially single, un-georeferenced
plots like `Plot_48_MyDiv.las`) were never registered to a real-world
coordinate system (like UTM) in the first place — the scanner just uses its
own **local/relative coordinate system** centered on the tripod. That's why
you see "no geographic coordinates": the plot exists in its own local XYZ
space, not tied to a map projection. **In the Python pipeline this doesn't
matter** — all processing (SOR, CSF, rasterization) works fine in local
coordinates. The output GeoTIFFs are written with `crs=None` for the same
reason; if you ever do have real UTM/geographic coordinates, you'd set the
correct `crs=` in `rasterio.open(...)`.

### What is "saturation" and why filter it?
A laser return can be saturated when the reflected signal is too strong
(e.g. very close/reflective surfaces) or too weak/noisy, producing points
with unreliable intensity values or slightly wrong positions. These show up
as scattered "fliers" — points floating away from any real surface, or
clusters of oddly-positioned points. SOR filtering (below) is the main tool
used to remove these.

---

## 2. Cleaning: SOR filtering

**SOR = Statistical Outlier Removal.**
For every point, look at its `k` nearest neighbours and compute the average
distance to them. Compute the mean and standard deviation of *all* these
average distances across the whole cloud. Any point whose average neighbour
distance is larger than `global_mean + std_ratio × global_std` is considered
isolated/noisy (an outlier) and removed.

- Points **inside** a surface (leaf, ground, trunk) have close neighbours →
  small average distance → kept.
- Stray/noisy points sit **away** from any real surface → large average
  distance → removed.

CC location: `Tools > Clean > SOR filter` (parameters: *Number of
neighbors* and *Std ratio*).
Python: `sor_filter()` in `tls_pipeline.py`, using a KD-tree
(`scipy.spatial.cKDTree`) for fast nearest-neighbour search.

---

## 3. Scalar fields

### What is a scalar field, and why use it?
A scalar field is just a single number attached to every point (as opposed
to the point's XYZ position or RGB color) — e.g. intensity, return number,
height above ground, or a classification code (0, 1, 2 ...). CC's
"Properties" panel lets you switch the *display* between RGB colors and any
scalar field, which lets you color the cloud by that value (e.g. color by
height to see terrain relief, or color by classification to see ground vs.
vegetation). You "choose" a scalar field depending on what you want to
inspect or filter: intensity to spot saturated points, return number to
separate first/last returns, or classification to separate ground/vegetation.

In our Python pipeline, the equivalent of a "scalar field" is simply a
NumPy array aligned with the points (e.g. `mean_dists` in the SOR step, or
the `classification` byte we write into the LAS file: `2` = ground,
`1` = non-ground/vegetation, following the standard ASPRS LAS classification
codes).

### Point classification
"Classification" is a scalar field where each point gets an integer code
describing what kind of object it belongs to (ground, low vegetation, high
vegetation, building, noise...). Airborne LiDAR datasets are often already
classified by the data provider. **TLS data usually is NOT classified** —
the handout notes this explicitly — so we have to compute it ourselves,
which is exactly what the CSF step below does (we then also *save* this
result into the LAS `classification` field, so future users/software can
read it directly).

### "Filter by values / splitting" (Edit → Scalar Fields → Filter by values)
Once a scalar field exists (e.g. classification = 0 or 1), CC lets you set a
value range with two sliders and "split" the cloud into two new clouds: one
inside the range, one outside. In Python this is trivial — it's exactly what
a **boolean mask** does:
```python
ground_mask = classification == 2          # or: dists <= threshold, etc.
ground_points   = points[ground_mask]
non_ground_pts  = points[~ground_mask]
```
This is used twice in the pipeline: once implicitly inside SOR filtering
(keep vs. remove), and once explicitly after CSF (ground vs. non-ground).

---

## 4. Ground classification: the CSF filter

### What is ground classification?
Deciding, point by point, whether a laser return came from bare
earth/soil ("ground") or from something sitting on top of it (grass, shrubs,
tree canopy, deadwood, rocks...). It's the necessary first step before you
can compute terrain models or canopy height, because "canopy height" is
literally defined as *height above the ground*.

### What is the CSF filter?
**CSF = Cloth Simulation Filter** (Zhang et al., 2016). It's a clever
physics-based trick, popular because it needs very few parameters:
1. Flip the point cloud upside down.
2. Imagine dropping a virtual piece of cloth (a grid of connected particles)
   onto the *inverted* surface, like a bedsheet falling onto a lumpy floor.
3. Gravity pulls the cloth down; where it touches points, it stops (collides).
   Because the cloth particles are connected (a spring-like grid), the cloth
   can't just fall through gaps left by removed "canopy" points — it
   bridges over them and settles onto the true ground shape.
4. Un-flip everything. The final shape of the cloth is your terrain surface.
   Points close enough to the settled cloth = **ground**; everything
   floating clearly above it = **non-ground** (vegetation, etc.).

### CSF parameters used in the handout (and in the code)
- **Scenes: Relief** — a convenience preset in CC that just sets the
  underlying `rigidness` (cloth stiffness) parameter for you, depending on
  how "bumpy" your terrain is:
  - *Steep slope* → soft cloth (`rigidness = 1`) — needed so the cloth can
    still drape down steep terrain instead of "bridging" over dips.
  - *Relief* → medium (`rigidness = 2`) — good default for natural,
    moderately uneven ground (forest plots, our MyDiv case).
  - *Flat* → stiff cloth (`rigidness = 3`) — best for flat, open ground.
- **Cloth res(olution): 1** — the grid spacing (in meters) between
  neighbouring cloth particles. Smaller = the cloth can follow finer terrain
  detail, but is slower and more easily "caught" by low vegetation (mistaken
  for ground bumps).
- **Max Iterations: 600** — how many simulation steps the cloth is allowed
  to fall/settle before we stop. More iterations = cloth has more time to
  relax fully onto the terrain; too few and the cloth may still be "hovering."
- **Classification threshold: 0.5** — after the cloth settles, any point
  within this distance (meters) *below* the cloth surface is labeled ground;
  anything farther above is non-ground. Larger threshold = more points
  (including short vegetation/rocks) get (mis)classified as ground.

Python: `csf_ground_classification()` uses the `CSF` package (the exact same
open-source C++ algorithm CloudCompare's plugin wraps, exposed to Python),
with parameters named identically to CC's dialog (`cloth_resolution`,
`rigidness`, `interations` [sic — that's the actual name in the library],
`class_threshold`).

---

## 5. Terrain & canopy models

### DEM — Digital Elevation Model
A generic umbrella term for *any* raster (grid) of elevation values. DTM and
DSM (below) are both specific kinds of DEM.

### DTM — Digital Terrain Model (a.k.a. "bare-earth" model)
A raster where each grid cell holds the elevation of the **ground surface
only** — no vegetation, no buildings. Built from the ground-classified
points. This is what the handout calls "DTM: Rasterize the ground points
from the classification."

### DSM — Digital Surface Model
A raster where each cell holds the elevation of the **highest thing present**
at that location — treetop, roof, rock, or bare ground if nothing is above
it. Built from *all* points (not just ground), taking the maximum height per
cell.

### CHM — Canopy Height Model
```
CHM = DSM − DTM
```
This subtraction cancels out the underlying terrain relief and leaves just
the **height of vegetation (or objects) above the ground** at every grid
cell — exactly what you want for measuring tree/canopy height, independent
of whether the plot itself is sloped.

### Rasterization
The general process of converting scattered point data into a regular grid
(raster) of cells, each holding one aggregated value (e.g. max height for
DSM, mean/interpolated height for DTM). CC's "Rasterize" tool does this
directly; our `rasterize_points()` function reproduces it:
1. Compute a grid at the requested **resolution** (e.g. 0.05 m = 5 cm cells,
   as specified in the handout).
2. Assign each point to the cell it falls into, and aggregate (max or mean).
3. **Interpolate** any empty cells (no points fell there) using neighbouring
   cell values — this is the "interpolate the points" step in the handout,
   done here with `scipy.interpolate.griddata`.

### Ground mesh ("off-ground_points.mesh")
A **mesh** connects points into triangles to form a continuous surface
(instead of disconnected dots), which is much easier to visually inspect —
you can see if the ground surface looks smooth and realistic, or has holes/
spikes indicating a bad classification. We reproduce this with a **Delaunay
triangulation** of the ground points in the XY plane (`save_ground_mesh()`),
exporting a `.obj` mesh file (you can open this in Blender, MeshLab, or
CloudCompare itself) plus a shaded PNG preview.

---

## 6. Mapping: CC handout step → Python function

| CC step (handout) | Python function / output |
|---|---|
| 1.1–1.2 Open .las, accept global shift | `load_las()` — no shift needed, local coords used directly |
| 1.5–1.6 Clone cloud | working copy kept as `xyz_clean` in memory |
| 1.7 SOR filtering | `sor_filter()` → `01_sor_filtered.las` |
| 1.8 RGB → Scalar fields, compare point counts | printed point counts at each step |
| 1.9 Return number / intensity scalar fields | available as `las.intensity`, `las.return_number` if present in your file |
| 2.2 CSF ground classification | `csf_ground_classification()` → `02_ground_points.las`, `03_nonground_vegetation_points.las` |
| 2.2 DTM rasterize + interpolate | `rasterize_points(..., agg="mean")` → `DTM.tif` |
| (DSM, not explicit in handout but needed for CHM) | `rasterize_points(..., agg="max")` → `DSM.tif` |
| CHM (implied, needed for "canopy height model") | `compute_chm()` → `CHM.tif` |
| Mesh + cloud visualization of ground | `save_ground_mesh()` → `ground_mesh.obj` + PNG preview |
| Save DTM as .tif | `save_geotiff()` |
| Filter by values / splitting scalar field | boolean masking (`ground_mask` / `~ground_mask`) |

---

## 7. Output files you'll get from `tls_pipeline.py`

| File | What it is |
|---|---|
| `01_sor_filtered.las` | Cloud after removing SOR outliers |
| `02_ground_points.las` | Ground-only points (classification = 2) |
| `03_nonground_vegetation_points.las` | Vegetation/other points (classification = 1) |
| `DTM.tif` | Digital Terrain Model raster |
| `DSM.tif` | Digital Surface Model raster |
| `CHM.tif` | Canopy Height Model raster (DSM − DTM) |
| `DTM_preview.png` / `DSM_preview.png` / `CHM_preview.png` | Quick-look images |
| `ground_mesh.obj` | Triangulated ground surface mesh |
| `ground_mesh_preview.png` | Shaded 3D preview of the ground mesh |

All `.tif` files can be opened in QGIS, ArcGIS, CloudCompare (as a raster),
or Python (`rasterio`/`matplotlib`) for further analysis (slope, hillshade,
zonal stats, etc.).

---

# Part 2 — Foliage density, LAI, and TLS vs. UAV-LiDAR

This section covers the extra concepts used in `01_tls_foliage_density_pipeline.py`
(single-plot + batch foliage-density metrics, joined with field LAI) and
`02_uav_tls_chm_comparison.py` (comparing TLS-derived CHMs against
UAV-LiDAR-derived CHMs across the 12 MyDiv plots).

## CRS / EPSG code
**CRS = Coordinate Reference System.** A specification of exactly how X/Y/Z
numbers map onto real positions on Earth (a map projection + a datum). MyDiv
plots near Leipzig commonly use **EPSG:25833** (ETRS89 / UTM zone 33N) — an
"EPSG code" is just a short numeric ID for a specific, standardized CRS.
Assigning a CRS to your point cloud (`assign_crs()` in the code) doesn't
move any points — it just labels the numbers so GIS software knows how to
interpret and align them with other georeferenced data (e.g. the UAV-LiDAR
data in script 02). Contrast this with the *local, un-georeferenced*
coordinate system discussed in Part 1 (the CC "global shift" case) — some
TLS exports have a real CRS, some don't; check your file's metadata.

## TIN — Triangular Irregular Network
A way of representing a surface (like the ground) as a mesh of connected,
non-overlapping triangles built directly from the ground points themselves
(a Delaunay triangulation), rather than a regular grid. To get an elevation
at any arbitrary (x, y), you find which triangle contains that point and
linearly interpolate between its three corner elevations. This is what
`rasterize_terrain(algorithm = tin())` does in `lidR`, and what
`build_tin_interpolator()` / `rasterize_terrain_tin()` reproduce in Python.
It tends to follow terrain shape more faithfully than simple grid-binning,
especially with irregularly-spaced ground points.

## Height normalization
The process of converting every point's Z value from **absolute elevation**
to **height above the local ground surface**, by subtracting the
interpolated DTM/TIN elevation at that point's (x, y) from its Z:
```
Z_normalized(x, y) = Z_raw(x, y) − DTM(x, y)
```
After normalization, a value of `Z_normalized = 0` always means "on the
ground," regardless of whether that spot is on a hill or in a dip — this is
essential before computing any height-based canopy metric (CHM, foliage
profiles, RH percentiles), because otherwise terrain slope would be
confused with vegetation height.

## Point-to-raster (p2r)
The simplest, fastest CHM-generation method: for every output grid cell,
just take the **maximum height** of any (height-normalized) point that
falls inside it — no smoothing, no interpolation model, just "what's the
tallest thing directly here?" Cells with no points at all are left empty
(`NA`/`NaN`) unless you explicitly fill the gaps (see below). This is what
`rasterize_canopy(algorithm = p2r())` does in `lidR`, reproduced by
`rasterize_canopy_p2r()` here. `subcircle` (an optional p2r parameter in
`lidR`, not required for our Python version) artificially "inflates" each
point into a small disk before rasterizing, to reduce empty cells caused by
low point density.

## Filling CHM gaps
Small grid cells with no laser returns (common at fine resolutions or in
sparser point clouds) show up as holes in the CHM. "Filling gaps" replaces
each empty cell with the value of its nearest non-empty neighbour (or an
interpolated value), producing a continuous-looking raster suitable for
visual comparison and summary statistics. (R: `fill_chm_gaps()`).

## Voxel / voxelization
A **voxel** ("volumetric pixel") is the 3D equivalent of a 2D pixel: a
small cubic cell of space (e.g. 0.5 × 0.5 × 0.5 m). **Voxelization** bins
every point into the voxel it falls inside, then works with **occupied
voxels** (voxels containing ≥1 point) rather than the raw point cloud. This
converts millions of individual points into a manageable, regular 3D
occupancy grid — the foundation for all the foliage-density metrics below.

## PAI / PAD — Plant Area Index / Plant Area Density
**PAD** is the amount of plant surface area (leaves + woody material) per
unit volume, at a given height. **PAI** is PAD integrated (summed) over a
height layer or the whole canopy column — conceptually "how much plant
surface area sits above one square meter of ground." Our voxel-occupancy
count is used as a simple proxy for PAD/PAI: more occupied voxels at a given
height = more laser hits = (approximately) more plant material there. Note
this measures **plant** area (leaves *and* wood), which is why it is not
identical to LAI (leaf area only, see below).

## FHD — Foliage Height Diversity
A single number (Shannon entropy) summarizing how **evenly** foliage/plant
material is spread across vertical height layers (MacArthur & MacArthur,
1961). High FHD = many layers each contain a meaningful share of the total
foliage (a structurally complex, multi-layered canopy). Low FHD = nearly
all the foliage sits in one narrow height band (a simple, single-layered
canopy) — **even if the total amount of foliage (LAI) is the same** in both
cases. FHD is about the *shape* of the vertical profile, not the *total
amount*.

## RH metrics (RHxx) — Relative Height percentiles
`RH98`, `RH75`, `RH50`, `RH25`, etc. are heights below which a given
percentage of the *cumulative vertical plant-area profile* is found
(bottom-to-top), analogous to percentiles of a distribution. `RH98` (98th
percentile height) is a common, noise-robust proxy for "canopy top height"
— more robust than the raw maximum height, which can be thrown off by a
single stray high point.

## Canopy volume / occupancy index
- **Canopy volume (m³)** = number of occupied voxels × voxel volume — a
  simple estimate of how much 3D space is actually filled with
  vegetation/structure.
- **Canopy volume per ground area** = canopy volume ÷ plot footprint area —
  normalizes for plot size so different plots can be compared directly;
  often the single TLS metric most strongly related to field LAI, because
  both describe "how much plant material is packed into the canopy," just
  measured in 3D vs. as a 2D projection.
- **Occupancy index** = occupied voxels ÷ total possible voxels in a fixed
  reference volume (e.g. the whole plot footprint × a fixed max height) —
  a density index that is comparable across plots even if their absolute
  canopy volumes differ.

## LAI — Leaf Area Index
The one-sided leaf area per unit ground area (m² leaf / m² ground) — a
classic, widely used measure of canopy density in ecology and remote
sensing. Unlike PAI (above), LAI specifically excludes woody material
(branches, trunks). In this exercise, LAI is measured in the field with a
**LI-COR LAI-2200(C)** plant canopy analyzer, and then compared against the
TLS/point-cloud-derived structural metrics.

## LI-COR LAI-2200(C)
A handheld optical instrument with a fisheye sensor that measures the
diffuse light transmitted through the canopy at 5 different view angles,
then inverts a physical light-transmission model (based on Miller's
theorem/gap-fraction inversion) to estimate LAI, without needing to
directly measure any leaves. Its raw `.TXT` output file is a compact table
of instrument readings; the key line for our purposes is simply:
```
LAI    5.054
```
`read_licor_lai()` in `tls_core.py` parses this file format directly
(tokenizing on whitespace and grabbing the value right after the exact
token `LAI`, which correctly skips the unrelated `LAI_FILE` field at the top
of the file).

## Why TLS and UAV-LiDAR CHMs of the *same* plot differ
Even from the same forest patch, ground-based TLS and airborne UAV-LiDAR
CHMs will not match perfectly, mainly because of:
- **Viewing geometry**: TLS looks up/outward from the ground; UAV-LiDAR
  looks down from above. Each sees a different "shell" of the canopy well
  and struggles with the opposite one.
- **Occlusion** (see Part 1 "saturation" section for the related idea of
  unreliable returns): TLS returns are blocked by anything between the
  scanner and a target point, producing systematic **shadow wedges** of
  missing/reduced data radiating from the scan position(s) — visible
  directly in the `Plot{n}_TLSminusUAV.tif` difference rasters produced by
  `02_uav_tls_chm_comparison.py`.
- **Resolution & point density**: different sensors, different flight/scan
  patterns, so different native resolutions (this exercise uses 0.05 m for
  TLS, 0.15 m for UAV) and different spatial point-density patterns.
- **Georeferencing**: any small misalignment between how each dataset was
  registered to real-world (or local) coordinates shows up as apparent
  height differences, especially at canopy edges.
- **Independent ground/DTM models**: each sensor's own ground classification
  and terrain interpolation slightly disagree, and that disagreement
  propagates directly into the height-normalized CHM.

## Resampling / reprojecting a raster onto another grid
Since the TLS CHM (0.05 m cells) and UAV CHM (0.15 m cells) don't share the
same grid, you can't subtract them directly cell-by-cell. **Resampling**
(here, bilinear resampling via `resample_grid_bilinear()`, mirroring
`terra::project(..., method="bilinear")` in R) re-evaluates the finer TLS
raster onto the exact grid of the coarser UAV raster, so every cell lines
up 1:1 and a difference raster (`TLS − UAV`) can be computed.

---

## File map for Part 2

| Script | Purpose |
|---|---|
| `tls_core.py` | Shared functions used by both pipelines below (ground classification, TIN/DTM, height normalization, CHM, voxelization, FHD/RH/structure metrics, LI-COR parser, grid resampling) |
| `01_tls_foliage_density_pipeline.py` | Per-plot + batch TLS processing → DTM, CHM, voxel-based foliage metrics, LAI join, correlation plots, **and detailed written answers to the 5 assignment questions**, saved to `TLS_assignment_answers.md` |
| `02_uav_tls_chm_comparison.py` | Per-plot + batch TLS vs. UAV-LiDAR CHM comparison → side-by-side CHM maps, difference rasters, SD comparison chart, foliage-height boxplot, **and detailed written answers to the discussion + challenge questions**, saved to `UAV_TLS_comparison_answers.md` |
