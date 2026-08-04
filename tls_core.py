"""
tls_core.py
===========
Shared, reusable functions for the MyDiv TLS + UAV-LiDAR foliage-density
workflow. This is the Python equivalent of the `lidR`/`terra` R functions
used in the two course R Markdown scripts:

  1. "TLS_MyDiv_processing.Rmd"          -> ground classification, DTM,
                                             height normalization, CHM,
                                             voxelization, FHD, RH, LAI join
  2. "UAV_LiDAR & TLS CHM comparison.Rmd" -> UAV vs TLS CHM comparison

Every function name/parameter mirrors its R counterpart as closely as
possible so you can read this side-by-side with the original Rmd files.
All terms (FHD, RH, PAI, voxel, TIN, p2r, occlusion...) are explained in
`point_cloud_glossary.md`.
"""

import os
import glob
import re
import numpy as np
import laspy
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import rasterio
from rasterio.transform import from_origin

try:
    import CSF
    _HAS_CSF = True
except ImportError:
    _HAS_CSF = False


# ------------------------------------------------------------------
# 0. CRS handling  (R: epsg(tls) <- epsg_code ; st_crs(tls) <- epsg_code)
# ------------------------------------------------------------------

def assign_crs(las, epsg_code):
    """
    Attach a coordinate reference system to a laspy LasData object,
    equivalent to `epsg(tls) <- epsg_code` in lidR. This only stamps
    metadata (which CRS the X/Y/Z numbers *mean*) -- it does NOT reproject
    or move the points. Requires pyproj.
    """
    try:
        from pyproj import CRS
        las.header.add_crs(CRS.from_epsg(epsg_code))
    except Exception as e:
        print(f"      ! could not assign EPSG:{epsg_code} ({e}); "
              f"continuing without CRS metadata (does not affect calculations)")
    return las


# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------

def read_las(path):
    """Read a .las/.laz file -> (xyz array[N,3], laspy LasData)."""
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    return xyz, las


def filter_duplicates(xyz, las=None):
    """
    Remove points with duplicated XYZ (R: filter_duplicates(uav_plot48)).
    UAV-LiDAR exports frequently contain exact duplicate points.
    Returns (xyz_unique, keep_mask).
    """
    _, unique_idx = np.unique(xyz, axis=0, return_index=True)
    keep_mask = np.zeros(len(xyz), dtype=bool)
    keep_mask[unique_idx] = True
    return xyz[keep_mask], keep_mask


# ------------------------------------------------------------------
# 2. SOR filtering (see previous script, kept here for completeness)
# ------------------------------------------------------------------

def sor_filter(xyz, k=8, std_ratio=1.0):
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    thr = mean_dists.mean() + std_ratio * mean_dists.std()
    keep_mask = mean_dists <= thr
    return xyz[keep_mask], keep_mask


# ------------------------------------------------------------------
# 3. Ground classification (R: classify_ground(tls, algorithm = csf()))
# ------------------------------------------------------------------

SCENE_TO_RIGIDNESS = {"steep_slope": 1, "relief": 2, "flat": 3}


def voxel_downsample(xyz, voxel_size):
    """
    Reduce point density by keeping exactly one point per `voxel_size`
    grid cell (a 3D voxel-grid filter). This is NOT the same as
    voxelise_point_cloud() below (which builds foliage-density metrics) --
    this is a simple point-count REDUCTION step, standard practice before
    an expensive O(n) operation on a very dense point cloud.

    Why this matters here specifically: `classify_ground_csf()` has to
    hand the point cloud to the underlying CSF C++ library through a SWIG
    Python binding, which requires a plain Python list (`xyz.tolist()`
    internally). Converting tens of millions of points into nested Python
    float objects is extremely memory-hungry -- empirically, roughly
    200-250 BYTES of pure Python object overhead per point on top of the
    original numpy array (measured: 3 million points ~= 480 MB as a
    Python list, vs. 72 MB as a numpy array). A real multi-scan-position
    TLS plot with e.g. 26 million points can need several extra GB just
    for this one conversion step -- exactly what causes a plain "Killed"
    message (the Linux OOM killer) with no Python traceback at all.

    The fix: since CSF's cloth resolution is typically 0.5-2 m, there is
    no ground-classification benefit whatsoever from feeding it
    sub-centimeter point spacing -- downsampling first to (e.g.) 2 cm
    loses no meaningful information for THIS step, while cutting point
    count dramatically (empirically, an 8-million-point dense canopy scan
    downsamples to ~2.9 million points at 0.02 m voxels -- the resulting
    count is governed by the plot's actual physical surface area divided
    by voxel size, NOT by the raw scan density, so this scales safely
    however dense your original scan is).

    IMPORTANT: only the COPY passed into classify_ground_csf() needs to be
    downsampled. The full-resolution point cloud should still be used for
    everything else (height normalization, CHM, voxelization) -- those
    steps are vectorized numpy array operations with no Python-object
    conversion, and scale fine in memory even at tens of millions of
    points.

    Returns (xyz_downsampled, keep_mask).
    """
    if not voxel_size or voxel_size <= 0:
        return xyz, np.ones(len(xyz), dtype=bool)
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    keep_mask = np.zeros(len(xyz), dtype=bool)
    keep_mask[unique_idx] = True
    return xyz[keep_mask], keep_mask


def classify_ground_csf(xyz, scene="relief", cloth_resolution=1.0,
                         max_iterations=600, class_threshold=0.5,
                         time_step=0.65, rigidness=None,
                         max_cloth_cells=20_000_000):
    """
    Cloth Simulation Filter ground classification.
    Returns a boolean array, True = ground point.
    Same algorithm/parameters as CloudCompare's CSF plugin and lidR's `csf()`.

    SAFETY CHECK: CSF builds its cloth simulation grid as
        (X extent / cloth_resolution) x (Y extent / cloth_resolution)
    A plot that's genuinely ~11 x 11 m only needs a tiny cloth grid. But if
    the point cloud's X/Y extent is much larger than expected -- e.g. the
    .las wasn't actually clipped to a single plot, or there's a units/scale
    problem in how the file was exported -- this grid can silently balloon
    to hundreds of millions of cells and get the process OOM-killed at the
    "Configuring cloth..." stage, before CSF even touches your points. This
    check computes the extent up front and raises a clear error instead,
    naming the likely causes, rather than letting the OS silently kill the
    process with no traceback.
    """
    if not _HAS_CSF:
        raise ImportError(
            "The 'CSF' package is required for ground classification. "
            "Install with: pip install cloth-simulation-filter"
        )

    x_extent = xyz[:, 0].max() - xyz[:, 0].min()
    y_extent = xyz[:, 1].max() - xyz[:, 1].min()
    est_cells = (x_extent / cloth_resolution) * (y_extent / cloth_resolution)
    if est_cells > max_cloth_cells:
        raise ValueError(
            f"Refusing to run CSF: the point cloud's X/Y extent is "
            f"{x_extent:.1f} x {y_extent:.1f} m, which at "
            f"cloth_resolution={cloth_resolution} m would need a cloth grid "
            f"of ~{est_cells:,.0f} cells (limit: {max_cloth_cells:,}) -- this "
            f"would very likely get the process OOM-killed rather than "
            f"finishing. A single MyDiv plot should be roughly 11 x 11 m, "
            f"so this strongly suggests EITHER (1) this .las file isn't "
            f"actually clipped to one plot (it may cover a much larger "
            f"survey area -- check with a tool like CloudCompare or "
            f"`las.header.mins`/`las.header.maxs` in laspy), OR (2) a "
            f"units/scale problem in how the file was exported (check "
            f"`las.header.scales` and `las.header.offsets` -- an incorrect "
            f"scale factor can inflate real-world coordinates by 1000x or "
            f"more). Fix the input data, or if this extent is somehow "
            f"genuinely correct, raise `cloth_resolution` and/or "
            f"`max_cloth_cells` explicitly."
        )

    rigidness = rigidness or SCENE_TO_RIGIDNESS[scene]

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.interations = max_iterations

    csf.setPointCloud(xyz.tolist())
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)

    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[np.array(ground_idx)] = True
    return ground_mask


# ------------------------------------------------------------------
# 4. TIN interpolation, DTM, height normalization
#    (R: rasterize_terrain(tls_gc, algorithm = tin()))
#    (R: normalize_height(tls_gc, dtm))
# ------------------------------------------------------------------

def build_tin_interpolator(ground_xyz):
    """
    Build a Triangular Irregular Network (TIN) interpolator from ground
    points: a Delaunay triangulation in XY, linearly interpolating Z across
    each triangle. This is exactly what lidR's `tin()` algorithm does, and
    what CloudCompare does when it 'interpolates the points' while
    rasterizing the DTM.
    A NearestNDInterpolator is used as a fallback for query points that fall
    outside the convex hull of the ground points (edges of the plot).
    """
    lin_interp = LinearNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])
    near_interp = NearestNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])

    def interpolate(xy_query):
        z = lin_interp(xy_query)
        nan_mask = np.isnan(z)
        if nan_mask.any():
            z[nan_mask] = near_interp(xy_query[nan_mask])
        return z

    return interpolate


def rasterize_terrain_tin(ground_xyz, res):
    """
    DTM: rasterize ground points using TIN interpolation, evaluated at the
    center of every grid cell (R: rasterize_terrain(..., algorithm=tin())).
    Returns (grid, transform, extent).
    """
    interp = build_tin_interpolator(ground_xyz)

    xmin, xmax = ground_xyz[:, 0].min(), ground_xyz[:, 0].max()
    ymin, ymax = ground_xyz[:, 1].min(), ground_xyz[:, 1].max()
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1

    xs = xmin + (np.arange(ncols) + 0.5) * res
    ys = ymax - (np.arange(nrows) + 0.5) * res  # row 0 = north
    grid_x, grid_y = np.meshgrid(xs, ys)
    query_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    z = interp(query_xy)
    grid = z.reshape(nrows, ncols)

    transform = from_origin(xmin, ymax, res, res)
    extent = (xmin, xmax, ymin, ymax)
    return grid, transform, extent, interp


def normalize_height(xyz, ground_interp):
    """
    Height-normalize a point cloud: subtract the interpolated ground (TIN)
    elevation from every point's Z, so Z becomes 'height above ground'
    rather than absolute elevation.
    (R: nlas <- normalize_height(tls_gc, dtm)  /  nlas <- tls_gc - dtm)
    """
    ground_z = ground_interp(xyz[:, :2])
    z_norm = xyz[:, 2] - ground_z
    xyz_norm = xyz.copy()
    xyz_norm[:, 2] = z_norm
    return xyz_norm


# ------------------------------------------------------------------
# 5. Canopy height model, point-to-raster (R: rasterize_canopy(..., p2r()))
# ------------------------------------------------------------------

def rasterize_canopy_p2r(xyz_norm, res, fill_gaps=True):
    """
    CHM via the 'point-to-raster' (p2r) method: for every grid cell, take
    the maximum normalized height of any point that falls inside it.
    Empty cells (no points) stay NaN unless fill_gaps=True, in which case
    they're filled with the nearest non-empty cell's value
    (R: chm <- fill_chm_gaps(chm)).
    Returns (grid, transform, extent).
    """
    x, y, z = xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1

    col = np.clip(((x - xmin) / res).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - y) / res).astype(int), 0, nrows - 1)
    flat_idx = row * ncols + col

    flat_grid = np.full(nrows * ncols, -np.inf)
    np.maximum.at(flat_grid, flat_idx, z)
    flat_grid[flat_grid == -np.inf] = np.nan
    grid = flat_grid.reshape(nrows, ncols)

    if fill_gaps and np.isnan(grid).any() and (~np.isnan(grid)).sum() >= 1:
        rows_idx, cols_idx = np.where(~np.isnan(grid))
        vals = grid[~np.isnan(grid)]
        nn = NearestNDInterpolator(np.column_stack([rows_idx, cols_idx]), vals)
        nan_rows, nan_cols = np.where(np.isnan(grid))
        grid[nan_rows, nan_cols] = nn(np.column_stack([nan_rows, nan_cols]))

    transform = from_origin(xmin, ymax, res, res)
    extent = (xmin, xmax, ymin, ymax)
    return grid, transform, extent


def save_geotiff(grid, transform, path, nodata=-9999.0, crs=None):
    grid_out = np.where(np.isnan(grid), nodata, grid).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=grid_out.shape[0], width=grid_out.shape[1],
        count=1, dtype=grid_out.dtype, crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(grid_out, 1)


# ------------------------------------------------------------------
# 6. Voxelization + vertical foliage-density metrics
#    (R: voxelise_point_cloud / build_vertical_profile /
#         calc_fhd_from_profile / calc_relative_heights /
#         summarise_tls_structure)
# ------------------------------------------------------------------

def voxelise_point_cloud(xyz_norm, x_res=0.5, y_res=0.5, z_res=0.5,
                          min_z=0.0, max_z=None):
    """
    Convert a (height-normalized) point cloud into a set of OCCUPIED voxels
    (3D grid cells that contain at least one point). Returns a structured
    dict of arrays: ix, iy, iz (voxel indices), x_center, y_center, z_center,
    n_points (returns per voxel).
    """
    x, y, z = xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2]
    if max_z is None:
        max_z = np.nanmax(z)

    keep = (z >= min_z) & (z <= max_z)
    x, y, z = x[keep], y[keep], z[keep]

    x0, y0 = x.min(), y.min()
    ix = np.floor((x - x0) / x_res).astype(int)
    iy = np.floor((y - y0) / y_res).astype(int)
    iz = np.floor((z - min_z) / z_res).astype(int)

    # unique voxels + point counts per voxel
    voxel_ids = np.stack([ix, iy, iz], axis=1)
    uniq, inv, counts = np.unique(voxel_ids, axis=0, return_inverse=True,
                                   return_counts=True)

    x_center = x0 + (uniq[:, 0] + 0.5) * x_res
    y_center = y0 + (uniq[:, 1] + 0.5) * y_res
    z_center = min_z + (uniq[:, 2] + 0.5) * z_res

    return {
        "ix": uniq[:, 0], "iy": uniq[:, 1], "iz": uniq[:, 2],
        "x_center": x_center, "y_center": y_center, "z_center": z_center,
        "n_points": counts,
        "x_res": x_res, "y_res": y_res, "z_res": z_res,
    }


def build_vertical_profile(vox, layer_height=3.0):
    """
    Aggregate occupied voxels into coarser vertical layers (default 3 m
    slabs) and compute a simple Plant-Area-Index-like quantity per layer:
        PAI_layer = (number of occupied voxels in this layer) * voxel_height
    This is a proxy for foliage/plant area within that height slab -- more
    occupied voxels at a given height = more foliage/wood intercepted the
    laser at that height.
    Returns dict with arrays: layer_id, z_low, z_mid, z_high, PAI_layer,
    n_occupied_voxels.
    """
    layer_id = np.floor(vox["z_center"] / layer_height).astype(int)
    uniq_layers = np.unique(layer_id)

    n_occ = np.array([(layer_id == l).sum() for l in uniq_layers])
    pai = n_occ * vox["z_res"]

    z_low = uniq_layers * layer_height
    z_high = (uniq_layers + 1) * layer_height
    z_mid = z_low + layer_height / 2

    order = np.argsort(uniq_layers)
    return {
        "layer_id": uniq_layers[order],
        "z_low": z_low[order], "z_mid": z_mid[order], "z_high": z_high[order],
        "PAI_layer": pai[order], "n_occupied_voxels": n_occ[order],
    }


def calc_fhd_from_profile(profile, standardize=False):
    """
    Foliage Height Diversity (FHD), MacArthur & MacArthur (1961): the
    Shannon entropy of the vertical distribution of foliage/plant area.
    High FHD = foliage evenly spread across many height layers (structurally
    complex canopy). Low FHD = foliage concentrated in one layer (e.g. a
    simple, single-layer canopy).
    """
    pai = profile["PAI_layer"]
    mask = pai > 0
    if mask.sum() == 0:
        return np.nan
    total = pai[mask].sum()
    if total <= 0:
        return np.nan
    p = pai[mask] / total
    fhd = -np.sum(p * np.log(p))
    if standardize:
        n_layers = mask.sum()
        fhd = fhd / np.log(n_layers) if n_layers > 1 else 0.0
    return fhd


def calc_relative_heights(vox, probs=(0.25, 0.50, 0.75, 0.98)):
    """
    Relative height metrics (RH25, RH50, RH75, RH98, ...): the height below
    which a given proportion of the cumulative vertical foliage profile
    (PAI, bottom-to-top) is found. RH98 is commonly used as a robust proxy
    for "top of canopy height" (more robust to a single noisy top point than
    the raw max height).
    """
    # group occupied voxels by unique z_center (equivalent of R's group_by(z))
    uniq_z, counts = np.unique(vox["z_center"], return_counts=True)
    order = np.argsort(uniq_z)
    uniq_z, counts = uniq_z[order], counts[order]
    pai_layer = counts * vox["z_res"]

    total = pai_layer.sum()
    result = {}
    if total <= 0 or len(uniq_z) < 2:
        for p in probs:
            result[f"RH{int(p*100)}"] = np.nan
        return result

    cum_prop = np.cumsum(pai_layer) / total
    for p in probs:
        result[f"RH{int(p*100)}"] = np.interp(p, cum_prop, uniq_z)
    return result


def summarise_tls_structure(vox, plot_width=11.0, plot_length=11.0,
                             plot_height=15.0, min_z=0.0):
    """
    Plot-level canopy structure summary:
      - n_occupied_voxels: how many 3D cells actually contain a laser hit
      - occupancy_index_fixed_volume: occupied / total-possible voxels in a
        fixed plot volume -> a density index comparable across plots
      - canopy_volume_m3 / canopy_volume_per_ground_area: total volume of
        space occupied by vegetation, normalized by plot footprint (a
        structural-complexity proxy correlated with LAI)
      - max_height / mean_height: simple height statistics
    """
    n_occ = len(vox["ix"])
    voxel_volume = vox["x_res"] * vox["y_res"] * vox["z_res"]
    plot_area = plot_width * plot_length

    max_height = vox["z_center"].max() if n_occ else np.nan
    mean_height = vox["z_center"].mean() if n_occ else np.nan

    canopy_volume_m3 = n_occ * voxel_volume
    canopy_volume_per_ground_area = canopy_volume_m3 / plot_area

    n_x = int(np.ceil(plot_width / vox["x_res"]))
    n_y = int(np.ceil(plot_length / vox["y_res"]))
    n_z = int(np.ceil((plot_height - min_z) / vox["z_res"]))
    n_possible = n_x * n_y * n_z

    occupancy_index = n_occ / n_possible if n_possible else np.nan

    return {
        "n_occupied_voxels": n_occ,
        "n_possible_voxels_fixed_volume": n_possible,
        "occupancy_index_fixed_volume": occupancy_index,
        "canopy_volume_m3": canopy_volume_m3,
        "canopy_volume_per_ground_area": canopy_volume_per_ground_area,
        "max_height": max_height,
        "mean_height": mean_height,
    }


# ------------------------------------------------------------------
# 7. LI-COR LAI-2200(C) text file parser
#    (R: read_licor_lai())
# ------------------------------------------------------------------

def read_licor_lai(path):
    """
    Extract the LAI value (and plot id) from a LI-COR LAI-2200(C) summary
    .TXT export. The file is a mix of tab-separated KEY / VALUE(S) rows,
    e.g. one row is literally:  "LAI    5.054"
    We tokenize the whole file on whitespace and grab the token right after
    the exact token 'LAI' (this correctly skips 'LAI_FILE', which is a
    different token).
    Returns dict: {file, plot_id, LAI}
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    tokens = text.split()

    lai_value = None
    for i, tok in enumerate(tokens):
        if tok == "LAI" and i + 1 < len(tokens):
            try:
                lai_value = float(tokens[i + 1])
                break
            except ValueError:
                continue

    fname = os.path.basename(path)
    m = re.search(r"\d+", fname)
    plot_id = f"Plot{m.group(0)}" if m else None

    if lai_value is None:
        print(f"      ! WARNING: no LAI value found in {fname}")

    return {"file": fname, "plot_id": plot_id, "LAI": lai_value}


def read_all_lai(lai_dir, pattern="*.TXT"):
    """Parse every LI-COR .TXT file in a folder -> list of dicts."""
    files = sorted(glob.glob(os.path.join(lai_dir, pattern)))
    return [read_licor_lai(f) for f in files]


# ------------------------------------------------------------------
# 8. Grid resampling (R: terra::project(chm_tls, chm_uav, method="bilinear"))
# ------------------------------------------------------------------

def resample_grid_bilinear(src_grid, src_transform, dst_shape, dst_transform):
    """
    Resample a source raster grid onto a different (usually coarser) target
    grid, using bilinear interpolation -- same idea as
    `terra::project(tls_chm, uav_chm, method="bilinear")` in R.
    Note: since TLS/UAV plots here use a local (non-geographic) coordinate
    system, we use a placeholder projected CRS on both sides purely so
    rasterio's reprojection engine has a valid (matching) CRS to work with
    -- no actual geographic reprojection happens, only grid resampling
    based on the two transforms, which are already in the same local units.
    """
    from rasterio.warp import reproject, Resampling
    placeholder_crs = "EPSG:32633"  # arbitrary but valid projected CRS
    dst_grid = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=np.where(np.isnan(src_grid), -9999, src_grid).astype(np.float32),
        destination=dst_grid,
        src_transform=src_transform, src_crs=placeholder_crs,
        dst_transform=dst_transform, dst_crs=placeholder_crs,
        src_nodata=-9999, dst_nodata=-9999,
        resampling=Resampling.bilinear,
    )
    dst_grid[dst_grid == -9999] = np.nan
    return dst_grid