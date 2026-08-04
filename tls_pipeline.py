"""
TLS Point Cloud Processing Pipeline
====================================
Reproduces, in Python, the CloudCompare workflow described in the
"Sampling Design & Ground Truthing in Remote Sensing" handout:

  1. Load a .las point cloud (e.g. Plot_48_MyDiv.las)
  2. Remove noisy/saturated points (Statistical Outlier Removal, SOR)
  3. Classify ground vs. non-ground points (Cloth Simulation Filter, CSF)
  4. Rasterize ground points -> Digital Terrain Model (DTM)
  5. Rasterize all/top points -> Digital Surface Model (DSM)
  6. DSM - DTM -> Canopy Height Model (CHM)
  7. Save everything as GeoTIFF + LAS + preview PNGs + a ground mesh

Every term used below is explained in the accompanying
`point_cloud_glossary.md` file.

Author: generated for the user's MyDiv TLS exercise
"""

import os
import numpy as np
import laspy
from scipy.spatial import cKDTree, Delaunay
import rasterio
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")  # no display needed, we save PNGs to disk
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. CONFIGURATION -- edit these values for your own data / preferences
# ----------------------------------------------------------------------

CONFIG = {
    # --- input / output ---
    "input_las": "MyDiv_43_L1.las",
    "output_dir": "outputs",
    
    #"input_las": "/mnt/user-data/uploads/Plot_48_MyDiv.las",
    #"output_dir": "/mnt/user-data/outputs",

    # --- SOR filtering (CC: Tools > Clean > SOR filter) ---
    "sor_k_neighbors": 8,      # how many nearest neighbours to check per point
    "sor_std_ratio": 1.0,      # points further than mean + ratio*std are removed

    # --- CSF ground classification (CC: CSF plugin) ---
    "csf_scene": "relief",        # "steep_slope" | "relief" | "flat"
    "csf_cloth_resolution": 1.0,  # grid spacing of the simulated cloth (m)
    "csf_max_iterations": 600,    # Max Iterations
    "csf_classification_threshold": 0.5,  # distance (m) cloth<->point => ground
    "csf_time_step": 0.65,
    "csf_rigidness": None,     # None -> derived automatically from csf_scene

    # --- Rasterization (CC: Rasterize) ---
    "raster_resolution": 0.05,  # 5 cm grid cells, matches the handout

    # --- misc ---
    "chm_min_height": 0.0,   # clip CHM below this to 0 (removes noise dips)
}

# CloudCompare's 3 CSF "Scenes" presets map onto the underlying `rigidness`
# parameter of the original CSF algorithm (Zhang et al., 2016):
#   Steep slope -> rigidness 1 (soft cloth, follows slopes)
#   Relief      -> rigidness 2 (medium)
#   Flat        -> rigidness 3 (stiff cloth, best for flat/open ground)
SCENE_TO_RIGIDNESS = {"steep_slope": 1, "relief": 2, "flat": 3}


# ----------------------------------------------------------------------
# 2. LOAD
# ----------------------------------------------------------------------

def load_las(path):
    """Read a .las/.laz file and return (xyz array, laspy LasData object)."""
    print(f"[1/6] Loading point cloud: {path}")
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    print(f"      -> {xyz.shape[0]:,} points loaded")
    print(f"      -> local bounding box: "
          f"X[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
          f"Y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
          f"Z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    return xyz, las


# ----------------------------------------------------------------------
# 3. SOR FILTER  (Statistical Outlier Removal)
# ----------------------------------------------------------------------

def sor_filter(xyz, k=8, std_ratio=1.0):
    """
    Statistical Outlier Removal.
    For every point, compute the mean distance to its k nearest neighbours.
    Points whose mean distance is larger than
        global_mean + std_ratio * global_std
    are considered noise/outliers and removed.
    Returns: (xyz_clean, keep_mask)
    """
    print(f"[2/6] SOR filtering (k={k}, std_ratio={std_ratio}) ...")
    tree = cKDTree(xyz)
    # query k+1 because the first neighbour returned is the point itself
    dists, _ = tree.query(xyz, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    keep_mask = mean_dists <= threshold
    print(f"      -> removed {np.sum(~keep_mask):,} outlier points "
          f"({100*np.sum(~keep_mask)/len(xyz):.2f}%)")
    return xyz[keep_mask], keep_mask


# ----------------------------------------------------------------------
# 4. CSF GROUND CLASSIFICATION
# ----------------------------------------------------------------------

def csf_ground_classification(xyz, cfg):
    """
    Cloth Simulation Filter (Zhang et al. 2016), same algorithm used by
    CloudCompare's CSF plugin. Requires the `cloth-simulation-filter`
    pip package (imported as `CSF`).
    Returns: (ground_mask) boolean array, True = ground point
    """
    import CSF
    print("[3/6] CSF ground classification ...")

    rigidness = cfg["csf_rigidness"] or SCENE_TO_RIGIDNESS[cfg["csf_scene"]]

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = cfg["csf_cloth_resolution"]
    csf.params.rigidness = rigidness
    csf.params.time_step = cfg["csf_time_step"]
    csf.params.class_threshold = cfg["csf_classification_threshold"]
    csf.params.interations = cfg["csf_max_iterations"]

    csf.setPointCloud(xyz.tolist())
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)

    ground_idx = np.array(ground_idx)
    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[ground_idx] = True

    print(f"      -> scene='{cfg['csf_scene']}' (rigidness={rigidness}), "
          f"cloth_res={cfg['csf_cloth_resolution']}, "
          f"iterations={cfg['csf_max_iterations']}, "
          f"threshold={cfg['csf_classification_threshold']}")
    print(f"      -> {ground_mask.sum():,} ground points, "
          f"{(~ground_mask).sum():,} non-ground (vegetation/other) points")
    return ground_mask


# ----------------------------------------------------------------------
# 5. RASTERIZATION -> DTM / DSM / CHM
# ----------------------------------------------------------------------

def rasterize_points(xyz, resolution, agg="max", fill_method="linear"):
    """
    Turn scattered XYZ points into a regular raster grid.
      agg="max"  -> per-cell max Z  (used for DSM, top-of-canopy surface)
      agg="mean" -> per-cell mean Z (used for DTM before interpolation)
    Empty cells are filled by linear interpolation (griddata), matching
    CC's "Rasterize ... and interpolate the points" step.
    Returns: (grid Z array [rows,cols], transform, extent)
    """
    from scipy.interpolate import griddata

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    ncols = int(np.ceil((xmax - xmin) / resolution)) + 1
    nrows = int(np.ceil((ymax - ymin) / resolution)) + 1

    col = np.clip(((x - xmin) / resolution).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - y) / resolution).astype(int), 0, nrows - 1)  # row0 = north

    grid = np.full((nrows, ncols), np.nan)
    count = np.zeros((nrows, ncols), dtype=int)

    if agg == "max":
        # np.maximum.at handles repeated indices correctly
        flat_idx = row * ncols + col
        flat_grid = np.full(nrows * ncols, -np.inf)
        np.maximum.at(flat_grid, flat_idx, z)
        flat_grid[flat_grid == -np.inf] = np.nan
        grid = flat_grid.reshape(nrows, ncols)
    elif agg == "mean":
        flat_idx = row * ncols + col
        sums = np.zeros(nrows * ncols)
        cnts = np.zeros(nrows * ncols)
        np.add.at(sums, flat_idx, z)
        np.add.at(cnts, flat_idx, 1)
        with np.errstate(invalid="ignore"):
            flat_grid = sums / cnts
        flat_grid[cnts == 0] = np.nan
        grid = flat_grid.reshape(nrows, ncols)
    else:
        raise ValueError("agg must be 'max' or 'mean'")

    # interpolate empty cells (gaps with no returns)
    valid = ~np.isnan(grid)
    if valid.sum() >= 4 and (~valid).any():
        rows_idx, cols_idx = np.where(valid)
        vals = grid[valid]
        all_rows, all_cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
        try:
            filled = griddata(
                (rows_idx, cols_idx), vals,
                (all_rows, all_cols), method=fill_method
            )
            # anything still NaN (outside convex hull) -> nearest
            still_nan = np.isnan(filled)
            if still_nan.any():
                nearest = griddata(
                    (rows_idx, cols_idx), vals,
                    (all_rows, all_cols), method="nearest"
                )
                filled[still_nan] = nearest[still_nan]
            grid = filled
        except Exception as e:
            print(f"      ! interpolation warning: {e}")

    transform = from_origin(xmin, ymax, resolution, resolution)
    extent = (xmin, xmax, ymin, ymax)
    return grid, transform, extent


def compute_chm(dsm, dtm, min_height=0.0):
    """Canopy Height Model = surface height above the ground."""
    chm = dsm - dtm
    chm = np.clip(chm, min_height, None)
    return chm


# ----------------------------------------------------------------------
# 6. SAVE OUTPUTS
# ----------------------------------------------------------------------

def save_geotiff(grid, transform, path, nodata=-9999.0):
    grid_out = np.where(np.isnan(grid), nodata, grid).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=grid_out.shape[0], width=grid_out.shape[1],
        count=1, dtype=grid_out.dtype,
        crs=None,  # No geographic CRS: TLS data uses a local/relative
                   # coordinate system (see glossary: "Why no geo coord in CC?")
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(grid_out, 1)
    print(f"      -> saved {path}")


def save_las_subset(las, mask, path, classification_value=None):
    """Save a subset of points (boolean mask) to a new LAS file."""
    sub = las[mask]
    if classification_value is not None:
        sub.classification = np.full(len(sub.points), classification_value, dtype=np.uint8)
    sub.write(path)
    print(f"      -> saved {path} ({mask.sum():,} points)")


def save_raster_preview(grid, extent, title, path, cmap="terrain"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, extent=extent, origin="upper", cmap=cmap)
    plt.colorbar(im, label="Height (m)")
    plt.title(title)
    plt.xlabel("X (local, m)")
    plt.ylabel("Y (local, m)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"      -> saved {path}")


def save_ground_mesh(ground_xyz, path_obj, path_png):
    """
    Triangulate the ground points (2.5D Delaunay in XY) to build a mesh,
    equivalent to CC's 'off-ground_points.mesh' preview. Saves an OBJ mesh
    and a shaded PNG preview.
    """
    print("      -> triangulating ground mesh (Delaunay in XY) ...")
    # Subsample if huge, to keep triangulation fast
    max_pts = 60000
    if len(ground_xyz) > max_pts:
        idx = np.random.choice(len(ground_xyz), max_pts, replace=False)
        pts = ground_xyz[idx]
    else:
        pts = ground_xyz

    tri = Delaunay(pts[:, :2])

    with open(path_obj, "w") as f:
        for p in pts:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for simplex in tri.simplices:
            f.write(f"f {simplex[0]+1} {simplex[1]+1} {simplex[2]+1}\n")
    print(f"      -> saved {path_obj}")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                     triangles=tri.simplices, cmap="terrain",
                     linewidth=0.05, antialiased=True)
    ax.set_title("Ground mesh (DTM surface)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
    print(f"      -> saved {path_png}")


# ----------------------------------------------------------------------
# 7. MAIN PIPELINE
# ----------------------------------------------------------------------

def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out = cfg["output_dir"]

    # 1. Load
    xyz, las = load_las(cfg["input_las"])

    # 2. SOR filter (remove noisy/saturated points)
    xyz_clean, keep_mask = sor_filter(
        xyz, k=cfg["sor_k_neighbors"], std_ratio=cfg["sor_std_ratio"]
    )
    save_las_subset(las, keep_mask, os.path.join(out, "01_sor_filtered.las"))

    # 3. CSF ground classification
    ground_mask = csf_ground_classification(xyz_clean, cfg)

    # Build a laspy object matching xyz_clean for saving classified outputs
    las_clean = las[keep_mask]
    print("[4/6] Saving classified point clouds ...")
    save_las_subset(las_clean, ground_mask,
                     os.path.join(out, "02_ground_points.las"),
                     classification_value=2)   # ASPRS class 2 = ground
    save_las_subset(las_clean, ~ground_mask,
                     os.path.join(out, "03_nonground_vegetation_points.las"),
                     classification_value=1)   # ASPRS class 1 = unclassified/veg

    ground_xyz = xyz_clean[ground_mask]
    all_xyz = xyz_clean  # top surface (DSM) uses all returns

    # 4. Rasterize -> DTM (from ground points only) and DSM (from all points)
    print("[5/6] Rasterizing DTM / DSM / CHM "
          f"(resolution={cfg['raster_resolution']} m) ...")
    dtm, dtm_transform, dtm_extent = rasterize_points(
        ground_xyz, cfg["raster_resolution"], agg="mean"
    )
    dsm, dsm_transform, dsm_extent = rasterize_points(
        all_xyz, cfg["raster_resolution"], agg="max"
    )
    chm = compute_chm(dsm, dtm, cfg["chm_min_height"])

    save_geotiff(dtm, dtm_transform, os.path.join(out, "DTM.tif"))
    save_geotiff(dsm, dsm_transform, os.path.join(out, "DSM.tif"))
    save_geotiff(chm, dtm_transform, os.path.join(out, "CHM.tif"))

    save_raster_preview(dtm, dtm_extent, "DTM - Digital Terrain Model",
                         os.path.join(out, "DTM_preview.png"), cmap="terrain")
    save_raster_preview(dsm, dsm_extent, "DSM - Digital Surface Model",
                         os.path.join(out, "DSM_preview.png"), cmap="terrain")
    save_raster_preview(chm, dtm_extent, "CHM - Canopy Height Model",
                         os.path.join(out, "CHM_preview.png"), cmap="viridis")

    # 6. Ground mesh (equivalent of CC's off-ground_points.mesh)
    print("[6/6] Building ground mesh preview ...")
    save_ground_mesh(
        ground_xyz,
        os.path.join(out, "ground_mesh.obj"),
        os.path.join(out, "ground_mesh_preview.png"),
    )

    print("\nDone! All outputs saved to:", out)
    print(" - 01_sor_filtered.las               (cleaned point cloud)")
    print(" - 02_ground_points.las              (classification = 2)")
    print(" - 03_nonground_vegetation_points.las(classification = 1)")
    print(" - DTM.tif / DSM.tif / CHM.tif        (GeoTIFF rasters)")
    print(" - *_preview.png                      (quick-look images)")
    print(" - ground_mesh.obj + preview           (triangulated ground mesh)")


if __name__ == "__main__":
    main()
