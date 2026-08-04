"""
02_uav_tls_chm_comparison.py
==============================
Python port of "UAV-LiDAR & TLS CHM comparison.Rmd".

For each plot:
  1. read the UAV-LiDAR point cloud (already classified: ground = 2)
  2. remove duplicate XYZ points
  3. build a UAV DTM (TIN, from the pre-existing ground classification)
  4. height-normalize + build the UAV CHM (point-to-raster, gaps filled)
  5. read/redo the TLS side (CSF ground classification + DTM + CHM,
     same as script 01, but at TLS resolution)
  6. resample the TLS CHM onto the (coarser) UAV CHM grid
  7. compute the difference raster (TLS - UAV) and summary statistics
  8. batch over all plots, produce a comparison table + figures
  9. answer the assignment's discussion + challenge questions (printed +
     saved to UAV_TLS_comparison_answers.md)

Run after requirements.txt is installed:
    python 02_uav_tls_chm_comparison.py
"""

import os
import glob
import numpy as np
import matplotlib
import rasterio
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tls_core as tc

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CONFIG = {
    "tls_dir": ".",             # Plot*_MyDiv.las -- "." = same folder as this script
    "uav_dir": ".",             # MyDiv_<n>_L1.las (pre-classified)
    "out_dir": "outputs/uav_tls_comparison",

    "plot_numbers": [43, 58, 70],
    "tls_name_pattern": "Plot{n}_MyDiv.las",   # {n} = plot number
    "uav_name_pattern": "MyDiv_{n}_L1.las",

    "epsg_code": 25833,

    # TLS side
    "tls_csf_scene": "relief",
    "tls_csf_cloth_resolution": 1.0,
    "tls_csf_max_iterations": 600,
    "tls_csf_class_threshold": 0.5,
    "tls_res": 0.05,

    # Downsample before CSF ground classification (meters) -- prevents an
    # out-of-memory "Killed" crash on large real TLS scans (tens of
    # millions of points). See tls_core.voxel_downsample()'s docstring.
    # Only affects ground classification; the CHM still uses full-res data.
    "tls_csf_downsample_voxel_m": 0.02,

    # If a plot's TLS and UAV point clouds turn out to be registered in
    # different coordinate systems (near-zero spatial overlap after
    # rasterization -- see process_one_plot()'s overlap check), whether to
    # automatically shift the TLS raster to align its centroid with the
    # UAV raster's centroid so a comparison can still be made. This is an
    # APPROXIMATION (assumes both scans cover the same physical area and
    # just disagree about the coordinate origin) -- set to False to
    # instead leave such plots as an honest all-NaN result.
    "auto_recenter_on_mismatch": True,

    # UAV side (already classified -> no CSF needed)
    "uav_res": 0.15,
}

SR_GROUPS = {
    19: "1SR", 76: "1SR", 68: "1SR", 70: "1SR",
    35: "2SR", 63: "2SR", 73: "2SR", 43: "2SR",
    40: "4SR", 48: "4SR", 74: "4SR", 58: "4SR",
}


# ----------------------------------------------------------------------
# TLS SIDE  (re-derives DTM/CHM at TLS resolution; reuses script 01 logic)
# ----------------------------------------------------------------------

def build_tls_chm(las_path, cfg):
    xyz, las = tc.read_las(las_path)
    if cfg["epsg_code"]:
        tc.assign_crs(las, cfg["epsg_code"])

    # SOR-filter the FULL point cloud first (matches script 01's structure).
    # WHY THIS MATTERS HERE, NOT JUST FOR CSF: a real scan can contain a
    # tiny handful of extreme stray points (sensor glitches, misregistered
    # returns, reflections off distant objects) that are invisible among
    # 26 million otherwise-normal points, but massively inflate the X/Y
    # bounding box. This doesn't just break CSF's cloth grid sizing (see
    # tls_core.classify_ground_csf()'s safety check) -- the CHM
    # rasterization below (rasterize_canopy_p2r) ALSO derives its grid
    # extent directly from the point array's own min/max, so the exact
    # same stray points would blow up the CHM's size too, even if CSF
    # were given a separately-cleaned copy. Filtering the full cloud once,
    # up front, protects every step that follows. (The original course
    # workflow for this comparison exercise didn't include a SOR step for
    # the TLS side -- this brings it into alignment with script 01, which
    # already did this and is why script 01 succeeded where this script
    # initially did not, on the same underlying file.)
    if cfg.get("tls_sor_before_processing", True):
        n_before = len(xyz)
        xyz, _ = tc.sor_filter(xyz, k=8, std_ratio=1.0)
        print(f"  SOR filter: kept {len(xyz):,} / {n_before:,} points")

    # Downsample a COPY for ground classification only -- see
    # tls_core.voxel_downsample()'s docstring: avoids an out-of-memory
    # "Killed" crash on large real TLS scans. The CHM below still uses
    # the full-resolution (but now SOR-filtered) point cloud.
    if cfg.get("tls_csf_downsample_voxel_m"):
        xyz_for_csf, _ = tc.voxel_downsample(xyz, cfg["tls_csf_downsample_voxel_m"])
        print(f"  downsampled {len(xyz):,} -> {len(xyz_for_csf):,} points "
              f"for CSF ground classification")
    else:
        xyz_for_csf = xyz

    ground_mask = tc.classify_ground_csf(
        xyz_for_csf, scene=cfg["tls_csf_scene"],
        cloth_resolution=cfg["tls_csf_cloth_resolution"],
        max_iterations=cfg["tls_csf_max_iterations"],
        class_threshold=cfg["tls_csf_class_threshold"],
    )
    ground_xyz = xyz_for_csf[ground_mask]

    # Thin ground points before TIN construction if very large -- Delaunay
    # triangulation doesn't benefit from ground points denser than the
    # DTM's own output resolution (see script 01's process_tls_plot() for
    # the same fix, with a fuller explanation in comments there).
    if len(ground_xyz) > 300_000:
        ground_xyz, _ = tc.voxel_downsample(ground_xyz, cfg["tls_res"])

    dtm, dtm_tr, dtm_ext, ground_interp = tc.rasterize_terrain_tin(ground_xyz, res=cfg["tls_res"])
    xyz_norm = tc.normalize_height(xyz, ground_interp)
    xyz_norm = xyz_norm[(xyz_norm[:, 2] >= 0) & (xyz_norm[:, 2] <= 40)]

    chm, chm_tr, chm_ext = tc.rasterize_canopy_p2r(xyz_norm, res=cfg["tls_res"], fill_gaps=True)
    return chm, chm_tr, chm_ext, xyz_norm


# ----------------------------------------------------------------------
# UAV SIDE  (already has Classification == 2 for ground -> no CSF needed)
# ----------------------------------------------------------------------

def build_uav_chm(las_path, cfg):
    import laspy
    las = laspy.read(las_path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    if cfg["epsg_code"]:
        tc.assign_crs(las, cfg["epsg_code"])

    xyz, keep_mask = tc.filter_duplicates(xyz)

    # Same rationale as the TLS side: a handful of stray points can blow
    # up the CHM's rasterization extent (rasterize_canopy_p2r derives its
    # grid size directly from the point array's own min/max), even though
    # this path doesn't run CSF and so never hits the cloth-grid safety
    # check. Filtering here protects the CHM step and the later TLS/UAV
    # grid alignment from the same failure mode.
    if cfg.get("uav_sor_before_processing", True):
        n_before = len(xyz)
        xyz, keep_mask2 = tc.sor_filter(xyz, k=8, std_ratio=1.0)
        keep_mask = keep_mask.copy()
        keep_mask[keep_mask] = keep_mask2
        print(f"  SOR filter (UAV): kept {len(xyz):,} / {n_before:,} points")

    if hasattr(las, "classification"):
        classification = np.asarray(las.classification)[keep_mask]
        ground_mask = classification == 2
        if ground_mask.sum() < 10:
            raise ValueError("UAV file has too few classified ground points "
                              "(expected Classification==2 from the provider)")
        ground_xyz = xyz[ground_mask]
    else:
        # Fallback: classify ourselves if the file has no classification
        # field. Downsample first too, in case this ever runs on a large
        # file (same rationale as the TLS side above).
        xyz_for_csf, _ = tc.voxel_downsample(xyz, cfg.get("tls_csf_downsample_voxel_m") or 0.02)
        ground_mask = tc.classify_ground_csf(xyz_for_csf, scene="flat")
        ground_xyz = xyz_for_csf[ground_mask]

    # Thin ground points before TIN construction if very large (same
    # rationale as the TLS side above).
    if len(ground_xyz) > 300_000:
        ground_xyz, _ = tc.voxel_downsample(ground_xyz, cfg["uav_res"])

    dtm, dtm_tr, dtm_ext, ground_interp = tc.rasterize_terrain_tin(ground_xyz, res=cfg["uav_res"])

    xyz_norm = tc.normalize_height(xyz, ground_interp)
    xyz_norm = xyz_norm[(xyz_norm[:, 2] >= 0) & (xyz_norm[:, 2] <= 40)]

    chm, chm_tr, chm_ext = tc.rasterize_canopy_p2r(xyz_norm, res=cfg["uav_res"], fill_gaps=True)
    return chm, chm_tr, chm_ext, xyz_norm


# ----------------------------------------------------------------------
# ONE PLOT: TLS vs UAV
# ----------------------------------------------------------------------

def raster_extent(transform, shape):
    """Returns (xmin, xmax, ymin, ymax) in real-world coordinates for a raster."""
    res = transform.a
    xmin, ymax = transform.c, transform.f
    xmax = xmin + shape[1] * res
    ymin = ymax - shape[0] * res
    return xmin, xmax, ymin, ymax


def overlap_fraction(extent_a, extent_b):
    """Fraction of extent_a's area that overlaps with extent_b (0 = no overlap)."""
    ax0, ax1, ay0, ay1 = extent_a
    bx0, bx1, by0, by1 = extent_b
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    a_area = (ax1 - ax0) * (ay1 - ay0)
    return inter_area / a_area if a_area > 0 else 0.0


def process_one_plot(n, cfg, out_dir):
    print(f"\n=== Plot {n} ===")
    tls_path = os.path.join(cfg["tls_dir"], cfg["tls_name_pattern"].format(n=n))
    uav_path = os.path.join(cfg["uav_dir"], cfg["uav_name_pattern"].format(n=n))

    if not os.path.exists(tls_path):
        raise FileNotFoundError(f"Missing TLS file: {tls_path}")
    if not os.path.exists(uav_path):
        raise FileNotFoundError(f"Missing UAV file: {uav_path}")

    chm_tls, chm_tls_tr, _, _ = build_tls_chm(tls_path, cfg)
    print(f"  TLS CHM: shape={chm_tls.shape}, mean={np.nanmean(chm_tls):.2f} m")

    chm_uav, chm_uav_tr, _, _ = build_uav_chm(uav_path, cfg)
    print(f"  UAV CHM: shape={chm_uav.shape}, mean={np.nanmean(chm_uav):.2f} m")

    # Check whether the TLS and UAV rasters actually occupy the same
    # real-world space BEFORE resampling. If your TLS and UAV .las files
    # were exported/registered with different coordinate systems for this
    # plot (e.g. one uses real UTM, the other a local/relative origin --
    # this happens when scans are processed inconsistently across a
    # dataset), the two rasters can end up nowhere near each other in
    # absolute coordinates, silently producing an all-NaN difference
    # raster (numpy's "Mean of empty slice" warning) instead of a
    # meaningful comparison.
    tls_extent = raster_extent(chm_tls_tr, chm_tls.shape)
    uav_extent = raster_extent(chm_uav_tr, chm_uav.shape)
    frac = overlap_fraction(tls_extent, uav_extent)

    if frac < 0.05:
        print(f"  ! WARNING: TLS and UAV rasters for Plot {n} overlap by only "
              f"{frac*100:.1f}% of the TLS extent -- they are very likely "
              f"registered in DIFFERENT coordinate systems for this plot.")
        print(f"    TLS extent  (x,y): [{tls_extent[0]:.1f}, {tls_extent[1]:.1f}] x "
              f"[{tls_extent[2]:.1f}, {tls_extent[3]:.1f}]")
        print(f"    UAV extent  (x,y): [{uav_extent[0]:.1f}, {uav_extent[1]:.1f}] x "
              f"[{uav_extent[2]:.1f}, {uav_extent[3]:.1f}]")
        if cfg.get("auto_recenter_on_mismatch", True):
            # Best-effort fallback: shift the TLS raster's origin so its
            # centroid lines up with the UAV raster's centroid, and redo
            # the comparison on that basis. This is an APPROXIMATION -- it
            # assumes the two scans cover the same physical area and just
            # disagree about where "the origin" is, which is a reasonable
            # assumption for a plot-level comparison but is NOT a rigorous
            # georeferencing fix. Results from a recentered plot should be
            # flagged as approximate in any write-up.
            tls_cx = (tls_extent[0] + tls_extent[1]) / 2
            tls_cy = (tls_extent[2] + tls_extent[3]) / 2
            uav_cx = (uav_extent[0] + uav_extent[1]) / 2
            uav_cy = (uav_extent[2] + uav_extent[3]) / 2
            dx, dy = uav_cx - tls_cx, uav_cy - tls_cy
            chm_tls_tr = rasterio.Affine(chm_tls_tr.a, chm_tls_tr.b, chm_tls_tr.c + dx,
                                          chm_tls_tr.d, chm_tls_tr.e, chm_tls_tr.f + dy)
            print(f"    -> auto-recentered TLS raster by (dx={dx:.1f}, dy={dy:.1f}) m "
              f"to align centroids with the UAV raster (approximate; set "
              f"'auto_recenter_on_mismatch': False in CONFIG to disable and "
              f"keep the NaN result instead)")
        else:
            print(f"    -> 'auto_recenter_on_mismatch' is False -- leaving this "
                  f"plot's comparison as-is (will likely be all-NaN)")

    # Resample TLS CHM onto the (coarser) UAV grid, like terra::project(..., "bilinear")
    tls_on_uav_grid = tc.resample_grid_bilinear(
        chm_tls, chm_tls_tr, chm_uav.shape, chm_uav_tr
    )

    diff = tls_on_uav_grid - chm_uav  # TLS - UAV

    tc.save_geotiff(chm_tls, chm_tls_tr, os.path.join(out_dir, f"Plot{n}_TLS_CHM.tif"))
    tc.save_geotiff(chm_uav, chm_uav_tr, os.path.join(out_dir, f"Plot{n}_UAV_CHM.tif"))
    tc.save_geotiff(diff, chm_uav_tr, os.path.join(out_dir, f"Plot{n}_TLSminusUAV.tif"))

    # side-by-side preview
    # IMPORTANT: plot all 3 panels using their REAL-WORLD extent (meters),
    # not raw pixel indices. TLS is rasterized at tls_res (e.g. 0.05 m/px)
    # and UAV at uav_res (e.g. 0.15 m/px) -- for the SAME ~11x11 m plot,
    # that's roughly 220x220 TLS pixels vs. ~73x73 UAV pixels. Plotting
    # both with plain imshow(grid) (no extent) shows each on its own pixel
    # -index scale, making a side-by-side visual comparison misleading
    # even though neither individual image is "stretched" per se. Using
    # each raster's real extent + aspect="equal" puts all three panels on
    # the same physical (meter) scale, so their apparent sizes are
    # directly comparable.
    tls_extent_plot = raster_extent(chm_tls_tr, chm_tls.shape)
    uav_extent_plot = raster_extent(chm_uav_tr, chm_uav.shape)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, grid, title, cmap, ext in [
        (axes[0], chm_tls, f"Plot {n} - TLS CHM", "viridis", tls_extent_plot),
        (axes[1], chm_uav, f"Plot {n} - UAV CHM", "viridis", uav_extent_plot),
        (axes[2], diff, f"Plot {n} - TLS minus UAV", "RdBu_r", uav_extent_plot),
    ]:
        extent_mpl = (ext[0], ext[1], ext[2], ext[3])  # (xmin, xmax, ymin, ymax)
        im = ax.imshow(grid, cmap=cmap, extent=extent_mpl, origin="upper", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Plot{n}_TLS_vs_UAV.png"), dpi=250)
    plt.close()

    return {
        "plot_number": n,
        "SR": SR_GROUPS.get(n),
        "TLS_mean": float(np.nanmean(chm_tls)),
        "UAV_mean": float(np.nanmean(chm_uav)),
        "TLS_sd": float(np.nanstd(chm_tls)),
        "UAV_sd": float(np.nanstd(chm_uav)),
        "TLS_max": float(np.nanmax(chm_tls)),
        "UAV_max": float(np.nanmax(chm_uav)),
        "diff_mean": float(np.nanmean(diff)),
        "diff_sd": float(np.nanstd(diff)),
        "SD_difference": float(np.nanstd(chm_tls) - np.nanstd(chm_uav)),
    }, chm_tls, chm_uav


def batch_process(cfg):
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    all_tls_heights = {}
    all_uav_heights = {}
    for n in cfg["plot_numbers"]:
        try:
            row, chm_tls, chm_uav = process_one_plot(n, cfg, out_dir)
            rows.append(row)
            all_tls_heights[n] = chm_tls[~np.isnan(chm_tls)].ravel()
            all_uav_heights[n] = chm_uav[~np.isnan(chm_uav)].ravel()
        except Exception as e:
            print(f"  ! skipped plot {n}: {e}")
    return rows, all_tls_heights, all_uav_heights


def save_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"saved {path}")


def plot_sd_comparison(rows, out_dir):
    if not rows:
        return
    plot_numbers = [r["plot_number"] for r in rows]
    tls_sd = [r["TLS_sd"] for r in rows]
    uav_sd = [r["UAV_sd"] for r in rows]

    x = np.arange(len(plot_numbers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, tls_sd, width, label="TLS_sd")
    ax.bar(x + width / 2, uav_sd, width, label="UAV_sd")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in plot_numbers])
    ax.set_xlabel("MyDiv plot")
    ax.set_ylabel("Standard deviation of CHM height (m)")
    ax.legend(title="Sensor")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "CHM_sd_comparison.png"), dpi=250)
    plt.close()
    print(f"saved {os.path.join(out_dir, 'CHM_sd_comparison.png')}")


def plot_foliage_boxplot(all_tls_heights, all_uav_heights, out_dir):
    """
    Challenge question figure: boxplot of the normalized-height distribution
    of CHM cells per plot, split by sensor -- a simple proxy for how
    foliage/point density is distributed across height strata (a taller box
    / higher median = points spread higher in the canopy; a wide box = more
    variable heights = more vertical structure captured).
    """
    if not all_tls_heights:
        return
    plot_numbers = sorted(all_tls_heights.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    positions_tls = np.arange(len(plot_numbers)) * 2.0
    positions_uav = positions_tls + 0.7

    bp1 = ax.boxplot([all_tls_heights[n] for n in plot_numbers], positions=positions_tls,
                      widths=0.5, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot([all_uav_heights[n] for n in plot_numbers], positions=positions_uav,
                      widths=0.5, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("tab:green")
    for patch in bp2["boxes"]:
        patch.set_facecolor("tab:orange")

    ax.set_xticks(positions_tls + 0.35)
    ax.set_xticklabels([str(n) for n in plot_numbers])
    ax.set_xlabel("MyDiv plot")
    ax.set_ylabel("CHM height above ground (m)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["TLS", "UAV"])
    ax.set_title("Distribution of canopy heights per plot: TLS vs UAV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "foliage_height_boxplot_TLS_vs_UAV.png"), dpi=250)
    plt.close()
    print(f"saved {os.path.join(out_dir, 'foliage_height_boxplot_TLS_vs_UAV.png')}")


# ----------------------------------------------------------------------
# ASSIGNMENT ANSWERS
# ----------------------------------------------------------------------

ASSIGNMENT_ANSWERS = """
# UAV-LiDAR vs TLS CHM Comparison — Answers

### Question 1
"Despite representing the same forest plot, the TLS and UAV-derived CHMs
are not identical. Identify and discuss at least three factors that explain
these differences, considering sensor geometry, spatial resolution, point
density, canopy occlusion, and georeferencing."

1. **Sensor geometry / viewing direction.** TLS scans from a tripod ON THE
   GROUND, looking up and outward through the canopy; UAV-LiDAR scans from
   ABOVE, looking down onto the canopy surface. These are fundamentally
   different viewing geometries: TLS is extremely good at capturing stems,
   understory, and the underside of the canopy, but struggles to see the
   very top of tall trees (near-nadir returns from directly overhead are
   geometrically hard for a ground-based scanner to capture). UAV-LiDAR is
   the opposite: excellent at capturing the true top-of-canopy envelope,
   but blind to anything hidden beneath the outer canopy shell. This alone
   explains most of the systematic CHM differences, especially near the top
   of the canopy and inside dense crowns.

2. **Canopy occlusion.** TLS point density decreases radially and drops off
   sharply behind any obstruction (branches, trunks, dense foliage) between
   the scanner and a given point in space -- producing the "shadow wedges"
   of missing/reduced-density data visible in the TLS-minus-UAV difference
   raster, radiating outward from the scan position(s). UAV-LiDAR has a
   comparatively unobstructed view of the canopy top from directly above
   (multiple flight lines further reduce occlusion), so it rarely
   experiences this same wedge-shaped data-gap pattern.

3. **Spatial resolution and point density.** TLS was rasterized at 0.05 m
   and typically has very high point density close to the scanner (and
   lower density farther away), while UAV-LiDAR here was rasterized at
   0.15 m and usually has a more spatially uniform point density across the
   whole plot (from a moving airborne platform). Different resolutions and
   point densities change how much small-scale canopy texture (individual
   leaf clusters, small gaps) is captured vs. smoothed out -- the TLS CHM
   typically looks "noisier"/more textured (reflected in a higher `TLS_sd`
   than `UAV_sd` in the printed statistics), while the UAV CHM looks smoother.

4. **Georeferencing / spatial alignment.** If the TLS scan's local
   coordinate system and the UAV point cloud's coordinate system were
   registered/georeferenced independently (e.g. different GNSS
   corrections, different registration targets, or a manual global shift
   in CloudCompare that wasn't identical between datasets), even a small
   (sub-meter) horizontal or vertical offset between the two point clouds
   will show up as an apparent height difference at canopy edges after
   resampling onto a common grid -- this is a purely geometric/registration
   artifact, not a real structural difference, and is hardest to fully rule
   out without independent ground control points.

5. **Different DTM/ground models.** TLS ground points were classified with
   CSF (this script) at 0.05 m resolution; UAV ground points came
   pre-classified by the data provider and were interpolated at 0.15 m.
   Small differences in where each sensor puts "0 m" (the ground surface)
   propagate directly into the height-normalized CHM, especially on sloped
   or micro-topographically complex ground.

Look at your own `Plot{n}_TLS_vs_UAV.png` panels and the
`CHM_sd_comparison.png` bar chart: plots where `TLS_sd` is much higher than
`UAV_sd` are the ones where TLS captured much more fine-scale canopy texture
(or noise) than the smoother, coarser-resolution UAV CHM.

---

### Challenge question 2
"How different are the foliage density from TLS data and UAV-LiDAR? Plot
the boxplot per plot and discuss the distribution of points across height
strata."

See `foliage_height_boxplot_TLS_vs_UAV.png` (produced by this script): for
each plot it shows the distribution of CHM cell heights (a proxy for where,
vertically, canopy surface material was detected) as a box-and-whisker pair,
TLS (green) vs UAV (orange).

What to look for and discuss:
- **Median height**: if the UAV median sits noticeably higher than the TLS
  median for a given plot, that's consistent with TLS underestimating
  upper-canopy presence due to occlusion (Q1/Q5 above) -- the "true" top of
  canopy is more reliably captured from above.
- **Box width / whisker spread (interquartile range)**: a WIDER box for TLS
  suggests the ground-based scan is picking up structure across many more
  height levels (branches, understory, mid-canopy, plus canopy top all
  contribute returns) -- more raw vertical detail, but not necessarily more
  ACCURATE canopy-top detection. A narrower, tighter UAV box close to the
  canopy top suggests UAV-LiDAR is consistently and precisely capturing the
  canopy surface itself, with comparatively little "noise" from lower strata
  (which the UAV pulses may never even reach if the canopy is closed).
- **Outliers / long whiskers**: extreme low values in the TLS distribution
  usually correspond to ground/understory hits that survived the CHM's
  point-to-raster maximum in nearly empty cells (rare in a closed canopy,
  more common at plot edges or through canopy gaps); UAV rarely produces
  these because it mostly measures the closed upper surface.
- **Between-plot comparison**: plots with a bigger gap between the TLS and
  UAV distributions are plots where occlusion/vertical-structure effects are
  strongest -- often the same plots with the largest `|TLS_sd - UAV_sd|`
  ("SD_difference" column) and the largest visible "shadow wedges" in the
  Plot{n}_TLS_vs_UAV.png difference panel.
- Overall, expect TLS to show a systematically wider, lower-shifted
  distribution (more low-height noise, less reliable top-of-canopy signal),
  while UAV shows a tighter, higher, more consistent distribution
  concentrated near the true canopy surface -- exactly mirroring the
  ground-based vs. above-canopy viewing geometry difference discussed in Q1.
"""


def save_answers(path):
    with open(path, "w") as f:
        f.write(ASSIGNMENT_ANSWERS)
    print(f"\nSaved detailed assignment answers to: {path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main(cfg=CONFIG):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    rows, all_tls_heights, all_uav_heights = batch_process(cfg)
    save_csv(rows, os.path.join(cfg["out_dir"], "chm_summary.csv"))
    plot_sd_comparison(rows, cfg["out_dir"])
    plot_foliage_boxplot(all_tls_heights, all_uav_heights, cfg["out_dir"])
    save_answers(os.path.join(cfg["out_dir"], "UAV_TLS_comparison_answers.md"))

    print("\nDone. See:", cfg["out_dir"])


if __name__ == "__main__":
    main()