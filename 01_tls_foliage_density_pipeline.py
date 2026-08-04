"""
01_tls_foliage_density_pipeline.py
===================================
Python port of "TLS_MyDiv_processing.Rmd".

Pipeline per plot:
  1. read .las
  2. (optionally) assign CRS
  3. classify ground points (CSF)
  4. build DTM (TIN interpolation of ground points)
  5. height-normalize the point cloud (subtract DTM)
  6. build CHM (point-to-raster of normalized heights)
  7. voxelize the normalized vegetation points
  8. compute vertical foliage-density metrics: FHD, RH25/50/75/98,
     canopy volume, occupancy index
  9. parse LI-COR LAI .TXT files and join with TLS metrics
 10. correlate TLS-derived structure metrics with field LAI
 11. answer the 5 assignment questions (printed + saved to
     TLS_assignment_answers.md)

Run this after installing requirements.txt:
    python 01_tls_foliage_density_pipeline.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import tls_core as tc

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CONFIG = {
    "las_dir": ".",       # folder with Plot*_MyDiv.las files -- "." = same folder as this script
    "lai_dir": ".",       # folder with LI-COR *.TXT files
    "out_dir": "outputs/tls_results",
    "las_glob": "Plot*_MyDiv.las",             # adjust if your filenames differ
    "lai_glob": "*.TXT",

    # Only process these plot numbers even if other Plot*_MyDiv.las files
    # exist in las_dir (e.g. leftovers from other exercises) -- set to None
    # to process every file matching las_glob instead.
    "plot_numbers": [43, 58, 70],

    "epsg_code": 25833,  # UTM zone 33N (ETRS89), used by MyDiv/Leipzig; set None to skip

    # SOR filter
    "sor_k": 8,
    "sor_std_ratio": 1.0,

    # CSF ground classification
    "csf_scene": "relief",
    "csf_cloth_resolution": 1.0,
    "csf_max_iterations": 600,
    "csf_class_threshold": 0.5,

    # Downsample the point cloud to this voxel size (meters) before feeding
    # it to CSF ground classification -- CRITICAL for large real TLS scans
    # (multi-scan-position captures commonly have tens of millions of
    # points). Without this, CSF's Python binding can trigger an
    # out-of-memory "Killed" crash with no traceback (see
    # tls_core.voxel_downsample()'s docstring for why). Only the ground-
    # classification step uses the downsampled copy -- DTM/CHM/voxelization
    # still use the full-resolution point cloud. 0.02 (2 cm) is a safe
    # default that loses no meaningful ground detail; set to None to
    # disable (fine for small/already-thinned point clouds).
    "csf_downsample_voxel_m": 0.02,

    # rasterization
    "dtm_res": 0.05,
    "chm_res": 0.05,

    # voxelization / foliage metrics
    "voxel_res": 0.5,
    "layer_height": 3.0,
    "plot_width": 11.0,
    "plot_length": 11.0,
    "plot_height": 15.0,
    "rh_probs": (0.25, 0.50, 0.75, 0.98),
}

# Species-richness grouping used in the handout (adjust to your plot list)
SR_GROUPS = {
    "Plot19": "1SR", "Plot76": "1SR", "Plot68": "1SR", "Plot70": "1SR",
    "Plot35": "2SR", "Plot63": "2SR", "Plot73": "2SR", "Plot43": "2SR",
    "Plot40": "4SR", "Plot48": "4SR", "Plot74": "4SR", "Plot58": "4SR",
}


# ----------------------------------------------------------------------
# SINGLE-PLOT PROCESSING
# ----------------------------------------------------------------------

def process_tls_plot(las_path, cfg, out_dir):
    """Full processing of ONE TLS .las file. Returns a metrics dict."""
    import re as _re
    digit_groups = _re.findall(r"\d+", os.path.basename(las_path))
    plot_id = "Plot" + digit_groups[0] if digit_groups else "PlotUnknown"
    print(f"\n=== Processing {os.path.basename(las_path)} ({plot_id}) ===")

    xyz, las = tc.read_las(las_path)
    print(f"  loaded {len(xyz):,} points")

    if cfg["epsg_code"]:
        tc.assign_crs(las, cfg["epsg_code"])

    # 1. Clean noise
    xyz_clean, keep_mask = tc.sor_filter(xyz, k=cfg["sor_k"], std_ratio=cfg["sor_std_ratio"])
    print(f"  SOR filter: kept {len(xyz_clean):,} / {len(xyz):,} points")

    # 2. Ground classification
    # For very dense point clouds (e.g. multi-scan-position TLS with tens
    # of millions of points), feed CSF a downsampled COPY -- see
    # voxel_downsample()'s docstring for why this is necessary to avoid an
    # out-of-memory kill, and why it's safe (only this classification step
    # uses the downsampled copy; everything else below uses the full
    # xyz_clean point cloud).
    if cfg.get("csf_downsample_voxel_m"):
        xyz_for_csf, _ = tc.voxel_downsample(xyz_clean, cfg["csf_downsample_voxel_m"])
        print(f"  downsampled {len(xyz_clean):,} -> {len(xyz_for_csf):,} points "
              f"for CSF ground classification (voxel={cfg['csf_downsample_voxel_m']} m)")
    else:
        xyz_for_csf = xyz_clean

    ground_mask = tc.classify_ground_csf(
        xyz_for_csf, scene=cfg["csf_scene"],
        cloth_resolution=cfg["csf_cloth_resolution"],
        max_iterations=cfg["csf_max_iterations"],
        class_threshold=cfg["csf_class_threshold"],
    )
    ground_xyz = xyz_for_csf[ground_mask]
    print(f"  ground points: {ground_mask.sum():,} / {len(xyz_for_csf):,} "
          f"(from the downsampled classification pass)")

    # The TIN/Delaunay interpolator built below doesn't benefit from ground
    # points denser than the DTM's own output resolution -- thin further if
    # still very large (e.g. from a multi-scan-position TLS merge) to keep
    # Delaunay triangulation fast. This does not affect DTM accuracy: the
    # DTM is only ever evaluated at dtm_res spacing anyway.
    if len(ground_xyz) > 300_000:
        ground_xyz_for_tin, _ = tc.voxel_downsample(ground_xyz, cfg["dtm_res"])
        print(f"  thinned ground points {len(ground_xyz):,} -> "
              f"{len(ground_xyz_for_tin):,} for faster TIN construction "
              f"(voxel={cfg['dtm_res']} m, matches dtm_res)")
    else:
        ground_xyz_for_tin = ground_xyz

    # 3. DTM (TIN interpolation)
    dtm, dtm_transform, dtm_extent, ground_interp = tc.rasterize_terrain_tin(
        ground_xyz_for_tin, res=cfg["dtm_res"]
    )
    tc.save_geotiff(dtm, dtm_transform, os.path.join(out_dir, "dtm", f"{plot_id}_DTM.tif"))

    # 4. Height normalization
    xyz_norm = tc.normalize_height(xyz_clean, ground_interp)
    xyz_norm = xyz_norm[xyz_norm[:, 2] >= 0]  # drop below-ground artefacts
    print(f"  height-normalized: {len(xyz_norm):,} points (Z>=0)")

    # 5. CHM
    chm, chm_transform, chm_extent = tc.rasterize_canopy_p2r(xyz_norm, res=cfg["chm_res"])
    tc.save_geotiff(chm, chm_transform, os.path.join(out_dir, "chm", f"{plot_id}_CHM.tif"))

    # 6. Voxelization + vertical profile
    vox = tc.voxelise_point_cloud(
        xyz_norm, x_res=cfg["voxel_res"], y_res=cfg["voxel_res"],
        z_res=cfg["voxel_res"], min_z=0.0
    )
    profile = tc.build_vertical_profile(vox, layer_height=cfg["layer_height"])
    np.savetxt(
        os.path.join(out_dir, f"{plot_id}_vertical_profile.csv"),
        np.column_stack([profile["layer_id"], profile["z_low"], profile["z_mid"],
                          profile["z_high"], profile["PAI_layer"], profile["n_occupied_voxels"]]),
        delimiter=",", header="layer_id,z_low,z_mid,z_high,PAI_layer,n_occupied_voxels",
        comments="", fmt="%.4f"
    )

    fhd = tc.calc_fhd_from_profile(profile)
    fhd_std = tc.calc_fhd_from_profile(profile, standardize=True)
    rh = tc.calc_relative_heights(vox, probs=cfg["rh_probs"])
    struct = tc.summarise_tls_structure(
        vox, plot_width=cfg["plot_width"], plot_length=cfg["plot_length"],
        plot_height=cfg["plot_height"], min_z=0.0
    )

    chm_valid = chm[~np.isnan(chm)]
    metrics = {
        "plot_id": plot_id,
        "las_file": os.path.basename(las_path),
        "n_points": len(xyz),
        "n_ground_points": int(ground_mask.sum()),
        "FHD": fhd,
        "FHD_std": fhd_std,
        "chm_mean": float(chm_valid.mean()) if len(chm_valid) else np.nan,
        "chm_sd": float(chm_valid.std()) if len(chm_valid) else np.nan,
        "chm_max": float(chm_valid.max()) if len(chm_valid) else np.nan,
        **struct,
        **rh,
    }
    return metrics


def batch_process(cfg):
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "dtm"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "chm"), exist_ok=True)

    las_files = sorted(glob.glob(os.path.join(cfg["las_dir"], cfg["las_glob"])))
    if not las_files:
        print(f"! No LAS files found matching {cfg['las_glob']} in {cfg['las_dir']}")
        return []

    plot_numbers = cfg.get("plot_numbers")
    if plot_numbers:
        import re as _re
        wanted = {str(n) for n in plot_numbers}
        filtered = []
        for f in las_files:
            digits = _re.findall(r"\d+", os.path.basename(f))
            if digits and digits[0] in wanted:
                filtered.append(f)
        print(f"Filtering to plot_numbers={plot_numbers}: "
              f"{len(filtered)}/{len(las_files)} files matched")
        las_files = filtered
        if not las_files:
            print(f"! None of {os.path.join(cfg['las_dir'], cfg['las_glob'])} "
                  f"matched plot_numbers={plot_numbers} -- check filenames")
            return []

    all_metrics = []
    for f in las_files:
        try:
            all_metrics.append(process_tls_plot(f, cfg, out_dir))
        except Exception as e:
            print(f"  ! FAILED on {f}: {e}")
    return all_metrics


def save_metrics_csv(metrics_list, path):
    if not metrics_list:
        return
    keys = list(metrics_list[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for m in metrics_list:
            f.write(",".join(str(m.get(k, "")) for k in keys) + "\n")
    print(f"saved {path}")


# ----------------------------------------------------------------------
# LAI JOIN + CORRELATION
# ----------------------------------------------------------------------

def join_lai(metrics_list, cfg):
    lai_records = tc.read_all_lai(cfg["lai_dir"], pattern=cfg["lai_glob"])
    lai_by_plot = {r["plot_id"]: r["LAI"] for r in lai_records if r["plot_id"]}

    for m in metrics_list:
        m["LAI"] = lai_by_plot.get(m["plot_id"], np.nan)
        m["SR"] = SR_GROUPS.get(m["plot_id"], None)
    return metrics_list, lai_records


def correlate_and_plot(metrics_list, out_dir):
    variables = ["FHD", "max_height", "canopy_volume_per_ground_area", "chm_mean", "RH98"]
    lai = np.array([m["LAI"] for m in metrics_list], dtype=float)
    valid_lai = ~np.isnan(lai)

    print("\n--- Correlations with field LAI ---")
    results = []
    for var in variables:
        x = np.array([m.get(var, np.nan) for m in metrics_list], dtype=float)
        valid = valid_lai & ~np.isnan(x)
        if valid.sum() >= 3:
            r, p = stats.pearsonr(x[valid], lai[valid])
            print(f"  LAI vs {var}: r={r:.3f}, p={p:.4f}  (n={valid.sum()})")
            results.append((var, r, p, int(valid.sum())))

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x[valid], lai[valid])
            for i in np.where(valid)[0]:
                ax.annotate(metrics_list[i]["plot_id"], (x[i], lai[i]), fontsize=7)
            if valid.sum() >= 2:
                b, a = np.polyfit(x[valid], lai[valid], 1)
                xs = np.linspace(x[valid].min(), x[valid].max(), 20)
                ax.plot(xs, a + b * xs, "r--")
            ax.set_xlabel(var)
            ax.set_ylabel("LAI (LI-COR)")
            ax.set_title(f"LAI vs {var}  (r={r:.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"LAI_vs_{var}.png"), dpi=250)
            plt.close()
        else:
            print(f"  LAI vs {var}: not enough matched plots to correlate "
                  f"(need >=3, have {valid.sum()})")
    return results


# ----------------------------------------------------------------------
# ASSIGNMENT ANSWERS
# ----------------------------------------------------------------------

ASSIGNMENT_ANSWERS = """
# TLS Foliage-Density Assignment — Answers

These answers are written generically (they describe what to look for, and
the ecological/methodological reasoning behind each answer). Once you run
the pipeline on your real 12 MyDiv plots, plug your actual numbers into the
`{...}` placeholders — the code already prints/saves everything you need
(`tls_metrics.csv`, `LAI_data.csv`, `TLS_LAI_comparison.csv`, and the
`LAI_vs_*.png` scatter plots) to fill them in directly.

---

### 1. Which plots have the highest and lowest LAI?

Open `TLS_LAI_comparison.csv` (produced by this script) and sort by the
`LAI` column. The plot with the largest value is your highest-LAI plot; the
smallest is your lowest-LAI plot.

What to discuss:
- Check whether the highest/lowest LAI plots correspond to the highest/
  lowest species-richness (SR) group (`1SR`, `2SR`, `4SR`). In many
  biodiversity-productivity experiments (like MyDiv), higher tree species
  richness is expected to correlate with higher canopy leaf area, because
  of complementary light/space use between species (niche complementarity)
  and a higher chance of including productive species (selection effect).
  If your highest-LAI plot is NOT the richest, that's worth flagging as a
  candidate outlier — check for occlusion or a small/patchy canopy gap in
  that specific plot before concluding species richness has no effect.
- LAI values around 4-6 are typical for closed temperate broadleaf canopies;
  values <2 usually indicate an open canopy, gaps, or young/sparse trees;
  values >6 suggest a very dense, multi-layered canopy.

### 2. Which TLS-derived structural metric is most strongly related to
    field-measured LAI: canopy volume per ground area, FHD, mean canopy
    height, or upper canopy height (RH98)?

Look at the printed Pearson correlation coefficients (`r`) from
`correlate_and_plot()` / the `LAI_vs_*.png` figures. In general, expect:

- `canopy_volume_per_ground_area` (occupied 3D space normalized by plot
  footprint) usually correlates best with LAI, because both quantities are
  fundamentally about "how much leaf/branch material is packed into the
  canopy volume" — LAI is a 2D projection of leaf area, and occupied voxel
  volume is a 3D proxy for the same underlying biomass/foliage quantity.
- `chm_mean` (mean canopy height) and `RH98` (upper canopy height) capture
  canopy STATURE (how tall the trees are), not how DENSE the foliage is.
  Two plots can have identical height but very different LAI (e.g. one
  thinned, one closed) — so height metrics usually correlate more weakly
  with LAI than volume/density metrics do.
- `FHD` (foliage height diversity) captures the SHAPE of the vertical
  profile (evenness across layers), not the total amount of foliage, so it
  often has the weakest (or even near-zero / non-significant) correlation
  with LAI in a small sample. FHD is more relevant for questions about
  structural complexity/habitat diversity than total leaf area.

State explicitly which variable had the highest |r| in YOUR run, and report
its r and p-value from the console output.

### 3. Does a higher LAI necessarily mean a more vertically complex canopy?

No. LAI is a measure of total leaf AMOUNT (leaf area per ground area),
integrated across the whole canopy column — it says nothing about HOW that
leaf area is distributed vertically. A single-layer, very dense canopy
(e.g. a monoculture plantation with a thick, uniform crown layer) can have
a high LAI but LOW foliage height diversity (FHD), because nearly all the
leaf area sits in one narrow height band. Conversely, a canopy with several
distinct layers (understory, mid-story, overstory) — each individually
sparse — can have moderate LAI but HIGH FHD, because the (smaller) leaf area
is spread evenly across many height strata. This is exactly why FHD and LAI
are complementary, not redundant, metrics: LAI ~ "how much", FHD ~ "how it's
arranged in 3D". A meaningful ecological interpretation therefore requires
looking at BOTH the total (LAI, canopy volume) and the vertical distribution
(FHD, the vertical profile plot) together, not just one number.

### 4. Which sources of uncertainty may affect the comparison between
    TLS-derived foliage density and LI-COR LAI?

- **Different physical principles**: LI-COR LAI-2200 estimates LAI from
  transmitted diffuse light (gap fraction inversion using Miller's theorem)
  — it is sensitive to canopy gap distribution and assumes a particular
  leaf angle distribution model. TLS-derived metrics count laser hits in
  3D space — a completely different physical measurement principle, so
  perfect 1:1 agreement is not expected even under ideal conditions.
- **Occlusion** (see Q5): TLS from the ground cannot "see" every leaf,
  especially those hidden behind other foliage/branches from every scan
  position — this systematically underestimates upper-canopy foliage
  density in a way LI-COR's optical inversion does not.
- **Footprint mismatch**: The LI-COR sensor typically measures a fisheye
  hemispherical footprint from one or a few positions, while the TLS
  point cloud covers the whole plot area but with radially decreasing
  point density away from the scanner. If the LI-COR measurement point(s)
  and the "densest" part of the TLS scan aren't co-located, you're not
  really comparing the same physical patch of canopy.
- **Woody material**: LAI, by definition, is LEAF area only. TLS-derived
  voxel occupancy/volume includes hits on branches, trunks, and twigs too
  (unless a wood-leaf separation/classification step is added), inflating
  the TLS "foliage density" proxy relative to true leaf area, especially in
  species-rich plots with more woody structure per unit ground area.
  Similarly, PAI (Plant Area Index, what our voxel-based PAI approximates)
  is not the same quantity as LAI — PAI = LAI + woody area index.
- **Timing/phenology**: if the TLS scan and the LI-COR measurement were not
  taken on the same day, seasonal leaf flush/senescence differences will
  add noise unrelated to any real structural relationship.
- **Small sample size**: with only ~12 plots, a single unusual plot (e.g.
  affected by occlusion or a local canopy gap) can swing the correlation
  substantially — always inspect a scatterplot, not just the r/p-value.

### 5. How could occlusion affect foliage-density estimates?

TLS scans from a fixed tripod position (or a few positions) on the ground.
Any leaf, branch, or trunk segment that lies directly behind another object
relative to every scan position is never hit by the laser — it is
"occluded" and simply missing from the point cloud, NOT recorded as "no
foliage there." This has systematic (not random) effects:

- **Underestimation increases with height and distance from the scanner**:
  the upper canopy and the far edges of the plot are typically the most
  occluded, because more foliage/branches from lower/closer parts of the
  canopy sit between the scanner and them. This means voxel occupancy, PAI,
  and total canopy volume are usually UNDERESTIMATED, especially at the top
  of the canopy — biasing metrics like `RH98`/`chm_max` less (since they
  only need ONE hit near the top) but biasing volume/occupancy-based
  metrics more (since they need MANY hits to register a fully "occupied"
  space).
  This is exactly the pattern you should see in the UAV vs. TLS CHM
  comparison exercise: UAV-LiDAR (scanning from above) sees the true canopy
  envelope well, while TLS (scanning from below) shows characteristic
  "shadow wedges" of missing/underestimated canopy behind dense foliage or
  trunks, radiating outward from the scan position(s).
- **Multiple scan positions reduce, but never eliminate, occlusion** — this
  is precisely why multi-scan TLS surveys (registering scans from several
  positions around the plot) are preferred over single-scan TLS when
  occlusion matters for the analysis.
- **Practical implication for LAI comparison**: because occlusion mostly
  hides upper/interior foliage, TLS-derived foliage density likely
  UNDERESTIMATES true LAI on average, and this bias will vary between plots
  depending on stand density and structure — adding scatter/noise to any
  TLS-vs-LAI correlation, independent of the real ecological relationship
  between species richness and leaf area.
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

    metrics_list = batch_process(cfg)
    save_metrics_csv(metrics_list, os.path.join(cfg["out_dir"], "tls_metrics.csv"))

    metrics_list, lai_records = join_lai(metrics_list, cfg)
    save_metrics_csv(
        [{"file": r["file"], "plot_id": r["plot_id"], "LAI": r["LAI"]} for r in lai_records],
        os.path.join(cfg["out_dir"], "LAI_data.csv"),
    )
    save_metrics_csv(metrics_list, os.path.join(cfg["out_dir"], "TLS_LAI_comparison.csv"))

    correlate_and_plot(metrics_list, cfg["out_dir"])

    save_answers(os.path.join(cfg["out_dir"], "TLS_assignment_answers.md"))

    print("\nDone. See:", cfg["out_dir"])


if __name__ == "__main__":
    main()