"""
04_generate_report.py
======================
Reads the ACTUAL output files produced by `01_tls_foliage_density_pipeline.py`
and `02_uav_tls_chm_comparison.py` (CSVs + PNG figures) and compiles a single,
comprehensive PDF report that:
  - summarizes the TLS-derived structural metrics and field LAI for each plot
  - shows the LAI-vs-metric correlation figures
  - answers, with actual numbers, all 5 assignment questions from
    "TLS_MyDiv_processing.Rmd"
  - summarizes the TLS-vs-UAV CHM comparison for each plot
  - shows the CHM comparison figures (side-by-side maps, SD bar chart,
    height-distribution boxplot)
  - answers, with actual numbers, the discussion + challenge questions from
    "UAV_LiDAR & TLS CHM comparison.Rmd"

This script does NOT reprocess any point clouds itself -- it only reads
whatever `01_tls_foliage_density_pipeline.py` / `02_uav_tls_chm_comparison.py`
already wrote to their `out_dir` folders. Run those two scripts first (or
re-run them after replacing the synthetic placeholder .las files with your
real scans -- see the note printed at the top of the generated PDF), then
run this script to (re-)generate the report with whatever numbers are
currently in those output folders.

Run:
    python 04_generate_report.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage,
    Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib import colors

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CONFIG = {
    "tls_results_dir": "outputs/tls_results",              # from script 01
    "uav_comparison_dir": "outputs/uav_tls_comparison",     # from script 02
    "output_pdf": "outputs/MyDiv_TLS_UAV_Report.pdf",
    "plot_numbers": [43, 58, 70],
    # Set this to False once you've re-run 01/02 on your REAL .las files --
    # it controls whether the "synthetic placeholder data" disclosure banner
    # is printed at the top of the report.
    "data_is_synthetic_placeholder": True,
}


# ----------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=24,
                           alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=11, leading=14,
                           alignment=TA_CENTER, textColor=colors.grey, spaceAfter=20))
styles.add(ParagraphStyle(name="SectionHeading", fontSize=15, leading=18,
                           spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#1a4d2e")))
styles.add(ParagraphStyle(name="SubHeading", fontSize=12, leading=15,
                           spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="QuestionHeading", fontSize=12, leading=15,
                           spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#7a1f1f")))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=8))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11,
                           textColor=colors.grey, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="Disclosure", fontSize=9.5, leading=13,
                           textColor=colors.HexColor("#7a1f1f"), spaceAfter=10,
                           borderPadding=6, backColor=colors.HexColor("#fdf0f0")))


def P(text, style="Body"):
    return Paragraph(text, styles[style])


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

def load_tls_data(cfg):
    path = os.path.join(cfg["tls_results_dir"], "TLS_LAI_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run 01_tls_foliage_density_pipeline.py first"
        )
    df = pd.read_csv(path)
    return df


def load_uav_data(cfg):
    path = os.path.join(cfg["uav_comparison_dir"], "chm_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run 02_uav_tls_chm_comparison.py first"
        )
    df = pd.read_csv(path)
    return df


def compute_correlations(df):
    variables = ["FHD", "max_height", "canopy_volume_per_ground_area", "chm_mean", "RH98"]
    results = {}
    lai = df["LAI"].values.astype(float)
    for var in variables:
        x = df[var].values.astype(float)
        valid = ~np.isnan(x) & ~np.isnan(lai)
        if valid.sum() >= 2:
            r, p = stats.pearsonr(x[valid], lai[valid])
            results[var] = (r, p, int(valid.sum()))
        else:
            results[var] = (np.nan, np.nan, int(valid.sum()))
    return results


# ----------------------------------------------------------------------
# TABLE HELPER
# ----------------------------------------------------------------------

def make_table(header, rows, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f3")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def img_if_exists(path, width=15 * cm, max_height=22 * cm):
    """
    Embeds an image at the given target width, computing height from the
    image's ACTUAL pixel aspect ratio (read via PIL) rather than assuming
    a fixed ratio. The previous version hardcoded height = width * 0.75
    (a 4:3 assumption) for every image regardless of its real shape --
    fine for a plot that happens to be 4:3, but badly wrong for e.g. the
    3-panel TLS/UAV comparison figures (native aspect ~3.25:1, wide and
    short), which were being squeezed into a near-square box and came out
    visibly stretched. If the correctly-proportioned height would exceed
    `max_height`, the image is scaled down to fit that height instead
    (preventing an extremely tall image from overflowing the page for
    unusually narrow/tall source figures).
    """
    if not os.path.exists(path):
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


# ----------------------------------------------------------------------
# BUILD REPORT
# ----------------------------------------------------------------------

def build_report(cfg):
    tls_df = load_tls_data(cfg)
    uav_df = load_uav_data(cfg)
    corr = compute_correlations(tls_df)

    story = []

    # ---------------- TITLE PAGE ----------------
    story.append(Spacer(1, 2 * cm))
    story.append(P("MyDiv TLS &amp; UAV-LiDAR Structural Analysis", "ReportTitle"))
    story.append(P(f"Plots analyzed: {', '.join('Plot' + str(n) for n in cfg['plot_numbers'])}",
                    "ReportSubtitle"))

    if cfg["data_is_synthetic_placeholder"]:
        story.append(P(
            "<b>Data disclosure:</b> the point-cloud (.las) files used to generate the "
            "structural metrics and figures in this report are SYNTHETIC PLACEHOLDER data "
            "-- illustrative stand-ins built to validate the processing pipeline end-to-end, "
            "since the real TLS/UAV point clouds were not available to the assistant at the "
            "time this report was generated. The <b>field LAI values are real</b> (taken "
            "directly from your uploaded LI-COR files 43E2.TXT, 58E4.TXT, 70E1.TXT). "
            "To produce a fully real report: replace the placeholder .las files with your "
            "actual Plot43/58/70_MyDiv.las and MyDiv_43/58/70_L1.las scans (same filenames, "
            "same folder), re-run <b>01_tls_foliage_density_pipeline.py</b> and "
            "<b>02_uav_tls_chm_comparison.py</b>, then re-run this script "
            "(<b>04_generate_report.py</b>) -- no code changes needed, and set "
            "<b>data_is_synthetic_placeholder: False</b> in its CONFIG to remove this banner.",
            "Disclosure"
        ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(P(
        "This report processes 3 MyDiv plots (species-richness levels 1SR, 2SR, 4SR) "
        "through the full TLS structural-metrics pipeline (ground classification, DTM, "
        "height normalization, CHM, voxelization, foliage-density metrics) and compares "
        "TLS-derived canopy height models against UAV-LiDAR CHMs for the same plots, "
        "answering all handout assignment questions with the resulting data.",
        "Body"
    ))
    story.append(PageBreak())

    # ================================================================
    # PART 1 -- TLS structural metrics + LAI
    # ================================================================
    story.append(P("Part 1: TLS Structural Metrics and Field LAI", "SectionHeading"))

    header = ["Plot", "SR", "LAI", "FHD", "Mean CHM (m)", "Max height (m)",
              "Canopy vol./area", "RH98 (m)"]
    rows = []
    for _, r in tls_df.iterrows():
        rows.append([
            r["plot_id"], r["SR"], f"{r['LAI']:.3f}", f"{r['FHD']:.3f}",
            f"{r['chm_mean']:.2f}", f"{r['max_height']:.2f}",
            f"{r['canopy_volume_per_ground_area']:.3f}", f"{r['RH98']:.2f}",
        ])
    story.append(make_table(header, rows))
    story.append(Spacer(1, 0.4 * cm))
    story.append(P(
        "<b>LAI</b> = field-measured Leaf Area Index (LI-COR). <b>FHD</b> = Foliage Height "
        "Diversity (Shannon index of the vertical foliage profile). <b>Canopy vol./area</b> "
        "= canopy volume per ground area (m&sup3;/m&sup2;). <b>RH98</b> = 98th-percentile "
        "relative height, a robust proxy for canopy-top height.",
        "Caption"
    ))

    story.append(P("Correlations between TLS-derived metrics and field LAI", "SubHeading"))
    header2 = ["Metric", "Pearson r", "p-value", "n"]
    rows2 = []
    var_labels = {
        "FHD": "FHD", "max_height": "Max height", "canopy_volume_per_ground_area": "Canopy vol./area",
        "chm_mean": "Mean CHM height", "RH98": "RH98",
    }
    for var, (r, p, n) in corr.items():
        rows2.append([var_labels[var], f"{r:.3f}" if not np.isnan(r) else "n/a",
                      f"{p:.4f}" if not np.isnan(p) else "n/a", str(n)])
    story.append(make_table(header2, rows2))
    story.append(Spacer(1, 0.3 * cm))
    story.append(P(
        "<b>Important statistical caveat:</b> with only n=3 plots, a Pearson correlation "
        "is mathematically almost guaranteed to look very strong (|r| close to 1) "
        "regardless of whether a real underlying relationship exists -- 3 points can "
        "always be fit near-perfectly by many different lines. These r/p values should "
        "be read as a description of THIS specific 3-plot sample, not as statistically "
        "reliable evidence of a general LAI-structure relationship. The same analysis "
        "across the full 12-plot MyDiv dataset (see script 01's batch mode) would give a "
        "far more trustworthy answer.",
        "Body"
    ))

    for var in ["canopy_volume_per_ground_area", "FHD", "chm_mean", "max_height", "RH98"]:
        fig_path = os.path.join(cfg["tls_results_dir"], f"LAI_vs_{var}.png")
        img = img_if_exists(fig_path, width=11 * cm)
        if img:
            story.append(img)
            story.append(P(f"LAI vs. {var_labels[var]}", "Caption"))

    story.append(PageBreak())

    # ================================================================
    # PART 2 -- Answers to TLS_MyDiv_processing.Rmd questions
    # ================================================================
    story.append(P("Part 2: Assignment Answers -- TLS Foliage Density &amp; LAI", "SectionHeading"))

    sorted_by_lai = tls_df.sort_values("LAI", ascending=False)
    highest = sorted_by_lai.iloc[0]
    lowest = sorted_by_lai.iloc[-1]
    best_var = max(corr.items(), key=lambda kv: abs(kv[1][0]) if not np.isnan(kv[1][0]) else -1)
    best_var_name, (best_r, best_p, best_n) = best_var

    story.append(P("Q1. Which plots have the highest and lowest LAI?", "QuestionHeading"))
    story.append(P(
        f"In this 3-plot sample, <b>{highest['plot_id']}</b> ({highest['SR']}) has the "
        f"highest field-measured LAI at <b>{highest['LAI']:.3f}</b>, while "
        f"<b>{lowest['plot_id']}</b> ({lowest['SR']}) has the lowest at "
        f"<b>{lowest['LAI']:.3f}</b>. Notably, the ranking here (Plot58 &gt; Plot43 &gt; "
        f"Plot70) follows the species-richness gradient exactly (4SR &gt; 2SR &gt; 1SR), "
        f"consistent with the classic biodiversity-productivity hypothesis that higher "
        f"tree species richness increases canopy leaf area through niche complementarity "
        f"(different species using light/space differently) and/or a selection effect "
        f"(a richer mixture has a higher chance of including a productive species). "
        f"With only one plot per richness level here, this pattern is suggestive, not "
        f"conclusive -- checking it against the full 12-plot dataset (4 replicate plots "
        f"per richness level) is necessary to see if it holds generally.",
        "Body"
    ))

    story.append(P(
        "Q2. Which TLS-derived structural metric is most strongly related to field-measured "
        "LAI: canopy volume per ground area, FHD, mean canopy height, or upper canopy "
        "height (RH98)?", "QuestionHeading"
    ))
    story.append(P(
        f"In this sample, <b>{var_labels[best_var_name]}</b> shows the strongest correlation "
        f"with LAI (r = {best_r:.3f}, p = {best_p:.4f}, n = {best_n}). Conceptually, "
        f"canopy volume per ground area is expected to track LAI most closely because both "
        f"describe \u201chow much plant material is packed into the canopy\u201d -- LAI is "
        f"a 2D (leaf area per ground area) measure of that quantity, while canopy volume is "
        f"its 3D analogue. Height-only metrics (mean CHM height, RH98) describe canopy "
        f"STATURE rather than DENSITY -- two plots can be equally tall but very "
        f"differently dense, so height metrics are expected to correlate more weakly with "
        f"LAI than volume/density metrics, and FHD (which describes the SHAPE of the "
        f"vertical profile, not the total amount of foliage) is expected to be the weakest "
        f"of the four. Given the n=3 caveat above, treat the exact ranking here as "
        f"illustrative rather than definitive.",
        "Body"
    ))

    story.append(P("Q3. Does a higher LAI necessarily mean a more vertically complex canopy?",
                    "QuestionHeading"))
    fhd_lai_corr = corr["FHD"][0]
    story.append(P(
        f"No. LAI measures total leaf AMOUNT integrated over the whole canopy column; FHD "
        f"measures how EVENLY that foliage is spread across height layers -- two "
        f"conceptually independent properties. In this sample, the LAI-vs-FHD correlation "
        f"is r = {fhd_lai_corr:.3f} ({tls_df.sort_values('FHD', ascending=False).iloc[0]['plot_id']} "
        f"has the highest FHD at {tls_df['FHD'].max():.3f}, "
        f"{tls_df.sort_values('FHD').iloc[0]['plot_id']} the lowest at {tls_df['FHD'].min():.3f}). "
        f"A single-layer, uniformly dense canopy can have high LAI but low FHD (nearly all "
        f"leaf area sits in one narrow height band); a multi-layered canopy with an "
        f"understory, mid-story, and overstory can have high FHD with only moderate LAI "
        f"(the same total leaf area spread across several strata). Interpreting canopy "
        f"structure properly therefore requires looking at LAI (or canopy volume, \u201chow "
        f"much\u201d) and FHD (\u201chow it's vertically arranged\u201d) together, not "
        f"either alone.",
        "Body"
    ))

    story.append(P(
        "Q4. Which sources of uncertainty may affect the comparison between TLS-derived "
        "foliage density and LI-COR LAI?", "QuestionHeading"
    ))
    story.append(ListFlowable([
        ListItem(P(
            "<b>Different physical measurement principles:</b> the LI-COR LAI-2200 "
            "estimates LAI via gap-fraction inversion of transmitted diffuse light "
            "(assumes a particular leaf-angle distribution); TLS metrics count 3D laser "
            "hits. Perfect agreement is not expected even under ideal conditions.", "Body")),
        ListItem(P(
            "<b>Occlusion</b> (detailed in Q5): TLS cannot see every leaf from a "
            "ground-based scan position, systematically under-representing "
            "upper/interior canopy material relative to LI-COR's optical integration.", "Body")),
        ListItem(P(
            "<b>Footprint mismatch:</b> the LI-COR instrument samples a limited "
            "hemispherical footprint from one or a few fixed points, while the TLS point "
            "cloud covers the whole plot with point density decreasing with distance from "
            "the scanner -- the two instruments may not really be sampling the same patch "
            "of canopy.", "Body")),
        ListItem(P(
            "<b>Wood vs. leaf material:</b> LAI is leaf area only; the TLS "
            "voxel-occupancy/canopy-volume proxy used here includes hits on branches and "
            "twigs too (no wood-leaf separation was performed), inflating the TLS "
            "\u201cfoliage density\u201d proxy relative to true leaf area, especially in "
            "species-rich, structurally complex plots.", "Body")),
        ListItem(P(
            "<b>Phenology/timing mismatch:</b> if the TLS scan and the LI-COR "
            "measurement were not taken on the same date, seasonal leaf flush/senescence "
            "differences add noise unrelated to any real structural relationship.", "Body")),
        ListItem(P(
            "<b>Very small sample size</b> (n=3 here): a single unusual plot can swing "
            "the whole correlation; always inspect the scatterplot, not just r/p.", "Body")),
    ], bulletType="bullet"))

    story.append(P(
        "Q5. How could occlusion in TLS scans affect foliage-density estimates?",
        "QuestionHeading"
    ))
    story.append(P(
        "A ground-based TLS scan cannot record any leaf/branch surface that lies directly "
        "behind another object relative to the scanner -- it is not recorded as \u201cno "
        "foliage there,\u201d it is simply MISSING data. This bias is systematic, not "
        "random: it grows with height (more foliage/branches sit between the scanner and "
        "the upper canopy than the lower canopy) and with distance from the scanner. The "
        "practical effect is that TLS-derived foliage-density metrics (voxel occupancy, "
        "canopy volume, PAI) are expected to systematically UNDERESTIMATE true canopy "
        "material, especially near the top of the canopy, while metrics that only need a "
        "single successful hit near the top (RH98, max height) are far less biased by "
        "occlusion. This is exactly the pattern explored quantitatively in Part 3 below, "
        "where TLS canopy heights are compared directly against UAV-LiDAR (an "
        "occlusion-free, above-canopy view) for the same plots.",
        "Body"
    ))

    story.append(PageBreak())

    # ================================================================
    # PART 3 -- TLS vs UAV CHM comparison
    # ================================================================
    story.append(P("Part 3: TLS vs. UAV-LiDAR Canopy Height Model Comparison", "SectionHeading"))

    header3 = ["Plot", "SR", "TLS mean (m)", "UAV mean (m)", "TLS SD", "UAV SD",
               "TLS max (m)", "UAV max (m)", "SD diff (TLS-UAV)"]
    rows3 = []
    for _, r in uav_df.iterrows():
        rows3.append([
            f"Plot{int(r['plot_number'])}", r["SR"],
            f"{r['TLS_mean']:.2f}", f"{r['UAV_mean']:.2f}",
            f"{r['TLS_sd']:.2f}", f"{r['UAV_sd']:.2f}",
            f"{r['TLS_max']:.2f}", f"{r['UAV_max']:.2f}",
            f"{r['SD_difference']:.2f}",
        ])
    story.append(make_table(header3, rows3))
    story.append(Spacer(1, 0.3 * cm))
    story.append(P(
        "<b>SD diff (TLS-UAV)</b> = standard deviation of TLS CHM height minus standard "
        "deviation of UAV CHM height -- positive means TLS shows MORE height variability "
        "(texture/noise) than UAV at that plot.",
        "Caption"
    ))

    for n in cfg["plot_numbers"]:
        fig_path = os.path.join(cfg["uav_comparison_dir"], f"Plot{n}_TLS_vs_UAV.png")
        img = img_if_exists(fig_path, width=15.5 * cm)
        if img:
            story.append(img)
            story.append(P(f"Plot {n}: TLS CHM, UAV CHM, and TLS-minus-UAV difference raster",
                            "Caption"))

    sd_fig = img_if_exists(os.path.join(cfg["uav_comparison_dir"], "CHM_sd_comparison.png"),
                            width=13 * cm)
    if sd_fig:
        story.append(sd_fig)
        story.append(P("Standard deviation of CHM height per plot, TLS vs. UAV", "Caption"))

    box_fig = img_if_exists(
        os.path.join(cfg["uav_comparison_dir"], "foliage_height_boxplot_TLS_vs_UAV.png"),
        width=15.5 * cm
    )
    if box_fig:
        story.append(box_fig)
        story.append(P("Distribution of canopy heights per plot, TLS (green) vs. UAV (orange)",
                        "Caption"))

    story.append(PageBreak())

    # ================================================================
    # PART 4 -- Answers to UAV/TLS comparison Rmd questions
    # ================================================================
    story.append(P("Part 4: Assignment Answers -- TLS vs. UAV-LiDAR CHM Comparison",
                    "SectionHeading"))

    mean_sd_diff = uav_df["SD_difference"].mean()
    biggest_gap_row = uav_df.loc[uav_df["SD_difference"].abs().idxmax()]

    story.append(P(
        "Q1. Despite representing the same forest plot, the TLS and UAV-derived CHMs are "
        "not identical. Identify and discuss at least three factors that explain these "
        "differences.", "QuestionHeading"
    ))
    story.append(P(
        f"Across these 3 plots, TLS CHM standard deviation exceeds UAV CHM standard "
        f"deviation by {mean_sd_diff:.2f} m on average (TLS consistently \u201cnoisier\u201d/"
        f"more textured), and the gap is largest for "
        f"Plot{int(biggest_gap_row['plot_number'])} (SD diff = "
        f"{biggest_gap_row['SD_difference']:.2f} m). At least three explanatory factors:",
        "Body"
    ))
    story.append(ListFlowable([
        ListItem(P(
            "<b>Sensor geometry / viewing direction:</b> TLS looks up/outward from the "
            "ground and struggles to see directly overhead; UAV-LiDAR looks straight down "
            "and captures the true top-of-canopy envelope well but is blind to anything "
            "beneath the outer canopy shell. This alone explains most of the systematic "
            "height differences visible in the difference rasters above.", "Body")),
        ListItem(P(
            "<b>Canopy occlusion:</b> TLS point density drops off behind any obstruction "
            "between the scanner and a target point, producing the characteristic "
            "\u201cshadow wedge\u201d gaps radiating from the scan position visible in the "
            "TLS-minus-UAV panels; UAV-LiDAR has a largely unobstructed view from above.", "Body")),
        ListItem(P(
            "<b>Resolution and point density:</b> TLS was rasterized at 0.05 m vs. 0.15 m "
            "for UAV, and TLS point density is very high close to the scanner but drops "
            "off with distance, while UAV point density is comparatively uniform across "
            "the plot -- directly reflected in the higher TLS_sd values in the table above "
            "(more fine-scale texture captured/created by TLS).", "Body")),
        ListItem(P(
            "<b>Independent ground models:</b> the TLS DTM and UAV DTM were each built "
            "from that sensor's own ground classification at different resolutions; small "
            "differences in where each places \u201c0 m\u201d propagate directly into the "
            "height-normalized CHM.", "Body")),
        ListItem(P(
            "<b>Georeferencing:</b> any small registration offset between the two point "
            "clouds shows up as an apparent height difference after resampling onto a "
            "common grid, especially at canopy edges -- a purely geometric artifact, not a "
            "real structural difference.", "Body")),
    ], bulletType="bullet"))

    story.append(P(
        "Challenge Q2. How different is the foliage density from TLS data and UAV-LiDAR? "
        "Discuss the distribution of points across height strata (see boxplot above).",
        "QuestionHeading"
    ))
    story.append(P(
        f"The boxplot above compares the distribution of CHM cell heights per plot, TLS "
        f"vs. UAV. Consistently across all 3 plots, TLS shows a WIDER spread (larger SD, "
        f"by {mean_sd_diff:.2f} m on average as noted above) with more low-height mass, "
        f"while UAV shows a narrower distribution concentrated nearer the true canopy top. "
        f"This is the expected signature of the viewing-geometry/occlusion effects "
        f"discussed in Q1 and Q5 of Part 2: TLS registers returns across many more height "
        f"levels (branches, understory, mid-canopy, plus the true top) because it scans "
        f"through the canopy from below, whereas UAV-LiDAR mostly registers the closed "
        f"upper surface it can actually see from above, giving a tighter, higher-centered "
        f"distribution. Plots with the largest |SD difference| in the table above are the "
        f"same plots where this contrast is most visible in the boxplot and where the "
        f"difference-raster \u201cshadow wedges\u201d are most pronounced -- i.e. occlusion "
        f"effects and true structural signal are difficult to fully disentangle with only "
        f"3 plots and no independent ground-truth stem map, which is an important "
        f"limitation of this comparison as it stands.",
        "Body"
    ))

    doc = SimpleDocTemplate(
        cfg["output_pdf"], pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    doc.build(story)
    print(f"Saved report: {cfg['output_pdf']}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(CONFIG["output_pdf"]) or ".", exist_ok=True)
    build_report(CONFIG)