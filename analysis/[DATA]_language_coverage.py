"""
Language coverage analysis for the parallel-text datasets used by OT-MOE:
`data/processed_alignment/{bible,flores,ntrex}.json`.

Data format
-----------
Each file is a JSON list of records. Every record is a dict that has some
non-language metadata (e.g. "id") plus one key per language, using an
NLLB-style `xxx_Yyyy` code (language subtag + script subtag), e.g.:

    {
      "id": 22873,
      "eng_Latn": "Then Herod, ...",
      "vie_Latn": "Vua He-rot thay minh ...",
      "zho_Hani": "...",
      ...
    }

A language is only counted as "present" in a record if its value is a
non-empty string (a missing key or an empty string both count as "not
covered" for that record). Language keys are auto-detected with a regex,
so the script does not need a hardcoded language list and works even if
bible.json / flores.json / ntrex.json don't share the exact same set of
languages or use a handful of non-standard codes (e.g. some Bible-corpus
codes like "jap_Hira" or "ojb_Cans" that don't match the usual NLLB codes).

What this script does
----------------------
1. Loads all 3 datasets and, for each, computes the *set* of languages that
   have at least one non-empty sentence somewhere in the file.
2. Computes the union of those 3 sets -> the total number of distinct
   languages covered across all datasets combined.
3. Draws exactly ONE chart -- a 3-set Venn diagram -- showing how the
   language sets of Bible / FLORES / NTREX overlap. Since FLORES has ~200
   languages and NTREX has ~128 (per their public releases) and Bible-corpus
   language lists are commonly in the same order of magnitude, listing every
   language on the chart isn't readable, so instead the chart carries a
   legend of the ~10 most widely-spoken languages (English, Chinese, Hindi,
   Spanish, ...), color-coded to match the Venn region they fall into.
4. Saves supporting artifacts (not additional chart types) to --output_dir:
   - language_coverage_venn.png   the chart itself
   - language_coverage_matrix.csv every language x which dataset(s) have it
   - language_coverage_summary.json  overall totals

Usage
-----
    python "[DATA]_language_coverage.py" \
        --data_root data/processed_alignment \
        --output_dir analysis/output

Optional dependency
--------------------
For a nicer, properly-proportioned Venn diagram, install matplotlib-venn:
    pip install matplotlib-venn
If it isn't installed, the script automatically falls back to a schematic
(non-proportional, but numerically correct) 3-circle Venn diagram drawn
with plain matplotlib, so it still works out of the box.
"""

import argparse
import csv
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from matplotlib_venn import venn3, venn3_circles

    HAS_VENN_LIB = True
except ImportError:
    HAS_VENN_LIB = False


# Times New Roman if available on the machine running this script;
# otherwise matplotlib falls back to the next available serif font
# (e.g. Liberation Serif, which is metrically compatible with Times New
# Roman) without erroring out.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

# One color per Venn region, shared between the diagram itself and the
# "top languages" legend, so a legend swatch's color always matches the
# region that language actually falls into.
REGION_COLORS = {
    "100": "#4C72B0",  # Bible only
    "010": "#DD8452",  # FLORES only
    "001": "#55A868",  # NTREX only
    "110": "#9878B5",  # Bible & FLORES only
    "101": "#7F9E76",  # Bible & NTREX only
    "011": "#C58F6B",  # FLORES & NTREX only
    "111": "#8C8C8C",  # all 3
    "000": "#E0E0E0",  # not found in any of the 3 (edge case)
}


# Matches NLLB-style language codes used in these datasets, e.g.
# "eng_Latn", "zho_Hani", "ojb_Cans", "jap_Hira". Any dict key that does
# NOT match this (like "id") is treated as metadata, not a language.
LANG_CODE_RE = re.compile(r"^[a-z]{2,4}_[A-Z][a-z]{3}$")

DATASETS = OrderedDict(
    [
        ("Bible", "bible.json"),
        ("FLORES", "flores.json"),
        ("NTREX", "ntrex.json"),
    ]
)

# ~10 widely-spoken languages used to annotate the plot, since the full
# language lists (roughly 128-200+ per dataset) are too long to label
# individually. Each entry lists the language-subtag prefixes (the part
# before "_") that should count as a match, to tolerate small naming
# differences between datasets (e.g. "jpn" vs "jap" for Japanese).
POPULAR_LANGUAGES = [
    ("English", ["eng"]),
    ("Chinese", ["zho", "cmn"]),
    ("Hindi", ["hin"]),
    ("Spanish", ["spa"]),
    ("French", ["fra"]),
    ("Arabic", ["arb", "ara"]),
    ("Portuguese", ["por"]),
    ("Russian", ["rus"]),
    ("Japanese", ["jpn", "jap"]),
    ("German", ["deu", "ger"]),
]


def load_records(path: Path):
    """Load a dataset file into a flat list of record dicts. Tolerates a
    plain list (the expected shape) as well as a couple of common wrapped
    shapes, so the script doesn't break on minor format differences."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    raise ValueError(f"Unrecognized JSON structure in {path}")


def analyze_dataset(path: Path):
    """Scan one dataset file and return:
    - lang_set: set of language codes with >=1 non-empty sentence anywhere
      in the file
    - lang_counts: Counter(lang_code -> #records with a non-empty sentence
      for that language) -- how *complete* the coverage is, not just
      present/absent
    - n_records: total number of records in the file
    """
    records = load_records(path)
    lang_counts = Counter()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key, value in rec.items():
            if not LANG_CODE_RE.match(key):
                continue
            if isinstance(value, str) and value.strip():
                lang_counts[key] += 1
    return set(lang_counts.keys()), lang_counts, len(records)


def find_popular_code(lang_set, prefixes):
    """Return the actual language code in `lang_set` matching one of
    `prefixes` (e.g. "eng" -> "eng_Latn"), or None if not present."""
    for code in sorted(lang_set):
        if code.split("_")[0] in prefixes:
            return code
    return None


def draw_schematic_venn(ax, names, set_a, set_b, set_c):
    """Fallback 3-circle Venn used only when matplotlib-venn isn't
    installed. The circles are drawn at a fixed, equal size purely for
    layout -- they are NOT area-proportional to the real set sizes -- but
    every count shown is always the real, correctly-computed set size."""
    print(
        "[WARN] matplotlib-venn not installed -- drawing a schematic "
        "(non-proportional) Venn diagram instead. For nicer, "
        "proportionally-sized circles run: pip install matplotlib-venn"
    )

    centers = {"A": (-0.65, 0.55), "B": (0.65, 0.55), "C": (0.0, -0.55)}
    radius = 1.35

    for key, (cx, cy) in centers.items():
        color = REGION_COLORS[{"A": "100", "B": "010", "C": "001"}[key]]
        ax.add_patch(mpatches.Circle((cx, cy), radius, alpha=0.45, color=color, ec="gray", lw=1.4))

    label_pos = {"A": (-1.55, 1.75), "B": (1.55, 1.75), "C": (0, -2.15)}
    for key, name, s in zip(["A", "B", "C"], names, [set_a, set_b, set_c]):
        cx, cy = label_pos[key]
        ax.text(cx, cy, f"{name}\n(n={len(s)} languages)", ha="center", va="center", fontsize=18, fontweight="bold")

    region_vals = {
        "only_a": set_a - set_b - set_c,
        "only_b": set_b - set_a - set_c,
        "only_c": set_c - set_a - set_b,
        "ab_only": (set_a & set_b) - set_c,
        "ac_only": (set_a & set_c) - set_b,
        "bc_only": (set_b & set_c) - set_a,
        "abc": set_a & set_b & set_c,
    }
    region_pos = {
        "only_a": (-1.15, 0.95),
        "only_b": (1.15, 0.95),
        "only_c": (0, -1.35),
        "ab_only": (0, 1.15),
        "ac_only": (-0.65, -0.15),
        "bc_only": (0.65, -0.15),
        "abc": (0, 0.35),
    }
    for key, s in region_vals.items():
        x, y = region_pos[key]
        ax.text(x, y, str(len(s)), ha="center", va="center", fontsize=20, fontweight="bold")

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.4)
    ax.set_aspect("equal")


def plot_venn(lang_sets, names, popular_rows, union_all, output_dir: Path):
    """The single chart this script produces: a 3-set Venn diagram of
    language coverage, with a legend highlighting the top widely-spoken
    languages (see POPULAR_LANGUAGES) color-coded to the Venn region each
    one falls into -- since the full language lists are too long to put
    on the chart directly."""
    set_a, set_b, set_c = (lang_sets[n] for n in names)

    fig, ax = plt.subplots(figsize=(13, 9))

    # Horizontal center (in axes-fraction coords, 0-1) that the title should
    # be aligned to. Defaults to 0.5 (middle of the axes); overridden below
    # once we know where the venn circles themselves actually sit, since
    # they are usually NOT centered on the axes (the axes also reserves
    # empty space on the right for the legend, and -- for the real,
    # size-proportional venn3 layout -- the three circles are different
    # sizes/positions depending on the real, unequal set sizes).
    diagram_x_center_frac = 0.5

    if HAS_VENN_LIB:
        v = venn3(
            [set_a, set_b, set_c],
            set_labels=[
                f"{names[0]}\n(n={len(set_a)})",
                f"{names[1]}\n(n={len(set_b)})",
                f"{names[2]}\n(n={len(set_c)})",
            ],
            ax=ax,
        )
        circles = venn3_circles([set_a, set_b, set_c], ax=ax, linewidth=1.4, color="gray")

        # Compute the true left/right extent of the three circles (their
        # actual centers +/- radii, as laid out by venn3's own algorithm)
        # and convert that midpoint into an axes-fraction x, so the title
        # can be centered over the diagram itself rather than over the
        # whole axes box.
        x_min = min(c.center[0] - c.radius for c in circles)
        x_max = max(c.center[0] + c.radius for c in circles)
        xlim = ax.get_xlim()
        if xlim[1] != xlim[0]:
            diagram_x_center_frac = ((x_min + x_max) / 2 - xlim[0]) / (xlim[1] - xlim[0])

        for patch_id in ("100", "010", "001", "110", "101", "011", "111"):
            patch = v.get_patch_by_id(patch_id)
            if patch is not None:
                patch.set_color(REGION_COLORS[patch_id])
                patch.set_alpha(0.6)
        for label_id in ("100", "010", "001", "110", "101", "011", "111"):
            label = v.get_label_by_id(label_id)
            if label is not None:
                label.set_fontsize(20)
                label.set_fontweight("bold")
        for set_label in v.set_labels:
            if set_label is not None:
                set_label.set_fontsize(18)
                set_label.set_fontweight("bold")
    else:
        # The schematic fallback lays its 3 circles out symmetrically
        # (mirrored left/right), so the axes center (0.5) is already the
        # correct horizontal center for the title -- no override needed.
        draw_schematic_venn(ax, names, set_a, set_b, set_c)

    # ---- legend: top popular languages, colored by which Venn region
    # (i.e. which combination of datasets) each one falls into ----
    if popular_rows:
        priority = ["FLORES", "NTREX", "Bible"]  # prefer the more standardized code for the label
        legend_handles = []
        for row in popular_rows:
            present = {name: (row[name] != "-") for name in names}
            bits = "".join("1" if present[n] else "0" for n in names)  # order matches names = [Bible, FLORES, NTREX]
            color = REGION_COLORS.get(bits, REGION_COLORS["000"])
            code = next((row[n] for n in priority if row[n] != "-"), "not found")
            legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="gray", label=f"{row['language']} ({code})"))

        legend = ax.legend(
            handles=legend_handles,
            title=f"Top {len(popular_rows)} Most Common Languages",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=14,
            title_fontsize=16,
            frameon=True,
            borderpad=1.0,
            labelspacing=1.0,
            handlelength=1.6,
            handleheight=1.6,
        )
        legend.get_title().set_fontweight("bold")

    ax.set_axis_off()

    # ---- Center the title on the FULL final page (diagram + legend
    # together), not just on the venn circles. ----
    # We measure the *actual* rendered bounding box of everything drawn so
    # far (circles, region counts, dataset-name labels, legend) -- this is
    # exactly what `bbox_inches="tight"` will crop the saved PNG to -- and
    # find the horizontal midpoint of that box in the page's own
    # (pre-crop) figure coordinates. This is more reliable than guessing a
    # fixed x, because the legend's real footprint (via bbox_to_anchor)
    # isn't known until things are actually laid out.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    content_bbox = fig.get_tightbbox(renderer)  # inches, whole-figure coords
    fig_width_in = fig.get_size_inches()[0]
    page_center_fig_frac = ((content_bbox.x0 + content_bbox.x1) / 2) / fig_width_in

    ax_pos = ax.get_position()  # axes rectangle, in figure-fraction coords
    title_axes_frac = (page_center_fig_frac - ax_pos.x0) / ax_pos.width

    ax.set_title(
        "Language Coverage Across Bible / FLORES / NTREX Datasets\n"
        f"Total unique languages across all 3 datasets (union of language sets): {len(union_all)}",
        fontsize=22,
        fontweight="bold",
        pad=24,
    )
    ax.title.set_x(title_axes_frac)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "language_coverage_venn.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_artifacts(lang_sets, lang_counts_by_ds, n_records_by_ds, names, union_all, inter_all, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full presence/absence matrix: one row per language, one column per
    # dataset. This is the ground truth the Venn diagram's counts come from.
    matrix_path = output_dir / "language_coverage_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language_code"] + names + ["n_datasets_covering_it"])
        for code in sorted(union_all):
            row = [code]
            n_ds = 0
            for name in names:
                present = code in lang_sets[name]
                row.append(int(present))
                n_ds += int(present)
            row.append(n_ds)
            writer.writerow(row)
    print(f"Saved: {matrix_path}")

    summary = {
        "datasets": {
            name: {
                "n_records": n_records_by_ds[name],
                "n_languages": len(lang_sets[name]),
            }
            for name in names
        },
        "total_unique_languages_union": len(union_all),
        "languages_present_in_all_3_datasets": len(inter_all),
        "languages_present_in_all_3_datasets_list": sorted(inter_all),
    }
    summary_path = output_dir / "language_coverage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {summary_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=script_dir.parent / "data" / "processed_alignment",
        help="Folder containing bible.json / flores.json / ntrex.json (default: ../data/processed_alignment)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=script_dir / "output",
        help="Where to save the chart and summary files (default: ./output)",
    )
    parser.add_argument(
        "--top_n_popular",
        type=int,
        default=10,
        help="How many well-known languages to show in the chart's reference table (default: 10)",
    )
    args = parser.parse_args()

    lang_sets, lang_counts_by_ds, n_records_by_ds = {}, {}, {}
    for name, filename in DATASETS.items():
        path = args.data_root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Pass --data_root to point at the folder "
                f"containing bible.json / flores.json / ntrex.json."
            )
        lang_set, lang_counts, n_records = analyze_dataset(path)
        lang_sets[name] = lang_set
        lang_counts_by_ds[name] = lang_counts
        n_records_by_ds[name] = n_records

        avg_completeness = (sum(lang_counts.values()) / len(lang_counts) / n_records) if lang_counts and n_records else 0.0
        print(
            f"[{name}] {path.name}: {n_records} records, {len(lang_set)} languages covered "
            f"(avg. completeness per covered language: {avg_completeness:.1%})"
        )

    names = list(DATASETS.keys())
    set_bible, set_flores, set_ntrex = (lang_sets[n] for n in names)
    union_all = set_bible | set_flores | set_ntrex
    inter_all = set_bible & set_flores & set_ntrex

    print("=" * 60)
    print(f"TOTAL UNIQUE LANGUAGES ACROSS ALL 3 DATASETS (union): {len(union_all)}")
    print(f"Languages present in ALL 3 datasets (intersection):   {len(inter_all)}")
    for name in names:
        print(f"  {name:8s}: {len(lang_sets[name])} languages")
    print("=" * 60)

    popular_rows = []
    for display_name, prefixes in POPULAR_LANGUAGES[: args.top_n_popular]:
        row = {"language": display_name}
        for name in names:
            code = find_popular_code(lang_sets[name], prefixes)
            row[name] = code if code else "-"
        popular_rows.append(row)

    plot_venn(lang_sets, names, popular_rows, union_all, args.output_dir)
    save_artifacts(lang_sets, lang_counts_by_ds, n_records_by_ds, names, union_all, inter_all, args.output_dir)


if __name__ == "__main__":
    main()