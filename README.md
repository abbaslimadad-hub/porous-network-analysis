# Ring Counter v2 — STM pore-network pentagon / hexagon / heptagon counter

A Python desktop tool for detecting and counting 5-, 6-, and 7-sided pores
in 2D network images (STM images for example). Optional preprocessing handles
scan-line offset, tilt, double-tip ghosts, drift shear, and slow creep before
detection.

## Requirements

- Python 3.9 or newer
- numpy, scipy, scikit-image, networkx, matplotlib
- tkinter (ships with CPython; on Linux install python3-tk from your distro)

Install with:

    pip install -r requirements.txt

## Running

    python ring_counter.py

Then:

1. Click Load Root and choose a directory containing folders named
   exactly `scale_png`. Images must be named `..._scale_<FOV_nm>.png`
   so nm/px can be resolved from the filename.
2. Press AUTO-DETECT (or A). Review and correct the overlay with
   left-click (add) / right-click (remove).
3. Press SAVE & NEXT (or S). Outputs are written to
   Analysis_Results/<name>_ANALYZED.{txt,json,png}.
4. For a whole folder at once, use BATCH ALL (double confirmation;
   writes without manual review and flags suspicious images as
   NEEDS_REVIEW).

Keyboard shortcuts are listed in the sidebar. D opens a stage-by-stage
debug view showing which faces were kept or rejected and why.

## Coordinates in the output

_ANALYZED.txt and _ANALYZED.json contain two coordinate sets:

- x, y         corrected frame, matching the saved _ANALYZED.png
- raw_x, raw_y raw image, after inverting every affine / drift step

The affine matrix used (if any) is stored in _ANALYZED.json under affine_A.

## Default detection parameters

| Parameter | Default |
|---|---|
| Periodicity window (all three shapes) | 2.3 - 3.1 nm, +/-15 % margin |
| Angle tolerance from ideal interior angle | 55 deg |
| Edge-length CV tolerance | 0.65 |
| Wall mask method | otsu_local (alternative: ridge) |
| Junction merge radius | 0.6 nm |
| Border exclusion margin | 0.35 x lattice period |
| Detection method | hybrid (face tracing + pore-centre fill) |

## Known limitations

- Faces touching the image border are excluded by default, which
  systematically undercounts very small fields.
- Skeleton face tracing assumes the planar-graph rotation system is
  well-defined; heavily non-planar skeletons may lose faces.
- Local drift correction is off by default; it can be misleading on
  very sparse networks.
- QC thresholds (min_rings = 10, min_coverage = 25 %) are tuned for
  fields of roughly 200-2000 nm and should be adjusted for other scales.

## Citation

If you use this code, please cite the DOI shown in the Zenodo badge
on this repository. The concept DOI always resolves to the latest
released version.

## License

Dedicated to the public domain under CC0 1.0 Universal. See LICENSE.
