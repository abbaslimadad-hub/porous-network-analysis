import os
import re
import json
import itertools
import warnings
import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import io, color, filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.filters import sato
from skimage.restoration import unwrap_phase
from skimage.transform import AffineTransform, warp
from scipy.ndimage import convolve, gaussian_filter, map_coordinates, median_filter, maximum_filter
from scipy.spatial import Delaunay, Voronoi
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

MODES = ["Hexagon", "Pentagon", "Heptagon"]
MODE_SIDES = {"Hexagon": 6, "Pentagon": 5, "Heptagon": 7}
SIDES_MODE = {v: k for k, v in MODE_SIDES.items()}
MODE_COLORS = {"Hexagon": "yellow", "Pentagon": "cyan", "Heptagon": "magenta"}
MODE_KEYS = {"1": "Hexagon", "2": "Pentagon", "3": "Heptagon"}
PROGRESS_FILENAME = ".ring_counter_progress.json"

DEFAULT_BG_SIGMA_NM = 0.35
DEFAULT_CLOSE_RADIUS_NM = 0.20
DEFAULT_JUNCTION_DILATE_NM = 0.1
DEFAULT_MASK_METHOD = "otsu_local"
MASK_METHODS = ["otsu_local", "ridge"]

GRID_BG_SIGMA_NM = [0.25, 0.35, 0.45, 0.6, 0.8]
GRID_CLOSE_RADIUS_NM = [0.15, 0.20, 0.25]
GRID_JUNCTION_DILATE_NM = [0.05, 0.1, 0.15]
GRID_MASK_METHODS = ["otsu_local", "ridge"]

DEFAULT_WINDOWS = {
    "Hexagon": (2.3, 3.1, 0.15),
    "Heptagon": (2.3, 3.1, 0.15),
    "Pentagon": (2.3, 3.1, 0.15),
}
DETECTION_METHODS = [("hybrid", "Hybrid (trace + pore-centre fill)"),
                     ("tracing", "Face tracing only"),
                     ("centers", "Pore centres only (Delaunay)")]

IDEAL_INTERIOR_ANGLE = {5: 108.0, 6: 120.0, 7: 128.5714}


def default_params():
    return dict(
        method="hybrid",
        angle_tolerance_deg=55.0,
        edge_tolerance=0.65,
        bg_sigma_nm=DEFAULT_BG_SIGMA_NM,
        close_radius_nm=DEFAULT_CLOSE_RADIUS_NM,
        junction_dilate_nm=DEFAULT_JUNCTION_DILATE_NM,
        mask_method=DEFAULT_MASK_METHOD,
        pore_neighbor_cutoff=1.4,
        exclude_border=True,
        border_margin_periods=0.35,
        junction_merge_nm=0.6,      
        windows=dict(DEFAULT_WINDOWS),
    )


def default_prep_opts():
    return dict(line_align=True, invert=False, affine=True, local_drift=False, double_tip=None)


def period_from_params(params):
    lo, hi, _ = params["windows"]["Hexagon"]
    return 0.5 * (lo + hi)



def parse_fov_nm(filename):
    """'..._scale_100.png' -> 100.0 nm. None if not found."""
    m = re.search(r'scale[_-]?(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
    return float(m.group(1)) if m else None


def load_gray(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return np.asarray(img, dtype=float)


def resolve_scale(path, shape):
    """-> (nm_per_px, parsed_from_filename)."""
    fov = parse_fov_nm(os.path.basename(path))
    if fov:
        return fov / max(shape), True
    return 100.0 / max(shape), False


def line_align(gray, nm_per_px=None, period_nm=None):
    
    g = np.asarray(gray, float).copy()
    rowmed = np.median(g, axis=1)
    win = 5
    if nm_per_px and period_nm:
        win = max(3, int(round(period_nm / nm_per_px / 4.0)))
    if win % 2 == 0:
        win += 1
    smooth = median_filter(rowmed, size=win, mode='nearest')
    g -= (rowmed - smooth)[:, None]

    h, w = g.shape
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    Am = np.column_stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    coef, *_ = np.linalg.lstsq(Am, g[::4, ::4].ravel(), rcond=None)
    yy, xx = np.mgrid[0:h, 0:w]
    g -= coef[0] * xx + coef[1] * yy + coef[2]
    return g



def remove_double_tip(gray, dx_px, dy_px, a=0.5, eps=0.05):
    gray = np.asarray(gray, float)
    h, w = gray.shape
    pad = int(2 * max(abs(dx_px), abs(dy_px))) + 8
    pad = min(pad, min(h, w) - 1)
    g = np.pad(gray - gray.mean(), pad, mode='reflect')
    H2, W2 = g.shape
    F = np.fft.fft2(g)
    ky = np.fft.fftfreq(H2)[:, None]
    kx = np.fft.fftfreq(W2)[None, :]
    Hf = 1.0 + a * np.exp(-2j * np.pi * (kx * dx_px + ky * dy_px))
    G = F * np.conj(Hf) / (np.abs(Hf) ** 2 + eps)
    out = np.fft.ifft2(G).real[pad:pad + h, pad:pad + w]
    return out + gray.mean()


def _acorr(gray):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    z = (gray - gray.mean()) * win
    F = np.fft.fft2(z, s=(2 * h, 2 * w))
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(F) ** 2).real)
    Wf = np.fft.fft2(win, s=(2 * h, 2 * w))
    wac = np.fft.fftshift(np.fft.ifft2(np.abs(Wf) ** 2).real)
    ac = ac / np.maximum(wac, 1e-6 * wac.max())
    return ac / ac[h, w]


def _inversion_score(gray):
    
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    z = (gray - gray.mean()) * win
    F = np.fft.fft2(z, s=(2 * h, 2 * w))
    conv = np.fft.ifft2(F * F).real
    return float(conv.max() / ((z ** 2).sum() + 1e-12))


def estimate_double_tip(gray, nm_per_px, period_nm):
    
    h, w = gray.shape
    fit, info = fit_hex_lattice(gray, nm_per_px, period_nm)
    if fit is None:
        return None, info
    ga, gb, gc = fit
    G = np.column_stack([ga, gc])
    Ginv_T = np.linalg.inv(G.T)
    A, _ = estimate_hex_affine(gray, nm_per_px, period_nm, max_aniso=10.0)
    if A is None:
        A = np.eye(2)
    Ainv = np.linalg.inv(A)

    P = period_nm / nm_per_px
    ac = _acorr(gray)
    acm = maximum_filter(ac, size=5)
    rmax = min(5.0 * P, 0.3 * min(h, w))
    yy, xx = np.mgrid[-h:h, -w:w]
    rr = np.hypot(xx, yy)
    region = (rr > 0.25 * P) & (rr < rmax) & ((yy > 0) | ((yy == 0) & (xx > 0)))
    pk = peak_local_max(np.where(region, ac, 0.0), min_distance=2, threshold_abs=0.03,
                        exclude_border=False)
    rots = [np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            for t in np.radians([60, 120, 180, 240, 300])]
    best = None
    for r, c in pk:
        d = np.array([c - w, r - h], float)
        f = G.T @ d
        if np.linalg.norm(Ginv_T @ (f - np.round(f))) < 0.15 * P:   # lattice vector -> harmless
            continue
        sym = []
        for R in rots:
            d2 = Ainv @ (R @ (A @ d))
            ix, iy = int(round(d2[0])) + w, int(round(d2[1])) + h
            if 0 <= iy < acm.shape[0] and 0 <= ix < acm.shape[1]:
                sym.append(acm[iy, ix])
        if not sym:
            continue
        excess = ac[r, c] - float(np.mean(sym))
        if excess > 0.05 and (best is None or excess > best[0]):
            # parabolic sub-pixel refinement of the peak position
            def par(m, c0, p):
                den = m - 2 * c0 + p
                return 0.0 if abs(den) < 1e-12 else 0.5 * (m - p) / den
            dxr = par(ac[r, c - 1], ac[r, c], ac[r, c + 1])
            dyr = par(ac[r - 1, c], ac[r, c], ac[r + 1, c])
            best = (excess, d + np.array([dxr, dyr]))
    if best is None:
        return None, {"reason": "no asymmetric autocorrelation peak (no ghost found)"}
    excess, d0 = best

    # central crop keeps the search fast
    m = min(h, w, 320)
    r0, c0 = (h - m) // 2, (w - m) // 2
    crop = np.asarray(gray, float)[r0:r0 + m, c0:c0 + m]

    def score(dx, dy, a):
        return _inversion_score(remove_double_tip(crop, dx, dy, a))

    cur = None
    for sign in (+1, -1):
        for a in (0.2, 0.35, 0.5, 0.65, 0.8):
            sc = score(sign * d0[0], sign * d0[1], a)
            if cur is None or sc > cur[0]:
                cur = (sc, sign * d0[0], sign * d0[1], a)
    sc, dx, dy, a = cur
    for step_xy, step_a in ((0.5, 0.05), (0.25, 0.025)):
        for _ in range(4):
            moved = False
            for ddx, ddy, dda in ((step_xy, 0, 0), (-step_xy, 0, 0), (0, step_xy, 0), (0, -step_xy, 0),
                                  (0, 0, step_a), (0, 0, -step_a)):
                na = a + dda
                if not (0.05 <= na <= 0.9):
                    continue
                s2 = score(dx + ddx, dy + ddy, na)
                if s2 > sc + 1e-4:
                    sc, dx, dy, a, moved = s2, dx + ddx, dy + ddy, na, True
            if not moved:
                break
    base_score = _inversion_score(crop)
    info = {"excess": float(excess), "score_before": base_score, "score_after": sc}
    if sc < base_score + 0.02:
        return None, dict(info, reason="deconvolution did not improve lattice symmetry")
    return (float(dx), float(dy), float(a)), info



def fit_hex_lattice(gray, nm_per_px, period_nm, band=(0.6, 1.6), n_peaks=10):
    """Find the three first-order reciprocal-lattice vectors (cycles/px).
    Returns ((ga, gb, gc), info) with ga + gc == gb (angularly sorted), or (None, info)."""
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    z = (gray - gray.mean()) * win
    H2, W2 = 2 * h, 2 * w
    F = np.abs(np.fft.fftshift(np.fft.fft2(z, s=(H2, W2))))
    ky = np.fft.fftshift(np.fft.fftfreq(H2))
    kx = np.fft.fftshift(np.fft.fftfreq(W2))
    KY, KX = ky[:, None], kx[None, :]
    kr = np.hypot(KX, KY)
    k_exp = 2.0 / (np.sqrt(3) * period_nm / nm_per_px)
    half = (KY > 0) | ((KY == 0) & (KX > 0))
    ring = (kr > band[0] * k_exp) & (kr < band[1] * k_exp) & half
    Fm = F * ring
    if Fm.max() <= 0:
        return None, {"reason": "empty spectrum in lattice band"}
    pk = peak_local_max(Fm, min_distance=4, num_peaks=n_peaks, threshold_rel=0.15,
                        exclude_border=False)
    if len(pk) < 3:
        return None, {"reason": "fewer than 3 lattice peaks"}

    cands = []
    for r, c in pk:
        r0, r1 = max(r - 2, 0), min(r + 3, H2)
        c0, c1 = max(c - 2, 0), min(c + 3, W2)
        wgt = Fm[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        s = wgt.sum() + 1e-12
        pc, pr = (cc * wgt).sum() / s, (rr * wgt).sum() / s
        g = np.array([np.interp(pc, np.arange(W2), kx), np.interp(pr, np.arange(H2), ky)])
        cands.append((np.arctan2(g[1], g[0]), g, float(F[r, c])))
    cands.sort(key=lambda t: t[0])

    best = None
    for (a1, g1, m1), (a2, g2, m2), (a3, g3, m3) in itertools.combinations(cands, 3):
        lens = [np.linalg.norm(g) for g in (g1, g2, g3)]
        if max(lens) / max(min(lens), 1e-12) > 1.6:
            continue
        resid = np.linalg.norm(g1 + g3 - g2) / (np.linalg.norm(g2) + 1e-12)
        if resid > 0.15:
            continue
        sc = (m1 + m2 + m3) * (1.0 - resid)
        if best is None or sc > best[0]:
            best = (sc, (g1, g2, g3), resid)
    if best is None:
        return None, {"reason": "no hex-consistent triple of lattice peaks"}
    _, trip, resid = best
    return trip, {"resid": float(resid),
                  "g_lengths": [float(np.linalg.norm(g)) for g in trip]}


def estimate_hex_affine(gray, nm_per_px, period_nm, max_aniso=1.35):
    
    fit, info = fit_hex_lattice(gray, nm_per_px, period_nm)
    if fit is None:
        info["ok"] = False
        return None, info
    ga, gb, gc = fit
    G = np.column_stack([ga, gc])
    k0 = np.sqrt(abs(np.linalg.det(G)) / (np.sqrt(3) / 2))
    th = np.arctan2(ga[1], ga[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    G0 = k0 * R @ np.array([[1, -0.5], [0, np.sqrt(3) / 2]])
    A = np.linalg.inv(G0).T @ G.T
    s = np.linalg.svd(A, compute_uv=False)
    aniso = float(s[0] / s[1])
    info.update(aniso=aniso, G=G)
    if aniso > max_aniso:
        info.update(ok=False, reason=f"correction too extreme (aniso {aniso:.2f})")
        return None, info
    if np.abs(A - np.eye(2)).max() < 0.01:
        info.update(ok=True, reason="already ideal (no correction needed)")
        return None, info
    info["ok"] = True
    return A, info


def apply_affine(gray, A):
    
    h, w = gray.shape
    c = np.array([w / 2.0, h / 2.0])
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = c - A @ c
    inv = AffineTransform(matrix=M).inverse
    mu = gray.mean()
    out = warp(gray - mu, inv, order=1, mode='constant', cval=0.0, preserve_range=True) + mu
    valid = warp(np.ones_like(gray), inv, order=0, mode='constant', cval=0.0) > 0.5
    hi = min(h, w) // 4
    if not valid[hi:h - hi, hi:w - hi].all():
        m = hi
    else:
        lo, up = 0, hi
        while lo < up:
            mid = (lo + up) // 2
            if valid[mid:h - mid, mid:w - mid].all():
                up = mid
            else:
                lo = mid + 1
        m = lo
    if m == 0:
        return out, 0
    return out[m:h - m, m:w - m], m



def local_drift_correct(gray, nm_per_px, period_nm, sigma_periods=1.5, max_disp_periods=0.75):
    
    fit, info = fit_hex_lattice(gray, nm_per_px, period_nm)
    if fit is None:
        info["ok"] = False
        return None, None, info
    ga, gb, gc = fit
    h, w = gray.shape
    P = period_nm / nm_per_px
    sigma = sigma_periods * P
    m = int(2 * sigma)
    if h - 2 * m < 40 or w - 2 * m < 40:
        return None, None, {"ok": False, "reason": "image too small for local drift fit"}
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    z = gray - gray.mean()
    phases, amps = [], []
    for g in (ga, gc):
        car = np.exp(-2j * np.pi * (g[0] * xx + g[1] * yy))
        d = z * car
        re = gaussian_filter(d.real, sigma)
        im = gaussian_filter(d.imag, sigma)
        phases.append(unwrap_phase(np.arctan2(im, re)) / (2 * np.pi))
        amps.append(np.hypot(re, im))

    def design(X, Y, const=True):
        cols = [X, Y, X * X, X * Y, Y * Y]
        if const:                       # unwrap_phase leaves an arbitrary constant offset
            cols.append(np.ones_like(X))
        return np.column_stack(cols)

    sub = (slice(m, h - m, 4), slice(m, w - m, 4))
    Xs = (xx[sub] / (w / 2.0) - 1).ravel()
    Ys = (yy[sub] / (h / 2.0) - 1).ravel()
    Ds = design(Xs, Ys)
    amp_ok = (np.minimum(amps[0], amps[1])[sub].ravel() > 0.3 * np.median(np.minimum(amps[0], amps[1])))
    Xf = xx / (w / 2.0) - 1
    Yf = yy / (h / 2.0) - 1
    P_fields, rms = [], []
    for ph in phases:
        y = ph[sub].ravel()
        keep = amp_ok.copy()
        coef = np.zeros(6)
        for _ in range(5):
            if keep.sum() < 20:
                return None, None, {"ok": False, "reason": "too few reliable phase samples"}
            coef, *_ = np.linalg.lstsq(Ds[keep], y[keep], rcond=None)
            res = y - Ds @ coef
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            keep = amp_ok & (np.abs(res) < 2.5 * max(s, 0.02))
        rms.append(float(np.sqrt(np.mean(res[keep] ** 2))))
        P_fields.append(design(Xf.ravel(), Yf.ravel(), const=False).dot(coef[:5]).reshape(h, w))
    Gm = np.array([[ga[0], ga[1]], [gc[0], gc[1]]])
    Gi = np.linalg.inv(Gm)
    ux = -(Gi[0, 0] * P_fields[0] + Gi[0, 1] * P_fields[1])
    uy = -(Gi[1, 0] * P_fields[0] + Gi[1, 1] * P_fields[1])
    max_u = float(np.hypot(ux, uy).max())
    info = {"ok": True, "max_disp_px": max_u, "fit_rms_cycles": float(np.mean(rms))}
    if max_u > max_disp_periods * P:
        info.update(ok=False, reason=f"displacement too large ({max_u:.1f}px) - fit unreliable")
        return None, None, info
    if max_u < 0.3:
        info["reason"] = "no measurable local drift"
        return None, None, info
    out = map_coordinates(gray, [yy + uy, xx + ux], order=1, mode='reflect')
    crop = int(np.ceil(max_u)) + 1
    out = out[crop:h - crop, crop:w - crop]
    return out, {"type": "drift", "ux": ux, "uy": uy, "crop": crop, "shape": (h, w)}, info



def prepare_image(gray_raw, nm_per_px, opts, period_nm):
    
    gray = np.asarray(gray_raw, float).copy()
    meta = {"steps": [], "notes": [], "problems": [], "affine_A": None,
            "affine_info": None, "drift_info": None, "raw_shape": gray.shape}

    if opts.get("line_align"):
        gray = line_align(gray, nm_per_px, period_nm)
        meta["notes"].append("line-align")
    if opts.get("invert"):
        gray = gray.max() + gray.min() - gray
        meta["notes"].append("inverted")
    dt = opts.get("double_tip")
    if dt:
        gray = remove_double_tip(gray, *dt)
        meta["notes"].append("double-tip dx=%.1f dy=%.1f a=%.2f" % dt)

    if opts.get("affine"):
        shape_before = gray.shape
        A, info = estimate_hex_affine(gray, nm_per_px, period_nm)
        meta["affine_info"] = info
        if A is not None:
            gray, crop = apply_affine(gray, A)
            meta["steps"].append({"type": "affine", "A": A, "crop": crop, "shape": shape_before})
            meta["affine_A"] = A
            meta["notes"].append("affine (aniso %.2f)" % info["aniso"])
        elif not info.get("ok"):
            meta["problems"].append("affine rejected: " + info.get("reason", "?"))
        else:
            meta["notes"].append("affine not needed")

    if opts.get("local_drift"):
        out, step, info = local_drift_correct(gray, nm_per_px, period_nm)
        meta["drift_info"] = info
        if out is not None:
            gray = out
            meta["steps"].append(step)
            meta["notes"].append("local drift (max %.1fpx)" % info["max_disp_px"])
        elif not info.get("ok", True):
            meta["problems"].append("local drift rejected: " + info.get("reason", "?"))
    return gray, meta


def to_original_coords(x, y, meta):
    """Map a point from the final (corrected, cropped) frame back to the raw image."""
    p = np.array([x, y], float)
    for st in reversed(meta["steps"]):
        p = p + st["crop"]
        if st["type"] == "affine":
            h, w = st["shape"]
            c = np.array([w / 2.0, h / 2.0])
            p = c + np.linalg.inv(st["A"]) @ (p - c)
        elif st["type"] == "drift":
            ux, uy = st["ux"], st["uy"]
            ix = int(np.clip(round(p[0]), 0, ux.shape[1] - 1))
            iy = int(np.clip(round(p[1]), 0, ux.shape[0] - 1))
            p = p + np.array([ux[iy, ix], uy[iy, ix]])
    return float(p[0]), float(p[1])



def compute_flat(gray, nm_per_px, bg_sigma_nm):
    bg_sigma_px = max(3.0, bg_sigma_nm / nm_per_px)
    bg = filters.gaussian(gray, sigma=bg_sigma_px)
    flat = gray - bg
    flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-9)
    return flat, bg_sigma_px


def build_skeleton_graph(gray, nm_per_px=None,
                         bg_sigma_nm=DEFAULT_BG_SIGMA_NM,
                         close_radius_nm=DEFAULT_CLOSE_RADIUS_NM,
                         junction_dilate_nm=DEFAULT_JUNCTION_DILATE_NM,
                         mask_method=DEFAULT_MASK_METHOD,
                         junction_merge_nm=1.2,
                         return_debug=False):
    
    if not nm_per_px or nm_per_px <= 0:
        nm_per_px = 1.0

    flat, bg_sigma_px = compute_flat(gray, nm_per_px, bg_sigma_nm)
    smooth_sigma_px = max(0.5, 0.5 / nm_per_px)
    smooth = filters.gaussian(flat, sigma=smooth_sigma_px)

    if mask_method == "ridge":
        
        sigmas = sorted({max(1.0, wn / nm_per_px / 2.0) for wn in (0.3, 0.8)})
        r = sato(smooth, sigmas=sigmas, black_ridges=False)
        bright_mask = r > filters.threshold_otsu(r)
    else:
        global_thresh = filters.threshold_otsu(smooth)
        block_size_px = int(round(bg_sigma_px * 4))
        if block_size_px % 2 == 0:
            block_size_px += 1
        block_size_px = max(block_size_px, 7)
        try:
            local_thresh_map = filters.threshold_local(smooth, block_size=block_size_px, offset=0.0)
            bright_mask = (smooth >= global_thresh) | (smooth >= local_thresh_map)
        except Exception:
            bright_mask = smooth >= global_thresh

    min_size_px2 = max(10, int(round(1.0 / (nm_per_px ** 2))))
    bright_mask = morphology.remove_small_objects(bright_mask, min_size=min_size_px2)
    bright_mask = morphology.remove_small_holes(bright_mask, area_threshold=min_size_px2)

    close_radius_px = max(1, int(round(close_radius_nm / nm_per_px)))
    bright_mask = morphology.closing(bright_mask, morphology.disk(close_radius_px))
    skel = morphology.skeletonize(bright_mask)

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(skel.astype(int), kernel, mode='constant')
    deg_map = neighbor_count * skel
    ys, xs = np.where(skel)
    pixset = set(zip(ys.tolist(), xs.tolist()))

    junction_mask = deg_map >= 3
    junction_dilate_px = max(1, int(round(junction_dilate_nm / nm_per_px)))
    junction_mask_dilated = morphology.dilation(junction_mask, morphology.disk(junction_dilate_px))
    junction_labels = measure.label(junction_mask_dilated & skel)

    if junction_labels.max() > 0:
        props = measure.regionprops(junction_labels)
        centroids = {p.label: np.array(p.centroid) for p in props}
        merge_radius_px = max(1.5, junction_merge_nm / nm_per_px)
        parent = {lab: lab for lab in centroids}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        labels_list = list(centroids.keys())
        carr = np.array([centroids[l] for l in labels_list])
        for i in range(len(labels_list)):
            d = np.linalg.norm(carr[i + 1:] - carr[i], axis=1)
            for k in np.where(d <= merge_radius_px)[0]:
                union(labels_list[i], labels_list[i + 1 + k])
        remap = {lab: find(lab) for lab in labels_list}
    else:
        remap = {}

    G = nx.Graph()
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p2 = (y + dy, x + dx)
                if p2 in pixset:
                    G.add_edge((y, x), p2)

    node_map = {}
    for y, x in zip(ys, xs):
        lab = junction_labels[y, x]
        if lab > 0:
            lab = remap.get(lab, lab)
            node_map[(y, x)] = ('J', lab)
        else:
            node_map[(y, x)] = ('P', y, x)

    H = nx.MultiGraph()
    cluster_pixels = {}
    for (y, x), nid in node_map.items():
        if nid[0] == 'J':
            cluster_pixels.setdefault(nid, []).append((y, x))
    for nid, pix in cluster_pixels.items():
        H.add_node(nid, pos=(float(np.mean([p[1] for p in pix])), float(np.mean([p[0] for p in pix]))))
    for (y, x) in zip(ys, xs):
        nid = node_map[(y, x)]
        if nid[0] == 'P':
            H.add_node(nid, pos=(float(x), float(y)))
    for a, b in G.edges():
        na, nb = node_map[a], node_map[b]
        if na != nb:
            H.add_edge(na, nb)

    changed = True
    while changed:
        changed = False
        for node in list(H.nodes()):
            if node[0] == 'P' and H.degree(node) == 2:
                nbrs = list(H.neighbors(node))
                if len(nbrs) == 2 and nbrs[0] != nbrs[1]:
                    a, b = nbrs
                    H.remove_node(node)
                    H.add_edge(a, b)
                    changed = True
            elif node[0] == 'P' and H.degree(node) < 2:
                H.remove_node(node)
                changed = True

    it = 0
    while it < 200:
        leaves = [n for n in H.nodes() if H.degree(n) <= 1]
        if not leaves:
            break
        H.remove_nodes_from(leaves)
        it += 1

    if return_debug:
        dbg = {"flat": flat, "smooth": smooth, "mask": bright_mask, "skel": skel,
               "junctions": junction_mask}
        return H, dbg
    return H


def find_faces(H):
    """Trace enclosed faces of the planar straight-line graph (rotation system)."""
    S = nx.Graph(H)          # collapse parallel edges so the rotation system is well defined
    pos = nx.get_node_attributes(S, 'pos')
    adj = {n: [] for n in S.nodes()}
    for a, b in S.edges():
        adj[a].append(b)
        adj[b].append(a)
    for n in adj:
        nbrs = adj[n]
        px, py = pos[n]
        angs = [np.arctan2(pos[m][1] - py, pos[m][0] - px) for m in nbrs]
        adj[n] = [nbrs[k] for k in np.argsort(angs)]

    def next_in_face(u, v):
        nbrs = adj[v]
        i = nbrs.index(u)
        return nbrs[(i - 1) % len(nbrs)]

    visited = set()
    faces = []
    for a, b in S.edges():
        for (u, v) in [(a, b), (b, a)]:
            if (u, v) in visited:
                continue
            face = []
            cu, cv = u, v
            steps = 0
            while True:
                visited.add((cu, cv))
                face.append(cu)
                w = next_in_face(cu, cv)
                cu, cv = cv, w
                steps += 1
                if (cu, cv) == (u, v) or steps > 30:
                    break
            faces.append(face)
    return faces, pos



def polygon_area_from_points(poly_pts):
    a = 0.0
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2


def interior_angles(poly_pts):
    n = len(poly_pts)
    angles = []
    for i in range(n):
        p_prev = np.array(poly_pts[(i - 1) % n])
        p_cur = np.array(poly_pts[i])
        p_next = np.array(poly_pts[(i + 1) % n])
        u = p_prev - p_cur
        w = p_next - p_cur
        cosang = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return angles


def max_angle_deviation(poly_pts, n_sides):
    ideal = IDEAL_INTERIOR_ANGLE[n_sides]
    return max(abs(a - ideal) for a in interior_angles(poly_pts))


def edge_length_cv(poly_pts):
    n = len(poly_pts)
    lens = np.array([np.linalg.norm(np.array(poly_pts[(i + 1) % n]) - np.array(poly_pts[i]))
                     for i in range(n)])
    mean_l = lens.mean()
    if mean_l < 1e-9:
        return 999.0
    return float(lens.std() / mean_l)


def polygon_periodicity_nm(poly_pts, nm_per_px):
    
    n = len(poly_pts)
    k = n // 2
    if k < 1:
        return None
    pts = [np.array(p) for p in poly_pts]
    mids = [(pts[i] + pts[(i + 1) % n]) / 2.0 for i in range(n)]
    dists = sorted((np.linalg.norm(mids[i] - mids[j]) for i in range(n) for j in range(i + 1, n)),
                   reverse=True)
    return float(np.mean(dists[:k])) * nm_per_px


def simplify_face_spikes(poly_pts, spike_threshold_deg=35.0, max_iterations=10):
    
    pts = list(poly_pts)
    for _ in range(max_iterations):
        if len(pts) <= 3:
            break
        angs = interior_angles(pts)
        i = int(np.argmin(angs))
        if angs[i] < spike_threshold_deg:
            pts.pop(i)
        else:
            break
    return pts


def _in_window(value_nm, window):
    lo, hi, margin = window
    return (lo * (1.0 - margin)) <= value_nm <= (hi * (1.0 + margin))


def detect_by_tracing(gray, nm_per_px, params, return_debug=False):
    H, sdbg = build_skeleton_graph(
        gray, nm_per_px, bg_sigma_nm=params["bg_sigma_nm"],
        close_radius_nm=params["close_radius_nm"],
        junction_dilate_nm=params["junction_dilate_nm"],
        mask_method=params.get("mask_method", DEFAULT_MASK_METHOD),
        junction_merge_nm=params.get("junction_merge_nm", 0.6), return_debug=True)
    faces, pos = find_faces(H)
    angle_tol = params["angle_tolerance_deg"]
    edge_tol = params["edge_tolerance"]
    stats = {"n_faces": len(faces), "n_567": 0, "n_shape_ok": 0, "n_traced": 0}
    face_dbg = []

    shape_ok = []
    for f in faces:
        raw_pts = [pos[v] for v in f]
        poly_pts = simplify_face_spikes(raw_pts)
        n = len(poly_pts)
        status = None
        if n not in (5, 6, 7):
            status = "sides"
        else:
            stats["n_567"] += 1
            if max_angle_deviation(poly_pts, n) > angle_tol:
                status = "angle"
            elif edge_length_cv(poly_pts) > edge_tol:
                status = "edgecv"
            else:
                area_nm2 = abs(polygon_area_from_points(poly_pts)) * nm_per_px ** 2
                shape_ok.append((poly_pts, n, area_nm2, len(face_dbg)))
        face_dbg.append({"poly": poly_pts, "status": status or "pending"})
    stats["n_shape_ok"] = len(shape_ok)

    P_px = period_from_params(params) / nm_per_px
    bm = max(3.0, params.get("border_margin_periods", 0.35) * P_px) if params.get("exclude_border", True) else -1.0
    hh, ww = gray.shape

    results = []
    for poly_pts, n, area_nm2, idx in shape_ok:
        mode = SIDES_MODE[n]
        if bm > 0 and any(px < bm or py < bm or px > ww - 1 - bm or py > hh - 1 - bm
                          for px, py in poly_pts):
            face_dbg[idx]["status"] = "border"
            continue
        per = polygon_periodicity_nm(poly_pts, nm_per_px)
        if per is None or not _in_window(per, params["windows"][mode]):
            face_dbg[idx]["status"] = "periodicity"
            continue
        face_dbg[idx]["status"] = "ok"
        results.append({"mode": mode,
                        "x": float(np.mean([p[0] for p in poly_pts])),
                        "y": float(np.mean([p[1] for p in poly_pts])),
                        "poly": poly_pts, "area_nm2": area_nm2,
                        "periodicity_nm": per, "source": "trace"})
    stats["n_traced"] = len(results)

    dbg = {"stats": stats}
    if return_debug:
        dbg.update(sdbg)
        dbg["faces"] = [f for f in face_dbg if len(f["poly"]) <= 14]
    return results, dbg


def detect_by_centers(gray, nm_per_px, params, return_debug=False):
    
    P = period_from_params(params) / nm_per_px
    dbg = {"centers": np.zeros((0, 2))}
    bg = filters.gaussian(gray, sigma=1.5 * P)
    sm = filters.gaussian(gray - bg, sigma=max(1.0, 0.18 * P))
    thr = filters.threshold_otsu(sm)
    pk = peak_local_max(-sm, min_distance=max(2, int(0.5 * P)), exclude_border=False)
    pk = [p for p in pk if sm[p[0], p[1]] < thr]
    if len(pk) < 7:
        return [], dbg
    xy = np.array([[c, r] for r, c in pk], float)
    dbg["centers"] = xy
    try:
        tri = Delaunay(xy)
        vor = Voronoi(xy)
    except Exception:
        return [], dbg

    cutoff = params.get("pore_neighbor_cutoff", 1.4) * P
    nbrs = {i: set() for i in range(len(xy))}
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(s[a]), int(s[b])
            if np.linalg.norm(xy[i] - xy[j]) <= cutoff:
                nbrs[i].add(j)
                nbrs[j].add(i)
    hull = set(int(v) for v in tri.convex_hull.ravel())

    out = []
    for i in range(len(xy)):
        if i in hull:
            continue
        n = len(nbrs[i])
        if n not in (5, 6, 7):
            continue
        angs = sorted(np.arctan2(xy[j, 1] - xy[i, 1], xy[j, 0] - xy[i, 0]) for j in nbrs[i])
        gaps = np.diff(angs + [angs[0] + 2 * np.pi])
        if gaps.max() > np.radians(105):
            continue
        mode = SIDES_MODE[n]
        per_nm = float(np.mean([np.linalg.norm(xy[j] - xy[i]) for j in nbrs[i]])) * nm_per_px
        if not _in_window(per_nm, params["windows"][mode]):
            continue
        poly, area_nm2 = None, None
        region = vor.regions[vor.point_region[i]]
        if len(region) >= 3 and -1 not in region:
            v = vor.vertices[region]
            order = np.argsort(np.arctan2(v[:, 1] - xy[i, 1], v[:, 0] - xy[i, 0]))
            poly = [(float(v[k, 0]), float(v[k, 1])) for k in order]
            area_nm2 = abs(polygon_area_from_points(poly)) * nm_per_px ** 2
        out.append({"mode": mode, "x": float(xy[i, 0]), "y": float(xy[i, 1]), "poly": poly,
                    "area_nm2": area_nm2, "periodicity_nm": per_nm, "source": "centers"})
    return out, dbg



def detect_rings(gray, nm_per_px, params, return_debug=False):
    
    method = params.get("method", "hybrid")
    traced, centers = [], []
    stats = {"n_faces": 0, "n_567": 0, "n_shape_ok": 0, "n_traced": 0,
             "n_centers": 0, "n_centers_added": 0}
    if method in ("tracing", "hybrid"):
        traced, tdbg = detect_by_tracing(gray, nm_per_px, params)
        stats.update(tdbg["stats"])
    if method in ("centers", "hybrid"):
        centers, _ = detect_by_centers(gray, nm_per_px, params)
        stats["n_centers"] = len(centers)

    merged = list(traced)
    if centers:
        P = period_from_params(params) / nm_per_px
        tx = np.array([[c["x"], c["y"]] for c in traced]) if traced else None
        for c in centers:
            if tx is not None and np.min(np.hypot(tx[:, 0] - c["x"], tx[:, 1] - c["y"])) < 0.45 * P:
                continue
            merged.append(c)
            stats["n_centers_added"] += 1
    return merged, stats


def score_candidates(candidates):
    
    total = 0.0
    for c in candidates:
        if c.get("source") != "trace":
            continue
        try:
            n = MODE_SIDES[c["mode"]]
            ang_dev = max_angle_deviation(c["poly"], n)
            cv = edge_length_cv(c["poly"])
        except Exception:
            continue
        total += max(0.0, 1.0 - ang_dev / 60.0) * max(0.0, 1.0 - cv)
    return total


def assess_quality(results, stats, shape, nm_per_px, params, meta, min_rings=10, min_coverage=0.25):
    """Return a list of human-readable problem flags (empty list = looks fine)."""
    flags = list(meta.get("problems", []))
    n = len(results)
    P = period_from_params(params)
    area_nm2 = shape[0] * shape[1] * nm_per_px ** 2
    expected = area_nm2 / (np.sqrt(3) / 2 * P * P)
    cov = n / max(expected, 1.0)
    if n < min_rings:
        flags.append(f"only {n} rings found (min {min_rings})")
    if cov < min_coverage:
        flags.append(f"low coverage: {n} of ~{expected:.0f} expected pores ({cov:.0%})")
    if stats.get("n_567", 0) >= 5 and stats.get("n_traced", 0) / stats["n_567"] < 0.3:
        flags.append("most 5/6/7-sided faces rejected by shape/periodicity filters")
    return flags



_WORKER = {}


def _init_worker(gray, nm_per_px, base_params):
    _WORKER["gray"] = gray
    _WORKER["nm"] = nm_per_px
    _WORKER["base"] = base_params


def _grid_search_worker(overrides):
    p = dict(_WORKER["base"])
    p.update(overrides)
    p["method"] = "tracing"          
    try:
        cands, _ = detect_rings(_WORKER["gray"], _WORKER["nm"], p)
        score, n_found = score_candidates(cands), len(cands)
    except Exception:
        score, n_found = -1.0, 0
    return overrides, score, n_found



BG = "#f0f0f0"
DEBUG_COLORS = {"ok": "lime", "angle": "red", "edgecv": "orange", "periodicity": "dodgerblue",
                "border": "gray"}


class ManualCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Ring Counter v2 (Hex, Penta, Hepta) - preprocess / detect / QC")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            try:
                self.root.attributes('-zoomed', True)
            except tk.TclError:
                pass

        self.root_dir = None
        self.file_list = []
        self.folder_map = {}
        self.original_folder_counts = {}
        self.done_files = set()
        self.review_files = set()
        self.current_idx = 0

        
        self.ring_data = {m: [] for m in MODES}
        self.current_mode = tk.StringVar(value="Hexagon")
        self.dirty = False

        self.raw_original = None     
        self.raw_data = None         
        self.prep_meta = None
        self.current_filename = ""
        self.nm_per_px = None
        self.nm_from_name = False

        self.batch_cancel = False
        self.grid_cancel = False
        self.busy = False
        self.pick_stage = None
        self.pick_p1 = None

        self.best_params_by_file = {}
        self.double_tip_by_file = {}
        self.flags_by_file = {}

        self.v = {}
        self.setup_ui()
        self.bind_shortcuts()

   
    def _hdr(self, text, first=False):
        tk.Label(self.sidebar, text=text if first else "\n" + text,
                 font=('Arial', 10, 'bold'), bg=BG).pack(anchor=tk.W)

    def _note(self, text):
        tk.Label(self.sidebar, text=text, font=('Consolas', 8), bg=BG,
                 justify=tk.LEFT, fg="#555").pack(anchor=tk.W, pady=(2, 0))

    def _slider(self, text, var, lo, hi, res):
        if text:
            tk.Label(self.sidebar, text=text, bg=BG, font=('Consolas', 9)).pack(anchor=tk.W)
        tk.Scale(self.sidebar, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 variable=var, bg=BG, length=250).pack(anchor=tk.W)

    def _check(self, text, var, command=None):
        tk.Checkbutton(self.sidebar, text=text, variable=var, bg=BG, justify=tk.LEFT,
                       anchor='w', command=command, wraplength=320).pack(anchor=tk.W)

    def _periodicity_block(self, title, key, mode, note):
        lo, hi, mg = DEFAULT_WINDOWS[mode]
        self._hdr(f"{title} PERIODICITY TARGET (nm)")
        self.v[f'{key}_min'] = tk.DoubleVar(value=lo)
        self.v[f'{key}_max'] = tk.DoubleVar(value=hi)
        self.v[f'{key}_margin'] = tk.DoubleVar(value=mg * 100.0)
        self._slider("Min periodicity:", self.v[f'{key}_min'], 1.0, 6.0, 0.1)
        self._slider("Max periodicity:", self.v[f'{key}_max'], 1.0, 6.0, 0.1)
        self._slider("Noise margin (% of window, both ends):", self.v[f'{key}_margin'], 0, 50, 1)
        self._note(note)

    def setup_ui(self):
        v = self.v
        self.sidebar_container = tk.Frame(self.root, width=400, bg=BG)
        self.sidebar_container.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_container.pack_propagate(False)
        self.sidebar_canvas = tk.Canvas(self.sidebar_container, bg=BG, highlightthickness=0)
        self.sidebar_scrollbar = tk.Scrollbar(self.sidebar_container, orient="vertical",
                                              command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sidebar = tk.Frame(self.sidebar_canvas, bg=BG, padx=10, pady=10)
        self.sidebar_window_id = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", lambda e: self.sidebar_canvas.configure(
            scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfig(
            self.sidebar_window_id, width=e.width))

        def _wheel(e):
            if e.delta:
                self.sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def _bind_wheel(e):
            self.sidebar_canvas.bind_all("<MouseWheel>", _wheel)
            self.sidebar_canvas.bind_all("<Button-4>", lambda ev: self.sidebar_canvas.yview_scroll(-1, "units"))
            self.sidebar_canvas.bind_all("<Button-5>", lambda ev: self.sidebar_canvas.yview_scroll(1, "units"))

        def _unbind_wheel(e):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.sidebar_canvas.unbind_all(seq)

        self.sidebar_canvas.bind("<Enter>", _bind_wheel)
        self.sidebar_canvas.bind("<Leave>", _unbind_wheel)

        # ---- mode
        self._hdr("MODE SELECTION  (keys 1/2/3)", first=True)
        for mode in MODES:
            key = [k for k, m in MODE_KEYS.items() if m == mode][0]
            tk.Radiobutton(self.sidebar, text=f"{mode}  [{key}]", variable=self.current_mode,
                           value=mode, bg=BG, command=self.update_plot).pack(anchor=tk.W)

        # ---- progress
        self._hdr("FOLDER PROGRESS (scale_png)")
        self.folder_status_box = tk.Text(self.sidebar, width=50, height=11, font=('Consolas', 9),
                                         state='disabled')
        self.folder_status_box.pack(fill=tk.Y, expand=True, pady=6)
        self.scale_label = tk.Label(self.sidebar, text="nm/px: unknown", bg=BG, font=('Consolas', 9))
        self.scale_label.pack(anchor=tk.W)

        # ---- preprocessing
        self._hdr("PREPROCESSING  (then press REPROCESS / R)")
        v['line_align'] = tk.BooleanVar(value=True)
        v['invert'] = tk.BooleanVar(value=False)
        v['affine'] = tk.BooleanVar(value=True)
        v['drift'] = tk.BooleanVar(value=False)
        self._check("Line alignment + tilt removal", v['line_align'])
        self._check("Walls are DARK (invert image)", v['invert'])
        self._check("Global affine correction (drift shear)", v['affine'])
        self._check("Local drift correction (creep, quadratic)", v['drift'])
        self._note("Affine: fits the 3 hex FFT spots, makes the\n"
                   "lattice ideal & area-preserving (nm/px kept).\n"
                   "Local drift: robust polynomial phase fit;\n"
                   "5-7 defects are ignored as outliers.")
        v['show_original'] = tk.BooleanVar(value=False)
        self._check("Show ORIGINAL image (key O; rings hidden)", v['show_original'], command=self.update_plot)
        tk.Button(self.sidebar, text="REPROCESS IMAGE  (R)", command=self.reprocess_current,
                  bg="#2c7fb8", fg="white").pack(anchor=tk.W, pady=4)

        self._hdr("DOUBLE-TIP GHOST REMOVAL")
        v['dt_enable'] = tk.BooleanVar(value=False)
        v['dt_dx'] = tk.DoubleVar(value=0.0)
        v['dt_dy'] = tk.DoubleVar(value=0.0)
        v['dt_amp'] = tk.DoubleVar(value=0.5)
        v['dt_all'] = tk.BooleanVar(value=False)
        self._check("Remove double tip", v['dt_enable'])
        self._slider("Ghost offset dx (px, in raw image):", v['dt_dx'], -150, 150, 0.25)
        self._slider("Ghost offset dy (px, in raw image):", v['dt_dy'], -150, 150, 0.25)
        self._slider("Ghost amplitude (0..1, weaker copy):", v['dt_amp'], 0.05, 0.95, 0.025)
        row = tk.Frame(self.sidebar, bg=BG)
        row.pack(anchor=tk.W, pady=2)
        tk.Button(row, text="PICK GHOST (2 clicks)", command=self.start_pick_ghost).pack(side=tk.LEFT, padx=2)
        tk.Button(row, text="AUTO-ESTIMATE", command=self.auto_estimate_double_tip).pack(side=tk.LEFT, padx=2)
        self._check("Use these settings for ALL images in BATCH\n(otherwise only images where you set them)", v['dt_all'])
        self._note("PICK: show ORIGINAL, click a feature, then its\n"
                   "weaker duplicate. AUTO-ESTIMATE is experimental:\n"
                   "always check the result by eye. Settings you\n"
                   "apply are remembered per image.")

        # ---- detection
        self._hdr("DETECTION METHOD")
        v['method'] = tk.StringVar(value="hybrid")
        for val, text in DETECTION_METHODS:
            tk.Radiobutton(self.sidebar, text=text, variable=v['method'], value=val, bg=BG).pack(anchor=tk.W)
        tk.Label(self.sidebar, text="Wall-mask method:", bg=BG, font=('Consolas', 9)).pack(anchor=tk.W, pady=(4, 0))
        v['mask_method'] = tk.StringVar(value=DEFAULT_MASK_METHOD)
        tk.OptionMenu(self.sidebar, v['mask_method'], *MASK_METHODS).pack(anchor=tk.W)
        self._note("otsu_local: brightness threshold (original).\n"
                   "ridge: Sato ridge filter - robust to drift in\n"
                   "contrast. Grid search tries both.")
        v['exclude_border'] = tk.BooleanVar(value=True)
        self._check("Exclude faces touching the image border", v['exclude_border'])

        self._hdr("JUNCTION MERGE RADIUS (nm)")
        v['merge_nm'] = tk.DoubleVar(value=0.6)
        self._slider("", v['merge_nm'], 0.2, 1.5, 0.05)
        self._note("Junction clusters closer than this are fused.\n"
                   "Was hard-coded 1.2 nm: that is ~80% of a real\n"
                   "bond length at 2.7 nm pores and turns distorted\n"
                   "hexagons into fake pentagons. Lower = safer.")

        self._hdr("ANGLE TOLERANCE (deg from ideal)")
        v['angle_tol'] = tk.DoubleVar(value=55.0)
        self._slider("", v['angle_tol'], 10, 60, 1)
        self._hdr("EDGE REGULARITY TOLERANCE")
        v['edge_tol'] = tk.DoubleVar(value=0.65)
        self._slider("", v['edge_tol'], 0.10, 0.90, 0.05)
        self._note("Max stddev/mean of a face's edge lengths.\n"
                   "(Spike-vertex collapse raises it, so keep\n"
                   "it fairly high.)")
        self._hdr("PORE-CENTRE NEIGHBOUR CUTOFF (x period)")
        v['cutoff'] = tk.DoubleVar(value=1.4)
        self._slider("", v['cutoff'], 1.1, 1.9, 0.05)

        self._periodicity_block("HEXAGON", "hex", "Hexagon",
                                "Flat-to-flat width = pore-centre spacing.\nHard physical filter, +/- noise margin.")
        self._periodicity_block("HEPTAGON", "hept", "Heptagon",
                                "Same idea; defaults match the hexagon window.")
        self._periodicity_block("PENTAGON", "pent", "Pentagon",
                                "Same idea; defaults match the hexagon window.")

        # ---- grid search
        self._hdr("GRID SEARCH")
        v['batch_grid'] = tk.BooleanVar(value=False)
        self._check("Also grid-search each image in BATCH ALL\n(slower; tunes bg/close/junction/mask per image)",
                    v['batch_grid'])
        self._note("Sweeps bg_sigma x close_radius x junction_dilate\n"
                   "x mask_method (90 combos) in parallel and keeps\n"
                   "the best shape-quality score. Saved per image.")

        # ---- quality control
        self._hdr("QUALITY CONTROL (BATCH ALL)")
        v['flag_review'] = tk.BooleanVar(value=True)
        v['min_rings'] = tk.IntVar(value=10)
        v['min_cov'] = tk.DoubleVar(value=25.0)
        self._check("Mark flagged images 'needs review'\ninstead of 'saved'", v['flag_review'])
        self._slider("Min rings found:", v['min_rings'], 0, 100, 1)
        self._slider("Min coverage (% of expected pores):", v['min_cov'], 0, 90, 5)
        self._note("Flags: affine/drift rejected, too few rings,\n"
                   "low coverage, most faces rejected by filters.\n"
                   "Key F jumps to the next flagged image.")

        self._hdr("SHORTCUTS")
        help_txt = ("1/2/3   switch mode\n"
                    "Left click   add point\n"
                    "Right click   remove nearest point\n"
                    "Z            undo last point\n"
                    "A            auto-detect\n"
                    "G            grid-search current image\n"
                    "D            debug stages window\n"
                    "R            reprocess image\n"
                    "O            toggle original view\n"
                    "F            next 'needs review' image\n"
                    "S / Enter    save & next\n"
                    "N / Right    skip (no save)\n"
                    "P / Left     previous file\n"
                    "Delete       delete current file\n"
                    "Esc          cancel batch / grid search")
        tk.Label(self.sidebar, text=help_txt, font=('Consolas', 9), bg=BG, justify=tk.LEFT).pack(anchor=tk.W)

        # ---- plot area
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.info_label = tk.Label(self.plot_frame, text="Load a folder to start...",
                                   font=('Arial', 11, 'bold'), wraplength=1000, justify=tk.LEFT)
        self.info_label.pack(side=tk.TOP, pady=5)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        def mk_buttons(frame, specs):
            for text, cmd, kw in specs:
                tk.Button(frame, text=text, command=cmd, **kw).pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)

        row1 = tk.Frame(self.plot_frame, pady=3)
        row1.pack(fill=tk.X)
        mk_buttons(row1, [
            ("Load Root", self.load_folder, {}),
            ("PREVIOUS", self.prev_file, {}),
            ("REPROCESS", self.reprocess_current, dict(bg="#2c7fb8", fg="white")),
            ("AUTO-DETECT", self.auto_detect, dict(bg="#4a90d9", fg="white")),
            ("GRID SEARCH", self.run_grid_search, dict(bg="#16a085", fg="white")),
            ("DEBUG STAGES", self.show_debug, dict(bg="#7f8c8d", fg="white")),
            ("BATCH ALL", self.batch_auto_detect_all, dict(bg="#8e44ad", fg="white")),
        ])
        row2 = tk.Frame(self.plot_frame, pady=3)
        row2.pack(fill=tk.X)
        mk_buttons(row2, [
            ("NEXT FLAGGED", self.jump_to_next_review, dict(bg="#e67e22", fg="white")),
            ("SKIP", self.skip_file, dict(bg="gray", fg="white")),
            ("UNDO LAST", self.undo_last_point, dict(bg="orange")),
            ("DELETE FILE", self.delete_current_file, dict(bg="red", fg="white")),
            ("SAVE & NEXT", self.save_and_next, dict(bg="green", fg="white", font=('Arial', 10, 'bold'))),
        ])

    def bind_shortcuts(self):
        def bind(keys, fn):
            for k in keys:
                self.root.bind(k, lambda e, f=fn: (None if self.busy else f()))

        for k, m in MODE_KEYS.items():
            bind([f'<Key-{k}>'], lambda m=m: self.set_mode(m))
        bind(['z', 'Z'], self.undo_last_point)
        bind(['a', 'A'], self.auto_detect)
        bind(['g', 'G'], self.run_grid_search)
        bind(['d', 'D'], self.show_debug)
        bind(['r', 'R'], self.reprocess_current)
        bind(['o', 'O'], self.toggle_original)
        bind(['f', 'F'], self.jump_to_next_review)
        bind(['s', 'S', '<Return>'], self.save_and_next)
        bind(['n', 'N', '<Right>'], self.skip_file)
        bind(['p', 'P', '<Left>'], self.prev_file)
        bind(['<Delete>'], self.delete_current_file)
        self.root.bind('<Escape>', lambda e: self.request_cancel())

    def request_cancel(self):
        self.batch_cancel = True
        self.grid_cancel = True
        self.pick_stage = None

    def set_mode(self, mode):
        self.current_mode.set(mode)
        self.update_plot()

    def toggle_original(self):
        self.v['show_original'].set(not self.v['show_original'].get())
        self.update_plot()

    # ------------------------------------------------------ parameters ---
    def collect_params(self, path=None, use_best=True):
        v = self.v
        p = default_params()
        p.update(method=v['method'].get(), mask_method=v['mask_method'].get(),
                 angle_tolerance_deg=float(v['angle_tol'].get()),
                 edge_tolerance=float(v['edge_tol'].get()),
                 pore_neighbor_cutoff=float(v['cutoff'].get()),
                 junction_merge_nm=float(v['merge_nm'].get()),
                 exclude_border=bool(v['exclude_border'].get()))
        wins = {}
        for mode, key in (("Hexagon", "hex"), ("Heptagon", "hept"), ("Pentagon", "pent")):
            lo, hi = sorted((float(v[f'{key}_min'].get()), float(v[f'{key}_max'].get())))
            wins[mode] = (lo, hi, float(v[f'{key}_margin'].get()) / 100.0)
        p["windows"] = wins
        if use_best:
            bp = self.best_params_by_file.get(path or self.current_filename)
            if bp:
                p.update(bg_sigma_nm=bp["bg_sigma_nm"], close_radius_nm=bp["close_radius_nm"],
                         junction_dilate_nm=bp["junction_dilate_nm"],
                         mask_method=bp.get("mask_method", p["mask_method"]))
        return p

    def collect_prep_opts(self, path=None, interactive=False):
        v = self.v
        opts = dict(line_align=bool(v['line_align'].get()), invert=bool(v['invert'].get()),
                    affine=bool(v['affine'].get()), local_drift=bool(v['drift'].get()),
                    double_tip=None)
        cur = (float(v['dt_dx'].get()), float(v['dt_dy'].get()), float(v['dt_amp'].get()))
        if interactive:
            if v['dt_enable'].get():
                opts["double_tip"] = cur
        elif path in self.double_tip_by_file:
            opts["double_tip"] = tuple(self.double_tip_by_file[path])
        elif v['dt_enable'].get() and v['dt_all'].get():
            opts["double_tip"] = cur
        return opts

    def update_info(self):
        if not self.current_filename:
            return
        folder_name = os.path.basename(os.path.dirname(self.current_filename))
        file_name = os.path.basename(self.current_filename)
        pos = f"{self.current_idx + 1}/{len(self.file_list)}"
        if self.current_filename in self.done_files:
            status = "SAVED"
        elif self.current_filename in self.review_files:
            status = "NEEDS REVIEW"
        else:
            status = "pending"
        txt = f"[{pos}] Folder: {folder_name} | File: {file_name} | {status}"
        bp = self.best_params_by_file.get(self.current_filename)
        if bp:
            txt += (f"  |  grid best: bg={bp['bg_sigma_nm']:.2f} close={bp['close_radius_nm']:.2f} "
                    f"jd={bp['junction_dilate_nm']:.2f} mask={bp.get('mask_method', '-')}")
        if self.prep_meta:
            if self.prep_meta["notes"]:
                txt += "\nprep: " + ", ".join(self.prep_meta["notes"])
        flags = self.flags_by_file.get(self.current_filename) or (self.prep_meta or {}).get("problems")
        if flags:
            txt += "\nFLAGS: " + "; ".join(flags)
        self.info_label.config(text=txt)

    # ------------------------------------------------------------ clicks ---
    def on_click(self, event):
        if event.inaxes != self.ax or self.raw_data is None or self.busy:
            return
        if self.pick_stage:
            self._handle_pick(event)
            return
        if self.v['show_original'].get():
            return
        mode = self.current_mode.get()
        if event.button == 1:
            self.ring_data[mode].append({"x": event.xdata, "y": event.ydata, "poly": None,
                                         "source": "manual"})
            self.dirty = True
        elif event.button == 3:
            self.remove_nearest_point(event.xdata, event.ydata)
        self.update_plot()

    def remove_nearest_point(self, x, y):
        pts = self.ring_data[self.current_mode.get()]
        if not pts:
            return
        idx = int(np.argmin([(p["x"] - x) ** 2 + (p["y"] - y) ** 2 for p in pts]))
        pts.pop(idx)
        self.dirty = True

    def undo_last_point(self):
        mode = self.current_mode.get()
        if self.ring_data[mode]:
            self.ring_data[mode].pop()
            self.dirty = True
            self.update_plot()

    def start_pick_ghost(self):
        if self.raw_data is None:
            return
        self.pick_stage = 1
        self.pick_p1 = None
        self.info_label.config(text="DOUBLE-TIP PICK: click a clear feature (e.g. a bright junction). "
                                    "Tip: tick 'Show ORIGINAL' first. ESC cancels.")

    def _handle_pick(self, event):
        if event.button != 1:
            return
        if self.pick_stage == 1:
            self.pick_p1 = (event.xdata, event.ydata)
            self.pick_stage = 2
            self.ax.plot(event.xdata, event.ydata, 'r+', markersize=14)
            self.canvas.draw_idle()
            self.info_label.config(text="DOUBLE-TIP PICK: now click the GHOST - the weaker duplicate of "
                                        "that same feature.")
            return
        d = np.array([event.xdata - self.pick_p1[0], event.ydata - self.pick_p1[1]])
        A = (self.prep_meta or {}).get("affine_A")
        if not self.v['show_original'].get() and A is not None:
            d = np.linalg.inv(A) @ d       
        self.pick_stage = None
        self.v['dt_dx'].set(float(np.clip(d[0], -150, 150)))
        self.v['dt_dy'].set(float(np.clip(d[1], -150, 150)))
        self.v['dt_enable'].set(True)
        self.v['show_original'].set(False)
        self.reprocess_current(force=True)

    def auto_estimate_double_tip(self):
        if self.raw_original is None:
            return
        v = self.v
        period = period_from_params(self.collect_params())
        g = self.raw_original
        if v['line_align'].get():
            g = line_align(g, self.nm_per_px, period)
        if v['invert'].get():
            g = g.max() + g.min() - g
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            est, info = estimate_double_tip(g, self.nm_per_px, period)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Auto-estimate error", str(e))
            return
        self.root.config(cursor="")
        if est is None:
            messagebox.showinfo("Auto-estimate", "No double-tip ghost detected.\n\n" + str(info.get("reason", "")))
            return
        dx, dy, a = est
        v['dt_dx'].set(float(np.clip(dx, -150, 150)))
        v['dt_dy'].set(float(np.clip(dy, -150, 150)))
        v['dt_amp'].set(float(np.clip(a, 0.05, 0.95)))
        v['dt_enable'].set(True)
        self.reprocess_current(force=True)
        messagebox.showinfo("Auto-estimate (experimental)",
                            f"Ghost offset: dx={dx:.2f}px dy={dy:.2f}px, amplitude {a:.2f}\n"
                            f"Lattice symmetry score {info['score_before']:.2f} -> {info['score_after']:.2f}\n\n"
                            f"Compare with the ORIGINAL view (key O) and fine-tune the sliders + REPROCESS.")

    # --------------------------------------------------- load / process ---
    def reprocess_current(self, force=False):
        if self.raw_original is None:
            return
        if not force and not self.confirm_discard_if_dirty():
            return
        opts = self.collect_prep_opts(self.current_filename, interactive=True)
        period = period_from_params(self.collect_params())
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            gray, meta = prepare_image(self.raw_original, self.nm_per_px, opts, period)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Preprocessing error", str(e))
            return
        self.root.config(cursor="")
        if opts["double_tip"]:
            self.double_tip_by_file[self.current_filename] = tuple(opts["double_tip"])
        else:
            self.double_tip_by_file.pop(self.current_filename, None)
        self.raw_data = gray
        self.prep_meta = meta
        self.flags_by_file.pop(self.current_filename, None)
        self.ring_data = {m: [] for m in MODES}
        self.dirty = False
        self.update_plot()
        self.update_info()

    def load_image(self):
        self.current_filename = self.file_list[self.current_idx]
        try:
            raw = load_gray(self.current_filename)
        except Exception as e:
            messagebox.showwarning("Load error", f"Could not read:\n{self.current_filename}\n\n{e}\n\nSkipping.")
            self.next_file(force=True)
            return
        self.raw_original = raw
        self.nm_per_px, self.nm_from_name = resolve_scale(self.current_filename, raw.shape)
        fov = self.nm_per_px * max(raw.shape)
        self.scale_label.config(
            text=(f"nm/px: {self.nm_per_px:.4f}  (FOV {fov:.0f}nm)" if self.nm_from_name else
                  f"nm/px: {self.nm_per_px:.4f}  (ASSUMED 100nm FOV - no 'scale_N' in name)"))
        dt = self.double_tip_by_file.get(self.current_filename)
        if dt:
            self.v['dt_dx'].set(dt[0]); self.v['dt_dy'].set(dt[1]); self.v['dt_amp'].set(dt[2])
            self.v['dt_enable'].set(True)
        elif not self.v['dt_all'].get():
            self.v['dt_enable'].set(False)
        self.pick_stage = None
        self.reprocess_current(force=True)

    
    def auto_detect(self):
        if self.raw_data is None:
            return
        if not self.nm_from_name and not messagebox.askyesno(
                "No scale found",
                "Couldn't parse a 'scale_<nm>' field of view from the filename, so size filtering "
                "will use a default 100nm assumption. Continue anyway?"):
            return
        params = self.collect_params()
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            results, stats = detect_rings(self.raw_data, self.nm_per_px, params)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Auto-detect error", str(e))
            return
        self.root.config(cursor="")
        added = {m: 0 for m in MODES}
        for c in results:
            existing = self.ring_data[c["mode"]]
            if any((p["x"] - c["x"]) ** 2 + (p["y"] - c["y"]) ** 2 < 25 for p in existing):
                continue
            existing.append(dict(c))
            added[c["mode"]] += 1
        self.dirty = True
        flags = assess_quality(results, stats, self.raw_data.shape, self.nm_per_px, params,
                               self.prep_meta, int(self.v['min_rings'].get()),
                               self.v['min_cov'].get() / 100.0)
        self.flags_by_file[self.current_filename] = flags
        self.update_plot()
        self.update_info()
        msg = (f"Added candidates:\n  Hexagons: {added['Hexagon']}\n  Pentagons: {added['Pentagon']}\n"
               f"  Heptagons: {added['Heptagon']}\n\n"
               f"Faces traced: {stats['n_faces']}  (5/6/7-sided: {stats['n_567']}, "
               f"kept: {stats['n_traced']})\n"
               f"Pore-centre detector: {stats['n_centers']} found, {stats['n_centers_added']} added "
               f"where tracing found nothing.\n")
        if flags:
            msg += "\nQUALITY FLAGS:\n  - " + "\n  - ".join(flags) + "\n(try DEBUG STAGES / D)\n"
        msg += "\nThis is a first pass - review with left/right click before saving."
        messagebox.showinfo("Auto-detect complete", msg)

    # ------------------------------------------------------- grid search ---
    def _run_grid_search_pool(self, gray, nm_per_px, base_params, status_prefix=""):
        
        combos = [dict(bg_sigma_nm=b, close_radius_nm=c, junction_dilate_nm=j, mask_method=m)
                  for b in GRID_BG_SIGMA_NM for c in GRID_CLOSE_RADIUS_NM
                  for j in GRID_JUNCTION_DILATE_NM for m in GRID_MASK_METHODS]
        total = len(combos)
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        best, completed = None, 0
        self.grid_cancel = False
        prev_busy = self.busy
        self.busy = True
        try:
            with multiprocessing.Pool(processes=n_workers, initializer=_init_worker,
                                      initargs=(gray, nm_per_px, base_params)) as pool:
                for ov, score, n_found in pool.imap_unordered(_grid_search_worker, combos, chunksize=1):
                    completed += 1
                    if score > 0 and (best is None or score > best["score"]):
                        best = dict(ov, score=score, n_found=n_found)
                    self.info_label.config(
                        text=f"{status_prefix}GRID SEARCH: {completed}/{total} combos  |  best score so far: "
                             f"{(('%.2f' % best['score']) if best else 'n/a')}  [ESC to cancel]")
                    self.root.update_idletasks()
                    self.root.update()
                    if self.grid_cancel:
                        pool.terminate()
                        break
        except Exception as e:
            self.busy = prev_busy
            messagebox.showerror("Grid Search error", str(e))
            return None
        self.busy = prev_busy
        return None if self.grid_cancel else best

    def run_grid_search(self):
        if self.raw_data is None:
            messagebox.showinfo("Grid Search", "Load an image first.")
            return
        if not self.confirm_discard_if_dirty():
            return
        base = self.collect_params(use_best=False)
        best = self._run_grid_search_pool(self.raw_data, self.nm_per_px, base)
        self.update_info()
        if best is None:
            messagebox.showinfo("Grid Search", "No valid parameter combination produced results "
                                               "(or the search was cancelled).")
            return
        self.best_params_by_file[self.current_filename] = best
        params = self.collect_params()
        try:
            results, stats = detect_rings(self.raw_data, self.nm_per_px, params)
        except Exception as e:
            messagebox.showerror("Grid Search error", f"Best params found but re-run failed: {e}")
            return
        self.ring_data = {m: [] for m in MODES}
        for c in results:
            self.ring_data[c["mode"]].append(dict(c))
        self.dirty = True
        self.flags_by_file[self.current_filename] = assess_quality(
            results, stats, self.raw_data.shape, self.nm_per_px, params, self.prep_meta,
            int(self.v['min_rings'].get()), self.v['min_cov'].get() / 100.0)
        self.update_plot()
        self.update_info()
        messagebox.showinfo(
            "Grid Search complete",
            f"Best parameters found:\n  bg_sigma_nm = {best['bg_sigma_nm']:.2f}\n"
            f"  close_radius_nm = {best['close_radius_nm']:.2f}\n"
            f"  junction_dilate_nm = {best['junction_dilate_nm']:.2f}\n"
            f"  mask_method = {best['mask_method']}\n"
            f"  score = {best['score']:.2f}  ({best['n_found']} traced rings)\n\n"
            f"Detection (method: {params['method']}) re-run with these. Review before saving.")

    # ------------------------------------------------------------ debug ---
    def show_debug(self):
        
        if self.raw_data is None:
            return
        params = self.collect_params()
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            _, tdbg = detect_by_tracing(self.raw_data, self.nm_per_px, params, return_debug=True)
            _, cdbg = detect_by_centers(self.raw_data, self.nm_per_px, params)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Debug error", str(e))
            return
        self.root.config(cursor="")
        top = tk.Toplevel(self.root)
        top.title("Detection debug - stage by stage")
        fig = Figure(figsize=(15, 8.6))
        axs = fig.subplots(2, 3).ravel()
        st = tdbg["stats"]
        axs[0].imshow(self.raw_original, cmap='gray'); axs[0].set_title("1  original")
        axs[1].imshow(self.raw_data, cmap='gray')
        axs[1].set_title("2  preprocessed: " + (", ".join(self.prep_meta["notes"]) or "none"), fontsize=9)
        axs[2].imshow(tdbg["flat"], cmap='gray'); axs[2].set_title("3  background-flattened")
        axs[3].imshow(tdbg["mask"], cmap='gray'); axs[3].set_title(f"4  wall mask ({params['mask_method']})")
        axs[4].imshow(self.raw_data, cmap='gray', alpha=0.6)
        sy, sx = np.where(tdbg["skel"])
        axs[4].scatter(sx, sy, s=1, c='cyan')
        jy, jx = np.where(tdbg["junctions"])
        axs[4].scatter(jx, jy, s=4, c='red')
        axs[4].set_title("5  skeleton (cyan) + junctions (red)")
        axs[5].imshow(self.raw_data, cmap='gray')
        for f in tdbg["faces"]:
            col = DEBUG_COLORS.get(f["status"])
            if col is None:
                axs[5].add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor='white',
                                            linewidth=0.4, alpha=0.5))
            else:
                axs[5].add_patch(MplPolygon(f["poly"], closed=True, fill=True, facecolor=col,
                                            edgecolor=col, alpha=0.4, linewidth=1))
        if len(cdbg["centers"]):
            axs[5].plot(cdbg["centers"][:, 0], cdbg["centers"][:, 1], 'w.', markersize=3)
        axs[5].set_title(f"6  faces: {st['n_faces']} traced, {st['n_567']} are 5/6/7-sided, "
                         f"{st['n_traced']} kept", fontsize=9)
        for a in axs:
            a.axis('off')
        fig.text(0.5, 0.005,
                 "green = kept | red = angle check failed | orange = edge-length check failed | "
                 "blue = outside periodicity window | gray = touches border | "
                 f"white dots = pore centres ({len(cdbg['centers'])})",
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        cv = FigureCanvasTkAgg(fig, master=top)
        cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------- batch mode ---
    def analyze_file(self, fpath, run_grid=False, status_prefix=""):
        
        raw = load_gray(fpath)
        nm_per_px, from_name = resolve_scale(fpath, raw.shape)
        params = self.collect_params(fpath)
        opts = self.collect_prep_opts(fpath)
        gray, meta = prepare_image(raw, nm_per_px, opts, period_from_params(params))
        if run_grid:
            best = self._run_grid_search_pool(gray, nm_per_px, self.collect_params(fpath, use_best=False),
                                              status_prefix)
            if self.grid_cancel:
                self.batch_cancel = True
                return None
            if best is not None:
                self.best_params_by_file[fpath] = best
                params = self.collect_params(fpath)
        results, stats = detect_rings(gray, nm_per_px, params)
        flags = assess_quality(results, stats, gray.shape, nm_per_px, params, meta,
                               int(self.v['min_rings'].get()), self.v['min_cov'].get() / 100.0)
        return dict(raw=raw, gray=gray, meta=meta, nm_per_px=nm_per_px, from_name=from_name,
                    results=results, stats=stats, flags=flags, params=params)

    def batch_auto_detect_all(self):
        
        if not self.file_list:
            messagebox.showinfo("Batch Auto-Detect", "Load a root folder first.")
            return
        if not self.confirm_discard_if_dirty():
            return
        reprocess = messagebox.askyesno(
            "Batch Auto-Detect - scope",
            "Also reprocess files already marked SAVED / NEEDS REVIEW (overwriting their results)?\n\n"
            "Yes = process every image\nNo = only process pending images")
        targets = [f for f in self.file_list
                   if reprocess or (f not in self.done_files and f not in self.review_files)]
        if not targets:
            messagebox.showinfo("Batch Auto-Detect", "Nothing to process (no pending files).")
            return
        p = self.collect_params()
        o = self.collect_prep_opts()
        pre = [n for n, on in (("line-align", o['line_align']), ("invert", o['invert']),
                               ("affine", o['affine']), ("local drift", o['local_drift'])) if on]
        if not messagebox.askyesno(
                "Confirm batch run",
                f"About to auto-detect and SAVE results for {len(targets)} image(s) with NO manual review.\n\n"
                f"  method: {p['method']}, mask: {p['mask_method']}\n"
                f"  preprocessing: {', '.join(pre) or 'none'}\n"
                f"  grid search per image: {'ON (much slower)' if self.v['batch_grid'].get() else 'off'}\n"
                f"  flagged images -> {'NEEDS REVIEW' if self.v['flag_review'].get() else 'saved anyway'}\n\n"
                f"Press ESC during the run to stop early.\n\nProceed?"):
            return

        self.batch_cancel = False
        self.busy = True
        self.v['show_original'].set(False)
        total = len(targets)
        tally = {m: 0 for m in MODES}
        processed, flagged = 0, []
        try:
            for i, fpath in enumerate(targets):
                if self.batch_cancel:
                    break
                self.info_label.config(text=f"BATCH ({i + 1}/{total}) processing: "
                                            f"{os.path.basename(fpath)}  [ESC to cancel]")
                self.root.update_idletasks()
                self.root.update()
                try:
                    res = self.analyze_file(fpath, run_grid=self.v['batch_grid'].get(),
                                            status_prefix=f"BATCH ({i + 1}/{total}) ")
                except Exception as e:
                    print(f"Skipping {fpath}: {e}")
                    continue
                if res is None:
                    break
                ring_data = {m: [] for m in MODES}
                for c in res["results"]:
                    ring_data[c["mode"]].append(dict(c))
                    tally[c["mode"]] += 1
                # show live
                self.current_filename = fpath
                self.raw_original, self.raw_data = res["raw"], res["gray"]
                self.prep_meta, self.nm_per_px = res["meta"], res["nm_per_px"]
                self.nm_from_name = res["from_name"]
                self.ring_data = ring_data
                self.flags_by_file[fpath] = res["flags"]
                self.scale_label.config(text=f"nm/px: {self.nm_per_px:.4f}")
                self.update_plot()
                self.root.update_idletasks()
                self.root.update()

                needs_review = bool(res["flags"]) and self.v['flag_review'].get()
                status = "NEEDS_REVIEW" if needs_review else "SAVED"
                self.write_outputs(fpath, ring_data, res["meta"], res["nm_per_px"], res["flags"],
                                   self.best_params_by_file.get(fpath), res["params"], status)
                if needs_review:
                    self.review_files.add(fpath)
                    self.done_files.discard(fpath)
                    flagged.append(os.path.basename(fpath))
                else:
                    self.done_files.add(fpath)
                    self.review_files.discard(fpath)
                processed += 1
        finally:
            self.busy = False
        self.dirty = False
        self.save_progress()
        self.update_sidebar()
        self.update_info()
        note = " (cancelled early)" if self.batch_cancel else ""
        fl = ""
        if flagged:
            fl = (f"\n{len(flagged)} image(s) FLAGGED -> NEEDS REVIEW (press F to jump to them):\n  "
                  + "\n  ".join(flagged[:12]) + ("\n  ..." if len(flagged) > 12 else ""))
        messagebox.showinfo(
            "Batch Auto-Detect complete",
            f"Processed {processed}/{total} image(s){note}.\n\nTotal detected across batch:\n"
            f"  Hexagons: {tally['Hexagon']}\n  Pentagons: {tally['Pentagon']}\n"
            f"  Heptagons: {tally['Heptagon']}\n{fl}\n\n"
            f"Unflagged images were saved WITHOUT manual review - spot-check a sample.")

    # ------------------------------------------------------------- plot ---
    def update_plot(self):
        if self.raw_data is None:
            return
        self.ax.clear()
        showing_orig = self.v['show_original'].get() and self.raw_original is not None
        self.ax.imshow(self.raw_original if showing_orig else self.raw_data, cmap='gray')
        if showing_orig:
            self.ax.set_title("ORIGINAL (unprocessed) view - rings hidden")
            self.canvas.draw()
            return
        for mode, points in self.ring_data.items():
            col = MODE_COLORS[mode]
            for i, p in enumerate(points):
                if p.get("poly"):
                    self.ax.add_patch(MplPolygon(p["poly"], closed=True, fill=True, facecolor=col,
                                                 edgecolor=col, alpha=0.35, linewidth=1.5))
                self.ax.plot(p["x"], p["y"], 'o', color=col, markersize=4)
                label = f"{mode[0]}{i + 1}"
                if mode == "Heptagon":
                    label = f"Hp{i + 1}"
                if p.get("periodicity_nm") is not None:
                    label += f" {p['periodicity_nm']:.2f}nm"
                if p.get("source") == "centers":
                    label += "*"
                self.ax.text(p["x"] + 2, p["y"] - 2, label, color=col, fontsize=9, fontweight='bold')
        self.ax.set_title(f"Mode: {self.current_mode.get()}  |  "
                          f"H:{len(self.ring_data['Hexagon'])} P:{len(self.ring_data['Pentagon'])} "
                          f"Hp:{len(self.ring_data['Heptagon'])}   (* = found by pore-centre detector)")
        if self.pick_stage == 2 and self.pick_p1:
            self.ax.plot(self.pick_p1[0], self.pick_p1[1], 'r+', markersize=14)
        self.canvas.draw()

    # ---------------------------------------------------------- sidebar ---
    def update_sidebar(self):
        self.folder_status_box.config(state='normal')
        self.folder_status_box.delete('1.0', tk.END)
        for folder, files in self.folder_map.items():
            parent = os.path.basename(os.path.dirname(folder))
            total = self.original_folder_counts.get(folder, len(files))
            done = sum(1 for f in files if f in self.done_files)
            rev = sum(1 for f in files if f in self.review_files)
            extra = f"   Review: {rev}" if rev else ""
            self.folder_status_box.insert(tk.END, f"{parent}/scale_png\n  Done: {done} / {total}{extra}\n\n")
        self.folder_status_box.config(state='disabled')

    
    def progress_path(self):
        return os.path.join(self.root_dir, PROGRESS_FILENAME)

    @staticmethod
    def _read_status(txt_path):
        try:
            with open(txt_path, 'r') as f:
                return "NEEDS_REVIEW" if "STATUS: NEEDS_REVIEW" in f.read() else "SAVED"
        except Exception:
            return "SAVED"

    def load_progress(self):
        self.done_files, self.review_files = set(), set()
        self.best_params_by_file, self.double_tip_by_file = {}, {}
        p = self.progress_path()
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                self.done_files = set(data.get("done_files", []))
                self.review_files = set(data.get("review_files", []))
                self.best_params_by_file = dict(data.get("best_params", {}))
                self.double_tip_by_file = {k: tuple(v) for k, v in data.get("double_tip", {}).items()}
            except Exception:
                pass
        for folder, files in self.folder_map.items():
            out_dir = os.path.join(folder, "Analysis_Results")
            for f in files:
                base = os.path.splitext(os.path.basename(f))[0]
                txt = os.path.join(out_dir, f"{base}_ANALYZED.txt")
                if os.path.exists(txt) and f not in self.done_files and f not in self.review_files:
                    if self._read_status(txt) == "NEEDS_REVIEW":
                        self.review_files.add(f)
                    else:
                        self.done_files.add(f)

    def save_progress(self):
        if not self.root_dir:
            return
        try:
            with open(self.progress_path(), 'w') as f:
                json.dump({"done_files": sorted(self.done_files),
                           "review_files": sorted(self.review_files),
                           "best_params": self.best_params_by_file,
                           "double_tip": {k: list(v) for k, v in self.double_tip_by_file.items()}},
                          f, indent=2)
        except Exception as e:
            print("Could not save progress file:", e)

    
    def load_folder(self):
        root_dir = filedialog.askdirectory()
        if not root_dir:
            return
        self.root_dir = root_dir
        self.file_list, self.folder_map, self.original_folder_counts = [], {}, {}
        for r, _, fs in os.walk(root_dir):
            if os.path.basename(r) == 'scale_png':
                imgs = sorted(os.path.join(r, f) for f in fs
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
                              and "ANALYZED" not in f.upper())
                if imgs:
                    self.folder_map[r] = imgs
                    self.original_folder_counts[r] = len(imgs)
                    self.file_list.extend(imgs)
        if not self.file_list:
            messagebox.showinfo("Info", "No images found inside 'scale_png' folders.")
            return
        self.load_progress()
        self.update_sidebar()
        self.current_idx = next((i for i, f in enumerate(self.file_list)
                                 if f not in self.done_files), 0)
        self.load_image()

    def confirm_discard_if_dirty(self):
        if not self.dirty:
            return True
        return messagebox.askyesno("Unsaved changes", "You have unsaved points on this image. Discard them?")

    # ------------------------------------------------------------- save ---
    def write_outputs(self, fpath, ring_data, meta, nm_per_px, flags, best, params, status):
        
        out_dir = os.path.join(os.path.dirname(fpath), "Analysis_Results")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(fpath))[0]
        raw_xy = {m: [to_original_coords(p["x"], p["y"], meta) for p in pts]
                  for m, pts in ring_data.items()}

        with open(os.path.join(out_dir, f"{base}_ANALYZED.txt"), "w") as f:
            for mode, pts in ring_data.items():
                f.write(f"{mode}s counted: {len(pts)}\n")
                for j, p in enumerate(pts):
                    extra = ""
                    if p.get("periodicity_nm") is not None:
                        area = f"{p['area_nm2']:.3f}" if p.get("area_nm2") is not None else "n/a"
                        extra = f"  (periodicity={p['periodicity_nm']:.3f}nm, area={area}nm^2)"
                    f.write(f"  {j + 1}: {p['x']:.2f}, {p['y']:.2f}{extra}\n")
            if best:
                f.write("\nBest Parameters (Grid Search):\n")
                f.write(f"  bg_sigma_nm = {best['bg_sigma_nm']:.4f}\n")
                f.write(f"  close_radius_nm = {best['close_radius_nm']:.4f}\n")
                f.write(f"  junction_dilate_nm = {best['junction_dilate_nm']:.4f}\n")
                f.write(f"  mask_method = {best.get('mask_method', DEFAULT_MASK_METHOD)}\n")
                f.write(f"  score = {best['score']:.4f}\n")
                f.write(f"  n_found = {best['n_found']}\n")
            f.write("\nPreprocessing: " + (", ".join(meta["notes"]) or "none") + "\n")
            f.write("  (coordinates above are in the CORRECTED frame = the saved PNG)\n")
            f.write("\nRaw-image coordinates (before affine/drift correction):\n")
            for mode, pts in ring_data.items():
                f.write(f"{mode}s:\n")
                for j, (rx, ry) in enumerate(raw_xy[mode]):
                    f.write(f"  {j + 1}: {rx:.2f}, {ry:.2f}\n")
            if flags:
                f.write("\nQuality flags: " + "; ".join(flags) + "\n")
            f.write(f"\nSTATUS: {status}\n")

        rings = []
        for mode, pts in ring_data.items():
            for p, (rx, ry) in zip(pts, raw_xy[mode]):
                rings.append({"mode": mode, "x": p["x"], "y": p["y"], "raw_x": rx, "raw_y": ry,
                              "periodicity_nm": p.get("periodicity_nm"), "area_nm2": p.get("area_nm2"),
                              "source": p.get("source", "auto")})
        A = meta.get("affine_A")
        data = {"file": fpath, "status": status, "nm_per_px": nm_per_px, "flags": flags,
                "preprocessing": meta["notes"], "affine_A": None if A is None else np.asarray(A).tolist(),
                "counts": {m: len(pts) for m, pts in ring_data.items()},
                "params": params, "grid_search_best": best, "rings": rings}
        with open(os.path.join(out_dir, f"{base}_ANALYZED.json"), "w") as f:
            json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        self.fig.savefig(os.path.join(out_dir, f"{base}_ANALYZED.png"), dpi=200)

    def save_and_next(self):
        if self.raw_data is None:
            return
        self.v['show_original'].set(False)      # the PNG must show the processed image + overlay
        self.update_plot()
        path = self.current_filename
        self.write_outputs(path, self.ring_data, self.prep_meta, self.nm_per_px,
                           self.flags_by_file.get(path, []), self.best_params_by_file.get(path),
                           self.collect_params(), "SAVED")
        self.done_files.add(path)
        self.review_files.discard(path)
        self.save_progress()
        self.dirty = False
        self.update_sidebar()
        self.next_file(force=True)

    def skip_file(self):
        if not self.confirm_discard_if_dirty():
            return
        self.next_file(force=True)

    def prev_file(self):
        if not self.confirm_discard_if_dirty():
            return
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def jump_to_next_review(self):
        if not self.review_files:
            messagebox.showinfo("Needs review", "No images are currently flagged 'needs review'.")
            return
        if not self.confirm_discard_if_dirty():
            return
        n = len(self.file_list)
        for k in range(1, n + 1):
            i = (self.current_idx + k) % n
            if self.file_list[i] in self.review_files:
                self.current_idx = i
                self.load_image()
                return

    def delete_current_file(self):
        if not self.file_list:
            return
        if not messagebox.askyesno("Confirm delete", f"Permanently delete:\n{self.current_filename}\n?"):
            return
        folder = os.path.dirname(self.current_filename)
        target = self.current_filename
        try:
            if os.path.exists(target):
                os.remove(target)
        except Exception as e:
            messagebox.showerror("Delete error", str(e))
            return
        self.file_list.pop(self.current_idx)
        if target in self.folder_map.get(folder, []):
            self.folder_map[folder].remove(target)
        for d in (self.done_files, self.review_files):
            d.discard(target)
        for d in (self.best_params_by_file, self.double_tip_by_file, self.flags_by_file):
            d.pop(target, None)
        self.save_progress()
        self.update_sidebar()
        if not self.file_list:
            self.root.destroy()
            return
        if self.current_idx >= len(self.file_list):
            self.current_idx -= 1
        self.load_image()

    def next_file(self, force=False):
        if not force and not self.confirm_discard_if_dirty():
            return
        self.current_idx += 1
        if self.current_idx < len(self.file_list):
            self.load_image()
        else:
            messagebox.showinfo("Done", "Reached the end of the file list.")
            self.current_idx = len(self.file_list) - 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = ManualCounter(root)
    root.mainloop()