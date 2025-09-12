# ----------------- Migration Image Analysis (PFAS, 2 doses/run) -----------------
import io, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Migration Image Analysis", layout="wide")

TIMEPOINTS = [0, 24, 48, 72]  # fixed
DOSE_CHOICES_NM = [1, 5, 10, 20, 40, 100, 500]
CELL_LINES = ["RCC", "Renca"]
COMPOUNDS = ["GenX", "PFOA", "PFOS"]
IMG_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]  # <— TIFF support

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ----------------------------- UTILITIES -------------------------------
@st.cache_data(show_spinner=False)
def _downscale(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

def illumination_correct(gray: np.ndarray, sigma_bg: float) -> np.ndarray:
    """Divide by heavy gaussian to flatten illumination; rescale to [0,1]."""
    bg = gaussian(gray, sigma=sigma_bg, preserve_range=True)
    corr = gray / np.clip(bg, 1e-6, None)
    corr = exposure.rescale_intensity(corr, in_range='image', out_range=(0, 1))
    return corr

def local_std(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Local texture std via Gaussian moments."""
    m1 = gaussian(gray, sigma=sigma, preserve_range=True)
    m2 = gaussian(gray * gray, sigma=sigma, preserve_range=True)
    var = np.clip(m2 - m1 * m1, 0, None)
    std = np.sqrt(var)
    std = std / (std.max() + 1e-8)
    return std

def center_roi_mask(h: int, w: int, margin_frac: float) -> np.ndarray:
    """Centered ROI; trims margins on all sides. margin=0 → full image."""
    r0, r1 = int(h * margin_frac), int(h * (1 - margin_frac))
    c0, c1 = int(w * margin_frac), int(w * (1 - margin_frac))
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m

def mask_scale_bar(h: int, w: int, width_frac: float, height_frac: float, inset_frac: float = 0.02) -> np.ndarray:
    """Mask a bottom-right rectangle (scale bar area)."""
    if width_frac <= 0 or height_frac <= 0:
        return np.zeros((h, w), dtype=bool)
    bw = int(w * width_frac); bh = int(h * height_frac)
    r1 = int(h * (1 - inset_frac)); r0 = max(0, r1 - bh)
    c1 = int(w * (1 - inset_frac)); c0 = max(0, c1 - bw)
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.38, color=(0, 180, 255)) -> np.ndarray:
    out = rgb.copy()
    col = np.zeros_like(out); col[..., 0], col[..., 1], col[..., 2] = color
    out = (out * (1 - alpha) + col * alpha * mask[..., None]).astype(np.uint8)
    return out

def segment_open(gray01: np.ndarray, std_sigma: float, sens: float,
                 open_r: int, close_r: int, min_area: int, open_class: str) -> np.ndarray:
    """
    Texture segmentation. open_class:
      - 'low'  -> open = low texture (recommended for phase contrast)
      - 'high' -> open = high texture
    """
    s = local_std(gray01, sigma=std_sigma)
    thr = threshold_otsu(s)
    thr = thr * (1.0 + sens)  # shift threshold
    mask = (s <= thr) if open_class == "low" else (s >= thr)

    if open_r > 0:
        mask = binary_opening(mask, footprint=disk(open_r))
    if close_r > 0:
        mask = binary_closing(mask, footprint=disk(close_r))
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)
    return mask

def load_uploaded_image(file_obj):
    """
    Robust loader for PNG/JPG/TIFF.
    - Scales 16-bit or float images to 8-bit
    - If multi-page TIFF, returns the first frame (noted in caption)
    Returns (pil_rgb_image, n_frames, is_tiff)
    """
    im = Image.open(file_obj)
    is_tiff = (im.format or "").upper() == "TIFF"
    n_frames = int(getattr(im, "n_frames", 1))
    try:
        im.seek(0)  # first frame
    except Exception:
        pass

    arr = np.array(im)

    # Scale to uint8 if needed (e.g., 16-bit TIFF)
    if arr.dtype != np.uint8:
        arr = exposure.rescale_intensity(arr, in_range="image", out_range=(0, 255)).astype(np.uint8)

    # Ensure 3-channel RGB
    if arr.ndim == 2:
        pil_rgb = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil_rgb = Image.fromarray(arr).convert("RGB")

    return pil_rgb, n_frames, is_tiff

def analyze_image(pil_image: Image.Image, roi_margin: float, bg_sigma: float,
                  std_sigma: float, sens: float, open_r: int, close_r: int,
                  min_area: int, open_class: str, sbw: float, sbh: float):
    """Return (raw_open_pct, overlay_png_bytes)."""
    rgb = np.array(pil_image.convert("RGB"))
    rgb = _downscale(rgb, max_side=1600)
    gray = rgb2gray(rgb).astype(np.float32)

    corr = illumination_correct(gray, sigma_bg=bg_sigma)
    mask_open = segment_open(corr, std_sigma=std_sigma, sens=sens,
                             open_r=open_r, close_r=close_r, min_area=min_area,
                             open_class=open_class)

    h, w = mask_open.shape
    roi = center_roi_mask(h, w, roi_margin)
    sb  = mask_scale_bar(h, w, width_frac=sbw, height_frac=sbh)
    keep = roi & ~sb
    keep_pix = int(keep.sum())
    valid = mask_open & keep
    raw_open_pct = 100.0 * (valid.sum() / max(1, keep_pix))

    base = (corr * 255).astype(np.uint8)
    base_rgb = np.repeat(base[..., None], 3, axis=2)
    overlay = overlay_mask(base_rgb, valid, alpha=0.42, color=(0, 180, 255))

    rr0, rr1 = int(h * roi_margin), int(h * (1 - roi_margin))
    cc0, cc1 = int(w * roi_margin), int(w * (1 - roi_margin))
    overlay[rr0:rr1, [cc0, cc1 - 1]] = (0, 255, 90)
    overlay[[rr0, rr1 - 1], cc0:cc1] = (0, 255, 90)
    overlay[sb] = (200, 200, 200)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return raw_open_pct, buf.getvalue()

def summarize_series(raw_by_t: dict):
    """Return DataFrame with Raw, Relative Open %, Closure % (baseline = 0h)."""
    baseline = raw_by_t.get(0, np.nan)
    rows = []
    for t in sorted(raw_by_t.keys()):
        raw = raw_by_t[t]
        rel_open = (raw / baseline) * 100.0 if (baseline == baseline) else np.nan
        closure  = 100.0 - rel_open if (rel_open == rel_open) else np.nan
        rows.append({"Hours": t, "Raw Open %": raw, "Relative Open %": rel_open, "Closure %": closure})
    return pd.DataFrame(rows).set_index("Hours"), baseline

def _try_load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def annotate_bytes(img_bytes: bytes, text: str, corner: str = "br", scale: float = 0.035,
                   fg=(255, 221, 0, 255), shadow=(0, 0, 0, 255)):
    """Draw a high-contrast, large label onto PNG bytes."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    fsize = max(16, int(W * scale))
    font = _try_load_font(fsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(6, fsize // 4)
    x0, y0 = W - tw - 2 * pad - 8, H - th - 2 * pad - 8
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 150))
    tx, ty = x0 + pad, y0 + pad
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
        draw.text((tx+dx, ty+dy), text, fill=shadow, font=font)
    draw.text((tx, ty), text, fill=fg, font=font)
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()

# ------------------------------- UI -----------------------------------
st.title("Migration Image Analysis")

# === Study metadata & processing controls (sidebar) ===
with st.sidebar:
    st.header("Experiment")
    cell_line = st.selectbox("Cell line", CELL_LINES, index=0)
    compound  = st.selectbox("Compound", COMPOUNDS, index=0)

    doses = st.multiselect(
        "Select exactly two doses (nM)", DOSE_CHOICES_NM, default=[10, 40],
        help="One run = one cell line × compound with two doses across 0/24/48/72h"
    )
    if len(doses) != 2:
        st.warning("Please pick exactly two doses.", icon="⚠️")

    st.header("Processing")
    roi_margin = st.slider("ROI margin (0 = full image)", 0.00, 0.25, 0.00, 0.01,
                           help="Crop edges to avoid artefacts; set 0 for full-frame")
    bg_sigma   = st.slider("BG sigma", 10.0, 80.0, 42.0, 2.0, help="Illumination correction blur")
    std_sigma  = st.slider("Texture sigma", 3.0, 30.0, 12.0, 1.0, help="Neighborhood size for local std")
    sens       = st.slider("Sensitivity", -0.35, 0.35, 0.00, 0.01, help="Lower→more open; Higher→less open")
    cleanup    = st.selectbox("Cleanup", ["Light (2/2)","Med (3/3)","Strong (4/3)","Custom (1/1)"], index=1)
    map_r      = {"Light (2/2)":(2,2), "Med (3/3)":(3,3), "Strong (4/3)":(4,3), "Custom (1/1)":(1,1)}
    open_r, close_r = map_r[cleanup]
    open_mode  = st.selectbox("Open class", ["Low texture","High texture","Auto"], index=0,
                              help="If results look inverted, try 'High' or 'Auto'")
    sb_mask    = st.checkbox("Mask scale bar", value=False)
    sb_width   = st.slider("Scale-bar width (frac)", 0.00, 0.30, 0.12, 0.01, disabled=not sb_mask)
    sb_height  = st.slider("Scale-bar height (frac)", 0.00, 0.20, 0.06, 0.01, disabled=not sb_mask)

# === Upload grid: 2 doses × 4 timepoints ===
st.markdown("#### Upload images (2 doses × 4 time points)")
uploads = {}  # {dose_nm: {t: PIL.Image}}
if len(doses) == 2:
    row1 = st.columns(4)
    row2 = st.columns(4)
    dose_to_cols = {doses[0]: row1, doses[1]: row2}
    for dose, cols in dose_to_cols.items():
        st.markdown(f"**Dose {dose} nM**")
        for t, col in zip(TIMEPOINTS, cols):
            with col:
                f = st.file_uploader(f"{t} h — {dose} nM", type=IMG_TYPES, key=f"{dose}_{t}")
                if f:
                    try:
                        img, n_frames, is_tiff = load_uploaded_image(f)
                        uploads.setdefault(dose, {})[t] = img
                        note = f" (TIFF, showing 1/{n_frames})" if is_tiff and n_frames > 1 else ""
                        st.image(img, caption=f"{t} h — {dose} nM{note}", use_container_width=True)
                    except Exception:
                        st.error("Could not read image.")

st.divider()
go = st.button("▶️ Analyze", type="primary", use_container_width=True)

# ------------------------------ ANALYSIS ------------------------------
def run_series(images_by_t, dose_nm, color_hex):
    """Analyze a set of 0/24/48/72 images for one dose."""
    raw, overlays = {}, {}
    mode = "low" if open_mode == "Low texture" else ("high" if open_mode == "High texture" else "low")
    for t in sorted(images_by_t.keys()):
        val, ov = analyze_image(images_by_t[t],
            roi_margin=roi_margin, bg_sigma=bg_sigma,
            std_sigma=std_sigma, sens=sens,
            open_r=open_r, close_r=close_r, min_area=600,
            open_class=mode, sbw=(sb_width if sb_mask else 0.0),
            sbh=(sb_height if sb_mask else 0.0)
        )
        raw[t], overlays[t] = val, ov

    # Auto-flip if later timepoints look more open than baseline
    if open_mode == "Auto" and 0 in raw and len(raw) > 1:
        later = [raw[t] for t in raw if t != 0 and not np.isnan(raw[t])]
        if later and np.nanmedian(later) > raw[0]:
            raw, overlays = {}, {}
            for t in sorted(images_by_t.keys()):
                val, ov = analyze_image(images_by_t[t],
                    roi_margin=roi_margin, bg_sigma=bg_sigma,
                    std_sigma=std_sigma, sens=sens,
                    open_r=open_r, close_r=close_r, min_area=600,
                    open_class="high", sbw=(sb_width if sb_mask else 0.0),
                    sbh=(sb_height if sb_mask else 0.0)
                )
                raw[t], overlays[t] = val, ov

    df, baseline = summarize_series(raw)
    # annotate overlays with dose + metrics
    ann = {}
    for t in overlays:
        rel = df.loc[t, "Relative Open %"] if t in df.index else np.nan
        clo = df.loc[t, "Closure %"] if t in df.index else np.nan
        label = f"{t} h — {dose_nm} nM\nOpen {raw[t]:.2f}%"
        if rel == rel:
            label += f" | Rel {rel:.1f}% | Close {clo:.1f}%"
        ann[t] = annotate_bytes(overlays[t], label, corner="br", scale=0.040, fg=(255,221,0,255))
    return df, baseline, ann, color_hex

if go and len(doses) == 2 and any(uploads.values()):
    # assign distinct colors for the two doses
    dose_colors = {doses[0]: OKABE_ITO[0], doses[1]: OKABE_ITO[1]}
    results = {}
    for dose in doses:
        if dose in uploads and uploads[dose]:
            results[dose] = run_series(uploads[dose], dose, dose_colors[dose])

    if not results:
        st.warning("Please upload at least one image.", icon="ℹ️")
    else:
        left, right = st.columns([1, 1])

        # --- Left: Overlays in a 2×4 grid ---
        with left:
            st.markdown(f"#### Detection overlays — **{cell_line} / {compound}**")
            g = st.columns(4)
            # row for each dose
            for row_i, dose in enumerate(doses):
                st.markdown(f"**Dose {dose} nM**")
                for j, t in enumerate(TIMEPOINTS):
                    img_bytes = results.get(dose, (None,None,{},None))[2].get(t)
                    if img_bytes is not None:
                        with g[j]:
                            st.image(img_bytes, use_container_width=True)
                    else:
                        with g[j]:
                            st.info(f"No image ({t} h)")

        # --- Right: Combined table + plot ---
        with right:
            st.markdown(f"#### 📊 Baseline-normalized results — **{cell_line} / {compound}**")
            # build long table Dose, Hours, Raw/Rel/Closure
            tbl_rows = []
            for dose in doses:
                if dose in results:
                    df, base, _, _ = results[dose]
                    ddf = df.reset_index().assign(Dose=f"{dose} nM", Baseline=f"{base:.2f}")
                    tbl_rows.append(ddf)
            if tbl_rows:
                long_df = pd.concat(tbl_rows, ignore_index=True)
                st.dataframe(
                    long_df[["Dose","Hours","Raw Open %","Relative Open %","Closure %","Baseline"]]
                        .style.format({"Raw Open %":"{:.2f}", "Relative Open %":"{:.2f}", "Closure %":"{:.2f}"}),
                    use_container_width=True, height=320
                )

                # Line plot with the two doses
                fig, ax = plt.subplots(figsize=(5.8, 3.8))
                for dose in doses:
                    if dose in results:
                        df, base, _, col = results[dose]
                        x = df.index.values
                        y = df["Closure %"].values.astype(float)
                        ax.plot(x, y, marker="o", linewidth=2.6, color=col,
                                label=f"{dose} nM (baseline {base:.2f}%)")
                ax.set_xlabel("Hours")
                ax.set_ylabel("Closure % (relative to 0h)")
                ax.set_title(f"{cell_line} — {compound} (two-dose run)")
                ax.grid(True, linestyle="--", alpha=0.5)
                # keep legends outside bottom
                fig.subplots_adjust(bottom=0.28)
                fig.legend(loc="lower center", ncols=2, frameon=True, facecolor="white", framealpha=0.9)
                st.pyplot(fig, use_container_width=True)

                # CSV download
                long_df_inspect = long_df.copy()
                long_df_inspect.insert(0, "Cell line", cell_line)
                long_df_inspect.insert(1, "Compound", compound)
                csv = long_df_inspect.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"migration_{cell_line}_{compound}_{doses[0]}nM_{doses[1]}nM.csv",
                    use_container_width=True
                )
            else:
                st.info("No analyzed images yet. Upload and click **Analyze**.")

else:
    st.info("Select two doses, upload images for **0/24/48/72 h** (for each dose), then click **Analyze**.")
