# uvvis_spc_dashboard.py
# Streamlit dashboard for plotting UV‑Vis spectra from .spc files (and CSV/TXT fallback)
# Author: ChatGPT (GPT-5 Thinking)
# Requirements: streamlit, matplotlib, numpy, pandas, (optional) spc-spectra
# If your files are .spc (Galactic/GRAMS), please:  pip install spc-spectra

import io
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Try to import the 'spc' package for Galactic .spc files.
# If unavailable, we'll show a help box.
try:
    # Try modern spc-spectra first, then fall back to old spc
    try:
        import spc_spectra as spc
    except ImportError:
        import spc
    SPC_AVAILABLE = True
except Exception:
    SPC_AVAILABLE = False

st.set_page_config(
    page_title="UV‑Vis .SPC Plotter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Sidebar: File upload & basic options ----------
st.sidebar.title("📁 Data & Options")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more spectra (.spc preferred; CSV/TXT also supported)",
    type=["spc", "csv", "txt", "tsv"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Axes & Scales")

x_label = st.sidebar.text_input("X‑axis label", "Time (min)")
y_label = st.sidebar.text_input("Y‑axis label", "Absorbance (A.U.)")

x_scale = st.sidebar.selectbox("X scale", ["linear", "log"], index=0)
y_scale = st.sidebar.selectbox("Y scale", ["linear", "log"], index=0)

# Limits
col_lims1, col_lims2 = st.sidebar.columns(2)
with col_lims1:
    x_min = st.number_input("x min", value=200.0, format="%.6g")
    y_min = st.number_input("y min", value=0.0, format="%.6g")
with col_lims2:
    x_max = st.number_input("x max", value=600.0, format="%.6g")
    y_max = st.number_input("y max", value=0.6, format="%.6g")

# Ticks
st.sidebar.markdown("### 📏 Tick controls")
col_tick_major, col_tick_minor = st.sidebar.columns(2)
with col_tick_major:
    x_major = st.number_input("x major step", value=50.0, format="%.6g")
    y_major = st.number_input("y major step", value=0.1, format="%.6g")
with col_tick_minor:
    x_minor = st.number_input("x minor step", value=10.0, format="%.6g")
    y_minor = st.number_input("y minor step", value=0.02, format="%.6g")

tick_in = st.sidebar.checkbox("Ticks inward", value=True)
show_top = st.sidebar.checkbox("Show top ticks", value=False)
show_right = st.sidebar.checkbox("Show right ticks", value=False)

# Styling
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Style")
font_size = st.sidebar.slider("Base font size", 10, 36, 22, 1)
line_width = st.sidebar.slider("Line width", 1.0, 5.0, 3.0, 0.5)
use_markers = st.sidebar.checkbox("Use markers", value=False)
marker_size = st.sidebar.slider("Marker size", 3, 12, 6)
legend_loc = st.sidebar.selectbox("Legend location", [
    "best","upper right","upper left","lower left","lower right",
    "right","center left","center right","lower center","upper center","center"
], index=0)
show_legend = st.sidebar.checkbox("Show legend", value=True)

# ---------- Helper functions ----------
def read_spc(file_obj, name_hint=""):
    """Read a .spc file using the 'spc-spectra' package.
    Returns a list of (x, y, label) tuples.
    Handles multi-subfile SPCs.
    """
    import tempfile
    
    if not SPC_AVAILABLE:
        raise RuntimeError("The 'spc-spectra' package is not installed. Run: pip install spc-spectra")
    
    # Read the file data
    data = file_obj.read()
    
    # Try different methods to read the SPC file
    f = None
    
    # Method 1: Try direct bytes (some versions might support this)
    try:
        f = spc.File(data)
    except:
        pass
    
    # Method 2: Try BytesIO
    if f is None:
        try:
            f = spc.File(io.BytesIO(data))
        except:
            pass
    
    # Method 3: Use temporary file (most reliable)
    if f is None:
        with tempfile.NamedTemporaryFile(suffix='.spc', delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_file_path = tmp_file.name
        
        try:
            f = spc.File(tmp_file_path)
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    if f is None:
        raise ValueError("Could not read SPC file with any method")
    
    series = []
    # 'sub' contains individual spectra (subfiles). If absent, data is in f.x/f.y
    if hasattr(f, "sub") and f.sub:
        for i, s in enumerate(f.sub):
            try:
                x = np.array(s.x)
                y = np.array(s.y)
                label = f"{os.path.basename(name_hint)} • sub{i+1}"
                series.append((x, y, label))
            except Exception as e:
                st.warning(f"Could not read subfile {i+1} from {name_hint}: {e}")
    else:
        try:
            x = np.array(f.x)
            y = np.array(f.y)
            label = os.path.basename(name_hint)
            series.append((x, y, label))
        except Exception as e:
            # Some SPC files might have data in different attributes
            if hasattr(f, 'data'):
                # Try to extract x and y from data attribute
                if isinstance(f.data, np.ndarray):
                    # Assume first column is x, second is y
                    if f.data.ndim == 2 and f.data.shape[1] >= 2:
                        x = f.data[:, 0]
                        y = f.data[:, 1]
                    else:
                        # Single column data, create x as indices
                        y = f.data.flatten()
                        x = np.arange(len(y))
                    label = os.path.basename(name_hint)
                    series.append((x, y, label))
                else:
                    raise e
            else:
                raise e
    
    if not series:
        raise ValueError("No data could be extracted from the SPC file")
    
    return series

def read_table(file_obj, name_hint=""):
    """Read CSV/TXT/TSV into x,y. Tries to infer columns:
    - If two columns: x,y
    - If more, tries common headers like 'wavelength','x','time' and 'absorbance','y'
    """
    # Read content once into memory
    content = file_obj.read()
    
    # Try different delimiters
    df = None
    for sep in [",", "\t", ";", r"\s+"]:
        try:
            # Use StringIO to create a file-like object from the content
            df = pd.read_csv(io.StringIO(content.decode('utf-8') if isinstance(content, bytes) else content), 
                           sep=sep, engine="python")
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    
    # If no valid dataframe was created, raise an error
    if df is None or df.shape[1] < 2:
        raise ValueError(f"Could not parse {name_hint} as a valid data file")

    # Try to identify x/y columns
    cols = [c.lower() for c in df.columns]
    x_idx = None
    y_idx = None
    
    # Heuristics for x column
    for k in ["x", "wavelength", "time", "lambda", "nm", "minute", "min"]:
        if k in cols and x_idx is None:
            x_idx = cols.index(k)
    
    # Heuristics for y column
    for k in ["y", "absorbance", "intensity", "a.u.", "au", "signal"]:
        if k in cols and y_idx is None:
            y_idx = cols.index(k)

    # Fallback to first two columns
    if x_idx is None: 
        x_idx = 0
    if y_idx is None: 
        y_idx = 1

    x = pd.to_numeric(df.iloc[:, x_idx], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, y_idx], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    return [(x[mask], y[mask], os.path.basename(name_hint))]

def load_series(uploaded_file):
    """Return list of (x,y,label) for a given uploaded_file, based on extension."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    file_like = uploaded_file
    if suffix == ".spc":
        return read_spc(file_like, uploaded_file.name)
    else:
        return read_table(file_like, uploaded_file.name)

# ---------- Main plotting area ----------
st.title("UV‑Vis Spectrum Plotter (.SPC)")
st.caption("Upload multiple .spc files (Galactic/GRAMS) or CSV/TXT files and customize the plot.")

if not uploaded_files:
    if not SPC_AVAILABLE:
        st.info("Tip: To read **.spc** files, install the optional `spc-spectra` package:\n\n`pip install spc-spectra`")
    st.stop()

# Collect all series
all_series = []
for uf in uploaded_files:
    try:
        series = load_series(uf)
        all_series.extend(series)
    except Exception as e:
        st.warning(f"Could not read {uf.name}: {e}")

if not all_series:
    st.error("No readable data found in the uploaded files.")
    st.stop()

# ---------- Matplotlib style to match provided code ----------
plt.rcParams.clear()

# Use a fallback font list in case Times New Roman is not available
font_list = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams.update({
    "font.family": font_list,
    "font.size": font_size,
    "font.weight": "bold",
    "axes.labelweight": "bold",
})

fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111)

# Tick direction & lengths (match the example)
which_major = dict(which="major", width=2.5, length=9, direction="in" if tick_in else "out",
                   bottom=True, top=show_top, left=True, right=show_right)
which_minor = dict(which="minor", width=1.5, length=6, direction="in" if tick_in else "out",
                   bottom=True, top=show_top, left=True, right=show_right)
plt.tick_params(**which_major)
plt.tick_params(**which_minor)

# Apply scales & limits
ax.set_xscale(x_scale)
ax.set_yscale(y_scale)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Major/minor locators (only apply for linear scale)
if x_scale == "linear":
    try:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_minor))
    except Exception:
        pass

if y_scale == "linear":
    try:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor))
    except Exception:
        pass

# Labels
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

# Plot all series
for (x, y, label) in all_series:
    if use_markers:
        ax.plot(x, y, linewidth=line_width, marker="o", markersize=marker_size, label=label)
    else:
        ax.plot(x, y, linewidth=line_width, label=label)

# Legend
if show_legend:
    ax.legend(loc=legend_loc, frameon=False)

st.pyplot(fig, clear_figure=True)

# ---------- Download consolidated data (optional) ----------
st.markdown("---")
st.subheader("⬇️ Download combined CSV (optional)")
if st.button("Build combined CSV from all visible series"):
    # Build a long-form dataframe: file, x, y
    rows = []
    for (x, y, label) in all_series:
        rows.append(pd.DataFrame({"file": label, "x": x, "y": y}))
    combined = pd.concat(rows, ignore_index=True)
    csv_bytes = combined.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="combined_uvvis.csv", mime="text/csv")

# ---------- Footer help ----------
with st.expander("ℹ️ Notes & Troubleshooting"):
    st.markdown(
        """
        **.SPC support:** This app reads Galactic/GRAMS `.spc` files via the optional `spc-spectra` package.
        If you see an error like *'No module named spc_spectra'*, install it with:

        ```bash
        pip install spc-spectra
        ```

        **CSV/TXT:** If your spectrometer exports text files instead, the app will infer the first two numeric columns as X and Y.
        You can rename axes and adjust limits, scales, and tick spacing from the sidebar.

        **Style:** The plot uses **Times New Roman**, bold labels, and inward ticks with separate major/minor lengths to match the look you shared.
        """
    )
