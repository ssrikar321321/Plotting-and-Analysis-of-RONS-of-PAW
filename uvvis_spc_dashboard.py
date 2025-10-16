# uvvis_spc_dashboard.py
# Streamlit dashboard for plotting UV‑Vis spectra from .spc files (and CSV/TXT fallback)
# Requirements: streamlit, matplotlib, numpy, pandas
# Optional for better SPC support: pip install specio OR pip install spc-spectra
# The app includes a basic SPC reader that works without additional packages!

import io
import os
import sys
import tempfile
import struct
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Try to import SPC reading capability from various sources
# We'll try multiple libraries in order of preference
SPC_AVAILABLE = False
SPC_METHOD = None

# Method 1: Try specio (most modern and maintained)
try:
    import specio
    SPC_AVAILABLE = True
    SPC_METHOD = "specio"
except ImportError:
    pass

# Method 2: Try spc-spectra
if not SPC_AVAILABLE:
    try:
        try:
            import spc_spectra as spc
        except ImportError:
            import spc
        SPC_AVAILABLE = True
        SPC_METHOD = "spc"
    except ImportError:
        pass

# Method 3: We'll implement a basic SPC reader if nothing else works
if not SPC_AVAILABLE:
    SPC_METHOD = "custom"
    SPC_AVAILABLE = True  # We'll always have our custom reader

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
def read_spc_custom(data):
    """Basic SPC file reader for Galactic/GRAMS format.
    This is a minimal implementation that handles common UV-Vis SPC files.
    """
    import struct
    
    # SPC file format has specific byte signatures
    # Check if this looks like an SPC file
    if len(data) < 512:
        raise ValueError("File too small to be a valid SPC file")
    
    # Read header information
    # Bytes 0-3: SPC file signature (should be specific values)
    # Byte 4: File version
    version = data[4]
    
    # Bytes 8-11: Number of points (little-endian)
    num_points = struct.unpack('<I', data[8:12])[0]
    
    # Bytes 16-19: First X value (little-endian float)
    # Bytes 20-23: Last X value (little-endian float)
    first_x = struct.unpack('<f', data[16:20])[0]
    last_x = struct.unpack('<f', data[20:24])[0]
    
    # Check data type flag at byte 5
    data_type = data[5]
    
    # For single file SPCs, Y data typically starts at byte 512
    y_start = 512
    
    try:
        # Generate X values (evenly spaced)
        if num_points > 0 and num_points < 100000:  # Sanity check
            x = np.linspace(first_x, last_x, num_points)
            
            # Read Y values (4-byte floats)
            y = []
            for i in range(num_points):
                offset = y_start + (i * 4)
                if offset + 4 <= len(data):
                    y_val = struct.unpack('<f', data[offset:offset+4])[0]
                    y.append(y_val)
            
            y = np.array(y[:len(x)])  # Ensure same length
            
            if len(y) > 0:
                return x, y
    except:
        pass
    
    # Fallback: Try to parse as different format
    # Some SPC files might have different structures
    raise ValueError("Could not parse SPC file structure")

def read_spc(file_obj, name_hint=""):
    """Read a .spc file using available methods.
    Returns a list of (x, y, label) tuples.
    """
    import tempfile
    
    # Read the file data
    file_obj.seek(0)  # Ensure we're at the beginning
    data = file_obj.read()
    
    series = []
    
    # Method 1: Try specio (if available)
    if SPC_METHOD == "specio":
        try:
            import specio
            # specio needs a file path, so use temporary file
            with tempfile.NamedTemporaryFile(suffix='.spc', delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_file_path = tmp_file.name
            
            try:
                # Read using specio
                spec_data = specio.load(tmp_file_path)
                
                # specio returns different formats, try to extract x,y
                if hasattr(spec_data, 'wavelength') and hasattr(spec_data, 'flux'):
                    x = np.array(spec_data.wavelength)
                    y = np.array(spec_data.flux)
                elif hasattr(spec_data, 'x') and hasattr(spec_data, 'y'):
                    x = np.array(spec_data.x)
                    y = np.array(spec_data.y)
                elif isinstance(spec_data, (list, tuple)) and len(spec_data) >= 2:
                    x = np.array(spec_data[0])
                    y = np.array(spec_data[1])
                else:
                    # Try to extract as array
                    arr = np.array(spec_data)
                    if arr.ndim == 2:
                        x = arr[0] if arr.shape[0] == 2 else arr[:, 0]
                        y = arr[1] if arr.shape[0] == 2 else arr[:, 1]
                    else:
                        raise ValueError("Unknown specio data format")
                
                label = os.path.basename(name_hint)
                series.append((x, y, label))
                return series
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        except Exception as e:
            # Fall back to next method
            pass
    
    # Method 2: Try spc/spc-spectra (if available)
    if SPC_METHOD == "spc":
        try:
            with tempfile.NamedTemporaryFile(suffix='.spc', delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_file_path = tmp_file.name
            
            try:
                f = spc.File(tmp_file_path)
                
                if hasattr(f, "sub") and f.sub:
                    for i, s in enumerate(f.sub):
                        x = np.array(s.x)
                        y = np.array(s.y)
                        label = f"{os.path.basename(name_hint)} • sub{i+1}"
                        series.append((x, y, label))
                else:
                    x = np.array(f.x)
                    y = np.array(f.y)
                    label = os.path.basename(name_hint)
                    series.append((x, y, label))
                
                return series
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        except Exception as e:
            # Fall back to custom reader
            pass
    
    # Method 3: Use custom SPC reader as fallback
    try:
        x, y = read_spc_custom(data)
        label = os.path.basename(name_hint)
        series.append((x, y, label))
        return series
    except Exception as e:
        # If custom reader fails, try one more approach
        # Some SPC files are actually text files with SPC extension
        try:
            # Try to decode as text
            text = data.decode('utf-8', errors='ignore')
            if '\n' in text and any(c in text for c in [',', '\t', ' ']):
                # Looks like text data, parse as CSV/TSV
                import io
                return read_table(io.StringIO(text), name_hint)
        except:
            pass
        
        raise ValueError(f"Could not read SPC file with any available method. Last error: {e}")

def read_table(file_obj, name_hint=""):
    """Read CSV/TXT/TSV into x,y. Tries to infer columns:
    - If two columns: x,y
    - If more, tries common headers like 'wavelength','x','time' and 'absorbance','y'
    """
    # Read content once into memory
    try:
        file_obj.seek(0)
    except:
        pass
    
    content = file_obj.read()
    
    # Handle both bytes and string content
    if isinstance(content, bytes):
        text = content.decode('utf-8', errors='ignore')
    else:
        text = content
    
    # Try different delimiters
    df = None
    for sep in [",", "\t", ";", r"\s+"]:
        try:
            # Use StringIO to create a file-like object from the content
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
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
    
    # Reset file position to beginning
    try:
        uploaded_file.seek(0)
    except:
        pass
    
    if suffix == ".spc":
        return read_spc(uploaded_file, uploaded_file.name)
    else:
        return read_table(uploaded_file, uploaded_file.name)

# ---------- Main plotting area ----------
st.title("UV‑Vis Spectrum Plotter (.SPC)")
st.caption("Upload multiple .spc files (Galactic/GRAMS) or CSV/TXT files and customize the plot.")

if not uploaded_files:
    st.info("""
    📊 **Welcome to the UV-Vis Spectrum Plotter!**
    
    Upload your spectroscopy files using the sidebar:
    - **SPC files** (Galactic/GRAMS format)
    - **CSV/TXT files** (with wavelength and absorbance columns)
    
    For better SPC support, you can optionally install:
    - `pip install specio` (recommended)
    - `pip install spc-spectra`
    
    The app includes a basic SPC reader that works without additional packages!
    """)
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
        **.SPC support:** This app can read Galactic/GRAMS `.spc` files using:
        - Built-in basic SPC reader (always available)
        - `specio` package for enhanced support: `pip install specio`
        - `spc-spectra` package as alternative: `pip install spc-spectra`

        **CSV/TXT:** The app automatically detects wavelength/absorbance columns from text files.
        Common column names like 'wavelength', 'nm', 'absorbance', 'intensity' are recognized.

        **Style:** The plot uses Times New Roman (or serif fallback), bold labels, and customizable ticks.
        
        **Troubleshooting SPC files:**
        - If an SPC file won't load, try exporting it as CSV/TXT from your instrument software
        - Some older SPC formats may not be fully supported
        - The app will try multiple methods to read your file automatically
        """
    )
