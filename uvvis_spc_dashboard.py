# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------- Page config -----------------------
st.set_page_config(page_title="In-situ UV/Vis – NO₂⁻ / NO₃⁻ Analyzer",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------- Utils -----------------------
def _split_excel_spectra(df_raw: pd.DataFrame):
    """
    Parse an Excel sheet that has some metadata rows followed by a header row
    whose first cell starts with 'Wavelength'. Returns dict:
      {sample_name: DataFrame(columns=['wavelength','absorbance'])}
    """
    header_row_idx = None
    for i in range(min(80, len(df_raw))):
        cell = str(df_raw.iloc[i, 0]).strip().lower()
        if cell.startswith("wavelength"):
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("Couldn't find a 'Wavelength [nm]' header row in this Excel file.")

    table = df_raw.iloc[header_row_idx:].copy()
    table.columns = table.iloc[0]
    table = table.iloc[1:]
    wl_col = [c for c in table.columns if str(c).lower().startswith("wavelength")][0]
    table = table.rename(columns={wl_col: "wavelength"})

    # numeric coercion
    table["wavelength"] = pd.to_numeric(table["wavelength"], errors="coerce")
    for c in table.columns:
        if c == "wavelength":
            continue
        table[c] = pd.to_numeric(table[c], errors="coerce")
    table = table.dropna(subset=["wavelength"])

    spectra = {}
    for c in table.columns:
        if c == "wavelength":
            continue
        df = table[["wavelength", c]].dropna().rename(columns={c: "absorbance"})
        if df["absorbance"].notna().sum() == 0:
            continue
        spectra[str(c)] = df.sort_values("wavelength")
    if not spectra:
        raise ValueError("No sample columns with numeric absorbance were found.")
    return spectra


def read_spectrum_file(file):
    """
    Wrapper that supports:
      - Excel (.xlsx/.xls) with metadata block + 'Wavelength [nm]' header row
      - CSV/TSV with (wavelength, absorbance) columns
    Returns dict {sample_name: df(wavelength, absorbance)}.
    """
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(file, header=None)
        return _split_excel_spectra(df_raw)
    else:
        # CSV/TSV
        content = file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=r"\s+|;|\t|,", engine="python")
        cols = [c.lower() for c in df.columns]
        w_candidates = [i for i, c in enumerate(cols) if "wav" in c or "nm" in c]
        a_candidates = [i for i, c in enumerate(cols) if "abs" in c or "od" in c or "a.u" in c]
        if not w_candidates or not a_candidates:
            w_idx, a_idx = 0, 1
        else:
            w_idx, a_idx = w_candidates[0], a_candidates[0]
        out = df.iloc[:, [w_idx, a_idx]].copy()
        out.columns = ["wavelength", "absorbance"]
        out = out.dropna().sort_values("wavelength")
        return {file.name: out}


def window_slice(df, wl_min, wl_max):
    return df[(df["wavelength"] >= wl_min) & (df["wavelength"] <= wl_max)].copy()


def apply_baseline(df):
    """Linear baseline across the current window."""
    if len(df) < 2:
        return df
    x = df["wavelength"].values
    y = df["absorbance"].values
    p = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
    df = df.copy()
    df["absorbance"] = y - (p[0] * x + p[1])
    return df


def resample_common_grid(dfs, wl_min, wl_max, step):
    """Interpolate all spectra to a common wavelength grid; returns (wl_grid, A)."""
    wl_grid = np.arange(wl_min, wl_max + 1e-12, step)
    mats = []
    for df in dfs:
        d = df.set_index("wavelength")["absorbance"].sort_index()
        d = d.reindex(wl_grid, method=None)
        d = d.interpolate(limit_direction="both")
        mats.append(d.values)
    A = np.vstack(mats).T  # shape: i x m
    return wl_grid, A


def derivative_spectrum(y, x):
    """First derivative dA/dλ using central gradient."""
    return np.gradient(y, x)


# ----------------------- Sidebar: acquisition & preprocessing -----------------------
st.sidebar.header("Acquisition / Preprocessing")

# Informational (path length is constant for standards & unknowns)
st.sidebar.number_input("Cuvette path length L (cm) – informational", value=1.0, step=0.1)

# Recommended window from the method
wl_min = st.sidebar.number_input("Wavelength min (nm)", value=227.1, step=0.1, format="%.1f")
wl_max = st.sidebar.number_input("Wavelength max (nm)", value=233.9, step=0.1, format="%.1f")
step = st.sidebar.selectbox("Interpolation step (nm)", [0.1, 0.2, 0.5, 1.0], index=2)

baseline_on = st.sidebar.checkbox("Linear baseline correction (per spectrum)", value=True)

use_savgol = st.sidebar.checkbox("Savgitzky–Golay smoothing", value=False)
if use_savgol:
    from scipy.signal import savgol_filter
    sg_win = st.sidebar.slider("Savgol window (odd)", 5, 51, 11, step=2)
    sg_poly = st.sidebar.slider("Savgol polyorder", 2, 5, 3)

use_derivative = st.sidebar.checkbox("Use first derivative (dA/dλ) for calibration & prediction", value=False,
                                     help="Enable only if overlapping bands or sloping baselines degrade the fit.")

# ----------------------- Demo calibration option -----------------------
st.sidebar.markdown("### Demo calibration (no standards)")
use_demo_K = st.sidebar.toggle("Use demo K (semi-quantitative)", value=False,
                               help="Synthesizes a K(λ) with tail-like shapes so you can run without standards.")
demo_params = {}
if use_demo_K:
    st.sidebar.caption("Adjust shapes to roughly mimic your NO₂⁻/NO₃⁻ tails:")
    demo_params["alpha_NO2"] = st.sidebar.slider("NO₂⁻ tail steepness α (1/nm)", 0.05, 0.60, 0.22, 0.01)
    demo_params["alpha_NO3"] = st.sidebar.slider("NO₃⁻ tail steepness β (1/nm)", 0.05, 0.60, 0.35, 0.01)
    demo_params["ratio_NO2_to_NO3"] = st.sidebar.slider("Relative strength NO₂⁻ / NO₃⁻", 0.10, 3.0, 0.8, 0.05)
    demo_params["scale_K"] = st.sidebar.number_input("Overall K scale", value=1.0, step=0.1)

# Optional absolute scaling (applied after prediction)
scale_abs = st.sidebar.number_input("Output concentration scale factor", value=1.0, step=0.1,
                                    help="Multiply predicted concentrations by this factor (leave 1.0 if unknown).")

# ----------------------- Section 1: Build K -----------------------
st.header("1) Calibration: Build **K(λ)**")

K = None
wl_grid = None

if use_demo_K:
    wl_grid = np.arange(wl_min, wl_max + 1e-12, step)
    lam0 = wl_min
    k_no2 = np.exp(-demo_params["alpha_NO2"] * (wl_grid - lam0))
    k_no3 = np.exp(-demo_params["alpha_NO3"] * (wl_grid - lam0))
    # normalize columns, then set NO2/NO3 relative strength and overall scale
    k_no2 = k_no2 / np.linalg.norm(k_no2)
    k_no3 = k_no3 / np.linalg.norm(k_no3)
    k_no2 = demo_params["ratio_NO2_to_NO3"] * k_no2
    K = demo_params["scale_K"] * np.column_stack([k_no2, k_no3])

    if use_derivative:
        # if derivative mode, K must also be derivative of k(λ)
        dk_no2 = derivative_spectrum(k_no2, wl_grid)
        dk_no3 = derivative_spectrum(k_no3, wl_grid)
        K = np.column_stack([dk_no2, dk_no3])

    st.info("Using **Demo K** (no standards). Results are semi-quantitative.")
    fig = plt.figure()
    plt.plot(wl_grid, K[:, 0], label="k(NO₂⁻) – demo" + (" (deriv)" if use_derivative else ""))
    plt.plot(wl_grid, K[:, 1], label="k(NO₃⁻) – demo" + (" (deriv)" if use_derivative else ""))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("k(λ) (arb.)")
    plt.legend()
    st.pyplot(fig)

else:
    st.caption("Upload **standard** spectra (Excel/CSV). Excel may contain multiple samples per sheet.")
    std_files = st.file_uploader("Standard spectra files", type=["xlsx", "xls", "csv", "tsv", "txt"],
                                 accept_multiple_files=True, key="std_files")
    if std_files:
        std_specs = []
        std_labels = []
        conc_rows = []

        for f in std_files:
            spec_dict = read_spectrum_file(f)
            for sample_name, df in spec_dict.items():
                label = f"{f.name}:{sample_name}"
                # preprocess
                df = window_slice(df, wl_min, wl_max)
                if baseline_on:
                    df = apply_baseline(df)
                if use_savgol and len(df) >= sg_win:
                    df = df.copy()
                    df["absorbance"] = savgol_filter(df["absorbance"].values, sg_win, sg_poly)
                std_specs.append(df)
                std_labels.append(label)

        with st.expander("Enter concentrations for each standard (µM)"):
            for label in std_labels:
                c_no2 = st.number_input(f"[{label}]  NO₂⁻ (µM)", min_value=0.0, value=0.0, step=1.0, key=f"no2_{label}")
                c_no3 = st.number_input(f"[{label}]  NO₃⁻ (µM)", min_value=0.0, value=0.0, step=1.0, key=f"no3_{label}")
                conc_rows.append({"label": label, "NO2_uM": c_no2, "NO3_uM": c_no3})
        C_df = pd.DataFrame(conc_rows).set_index("label")

        # Common grid and A matrix
        wl_grid, A = resample_common_grid(std_specs, wl_min, wl_max, step)   # i x m
        # If derivative selected, convert A -> dA/dλ
        if use_derivative:
            A = np.apply_along_axis(lambda y: derivative_spectrum(y, wl_grid), 0, A)

        # Build concentration matrix C (n x m), n=2 components
        C = C_df[["NO2_uM", "NO3_uM"]].to_numpy().T
        try:
            K = A @ C.T @ np.linalg.inv(C @ C.T)  # i x n
            st.success("Computed K from standards.")
            fig = plt.figure()
            plt.plot(wl_grid, K[:, 0], label="k(NO₂⁻)" + (" (deriv)" if use_derivative else ""))
            plt.plot(wl_grid, K[:, 1], label="k(NO₃⁻)" + (" (deriv)" if use_derivative else ""))
            plt.xlabel("wavelength (nm)")
            plt.ylabel("k(λ)")
            plt.legend()
            st.pyplot(fig)
        except np.linalg.LinAlgError:
            st.error("Matrix C·Cᵀ is singular. Provide more varied standards (non-collinear in composition).")

# ----------------------- Section 2: Analyze unknowns -----------------------
st.header("2) Analyze unknown spectra")

unk_files = st.file_uploader("Unknown spectra files", type=["xlsx", "xls", "csv", "tsv", "txt"],
                             accept_multiple_files=True, key="unk_files")

if unk_files and K is not None:
    unk_specs = []
    labels = []

    for f in unk_files:
        try:
            spec_dict = read_spectrum_file(f)
        except Exception as e:
            st.error(f"{f.name}: {e}")
            continue

        for sample_name, df in spec_dict.items():
            label = f"{f.name}:{sample_name}"
            df = window_slice(df, wl_min, wl_max)
            if baseline_on:
                df = apply_baseline(df)
            if use_savgol and len(df) >= (sg_win if use_savgol else 5):
                df = df.copy()
                df["absorbance"] = savgol_filter(df["absorbance"].values, sg_win, sg_poly)
            unk_specs.append(df)
            labels.append(label)

    wl_grid_unk, Aunk = resample_common_grid(unk_specs, wl_min, wl_max, step)
    if not np.allclose(wl_grid_unk, wl_grid):
        # Re-interpolate to match K grid if needed
        wl_grid = wl_grid_unk

    if use_derivative:
        Aunk = np.apply_along_axis(lambda y: derivative_spectrum(y, wl_grid), 0, Aunk)

    # Pseudoinverse for stability
    Kpinv = np.linalg.pinv(K)   # shape: (n x i)

    concs = []
    for j in range(Aunk.shape[1]):     # for each unknown spectrum
        Acol = Aunk[:, j]
        c = Kpinv @ Acol               # (2,)
        concs.append(c)
    concs = np.array(concs) * scale_abs
    conc_df = pd.DataFrame(concs, columns=["[NO2-]_uM", "[NO3-]_uM"])
    conc_df.insert(0, "sample", labels)

    st.subheader("Predicted concentrations (µM)")
    st.dataframe(conc_df, use_container_width=True)

    # Bar plot
    fig2 = plt.figure()
    x = np.arange(len(labels))
    plt.bar(x - 0.15, conc_df["[NO2-]_uM"], width=0.3, label="NO₂⁻")
    plt.bar(x + 0.15, conc_df["[NO3-]_uM"], width=0.3, label="NO₃⁻")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Concentration (µM)")
    plt.legend()
    st.pyplot(fig2)

    # ---------------- Energy efficiency ----------------
    st.subheader("3) Energy efficiency η")
    col1, col2, col3 = st.columns(3)
    with col1:
        V_mL = st.number_input("Water volume V (mL)", value=3.0, step=0.1)
    with col2:
        Pd_W = st.number_input("Discharge power P_d (W)", value=10.0, step=0.1)
    with col3:
        add_o3 = st.checkbox("Add 8 W for ozone generator", value=False)

    Pd_eff = Pd_W + (8.0 if add_o3 else 0.0)
    default_t = st.number_input("Treatment time Δt per sample (s)", value=60.0, step=1.0)

    eta_rows = []
    for _, row in conc_df.iterrows():
        NOx_uM = row["[NO2-]_uM"] + row["[NO3-]_uM"]
        V_L = V_mL / 1000.0
        eta_umol_per_J = (NOx_uM * V_L) / (Pd_eff * default_t)   # µmol/J
        eta_nmol_per_J = eta_umol_per_J * 1e3                    # nmol/J
        eta_rows.append({"sample": row["sample"], "eta (nmol/J)": eta_nmol_per_J})
    eta_df = pd.DataFrame(eta_rows)

    st.dataframe(eta_df, use_container_width=True)
    fig3 = plt.figure()
    plt.plot(eta_df["sample"], eta_df["eta (nmol/J)"], marker="o")
    plt.ylabel("η (nmol/J)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig3)

    # Download
    out = conc_df.merge(eta_df, on="sample", how="left")
    st.download_button("⬇️ Download results (CSV)",
                       out.to_csv(index=False),
                       file_name="NOx_concentrations_and_eta.csv",
                       mime="text/csv")
elif unk_files and K is None:
    st.warning("Please build K first (use Demo K or upload standards).")

# ----------------------- Method notes -----------------------
st.divider()
st.markdown(
    "**Notes**  \n"
    "- Default window **227.1–233.9 nm** minimizes pH sensitivity and NO₃⁻ saturation below 210 nm.  \n"
    "- Keep **baseline correction** enabled unless your blank subtraction is perfect.  \n"
    "- **First-derivative mode** helps when bands overlap or the baseline drifts; it increases noise, so use with smoothing.  \n"
    "- The **Demo K** is for workflow testing only; switch to real standards for quantitative work."
)
