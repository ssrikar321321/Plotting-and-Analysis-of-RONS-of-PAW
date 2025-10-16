# UV‑Vis `.SPC` Plotter – Streamlit App

A lightweight Streamlit dashboard to **read and plot multiple UV‑Vis spectra** from Galactic/GRAMS **`.spc`** files.  
CSV/TXT files are also supported as a fallback. The app reproduces a **publication‑style** look (Times New Roman, bold labels, inward ticks, major/minor tick spacing) and gives full control over **axis scales, limits, and ticks**.

![example](UV_vis.png)

---

## 🚀 Quick Start

```bash
# 1) Create a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run the app
streamlit run uvvis_spc_dashboard.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

> **Note:** The `spc` package is used for reading `.spc` files. If your spectrometer exports CSV/TXT instead, the app will still work without `spc`.

---

## 📂 Files

- `uvvis_spc_dashboard.py` – the Streamlit app
- `requirements.txt` – Python dependencies
- `UV_vis.png` – example style reference (used in this README)

---

## ✨ Features

- Upload **multiple** `.spc` files (handles multi‑subfile SPCs) or CSV/TXT files
- Customize:
  - **X/Y scales**: linear or log
  - **X/Y limits**
  - **Major/minor tick spacing**
  - **Tick direction** (inward/outward), show/hide top/right ticks
  - **Font size**, **line width**, **markers**, legend position
- **Times New Roman** + **bold** labels, inward tick styling to match journal plots
- One‑click **download** of combined CSV for all displayed traces

---

## 🧭 Usage Notes

1. **Uploading files**
   - `.spc` (Galactic/GRAMS): requires the `spc` Python package.
   - `.csv/.txt/.tsv`: the first two numeric columns are treated as **X** and **Y** (or the app will try to detect headers like `time`, `wavelength`, `absorbance`, etc.).

2. **Axes & scales**
   - Choose linear/log for both axes.
   - Set exact limits for X/Y.
   - Control major/minor tick spacing independently.

3. **Styling**
   - The app uses **Times New Roman** and bold axis labels.
   - Tick marks are **inward** by default with separate lengths for major/minor ticks.
   - Toggle legend, markers, and line width to match your lab’s style guide.

4. **Export**
   - Click **“Build combined CSV from all visible series”** to export long‑format data (`file`, `x`, `y`).

---

## 🧩 Troubleshooting

- **`ModuleNotFoundError: No module named 'spc'`**
  - Run: `pip install spc`  
  - If you only use CSV/TXT data, you can ignore this by not uploading `.spc` files.
- **Fonts not matching Times New Roman**
  - Ensure the font is installed on your OS. Matplotlib will fall back if it’s missing.
- **CSV/TXT not parsing**
  - Check delimiters (comma, tab, semicolon). The app tries common separators automatically.
  - Make sure the numeric columns are clean (no units strings in the same cell).

---

## 🛠️ Dev Notes

- Tested with Python 3.9+ on macOS and Windows.
- If you want additional features (baseline correction, smoothing, normalization, color palettes, saving high‑DPI PNG/SVG), open an issue or extend the app—code is a single file.

---

## 📜 License

MIT
