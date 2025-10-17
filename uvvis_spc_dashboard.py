#!/usr/bin/env python3
"""
Interactive Shimadzu SPC to CSV Converter
Allows you to verify and correct wavelength ranges
"""

import streamlit as st
import numpy as np
import pandas as pd
import olefile
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Shimadzu SPC Converter", layout="wide")
st.title("🔬 Shimadzu SPC to CSV Converter")
st.caption("Extract correct wavelength and absorbance data from OLE-format SPC files")

# File upload
uploaded_file = st.file_uploader("Upload Shimadzu .spc file", type=['spc'])

if uploaded_file:
    # Read file
    data = uploaded_file.read()
    
    # Save to temp file (olefile needs path)
    with tempfile.NamedTemporaryFile(suffix='.spc', delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    try:
        ole = olefile.OleFileIO(tmp_path)
        
        st.success(f"✅ Opened OLE file: {uploaded_file.name}")
        
        # Find all float arrays
        all_arrays = []
        
        with st.expander("📁 OLE Structure Analysis"):
            for stream_path in ole.listdir():
                stream_name = '/'.join(stream_path)
                stream_data = ole.openstream(stream_path).read()
                
                st.write(f"**Stream:** {stream_name} ({len(stream_data)} bytes)")
                
                # Try to extract float arrays
                for offset in [0, 32, 64, 128, 256, 512]:
                    if len(stream_data) > offset:
                        remaining = stream_data[offset:]
                        
                        if len(remaining) % 4 == 0:
                            try:
                                arr = np.frombuffer(remaining, dtype='<f4')
                                if arr.size > 10 and np.all(np.isfinite(arr[:10])):
                                    all_arrays.append({
                                        'name': f"{stream_name}_offset{offset}",
                                        'data': arr,
                                        'min': arr.min(),
                                        'max': arr.max(),
                                        'size': arr.size,
                                        'is_monotonic': np.all(np.diff(arr) >= 0)
                                    })
                                    st.write(f"  - Found array at offset {offset}: "
                                           f"{arr.size} points, range [{arr.min():.2f}, {arr.max():.2f}]")
                            except:
                                pass
        
        ole.close()
        os.unlink(tmp_path)
        
        if all_arrays:
            st.header("🔍 Data Selection")
            
            # Identify likely candidates
            wavelength_candidates = [a for a in all_arrays 
                                    if a['is_monotonic'] and 100 < a['min'] < 400 and a['max'] > 400]
            absorbance_candidates = [a for a in all_arrays 
                                    if -1 < a['min'] < 2 and a['max'] < 10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Wavelength Data")
                
                if wavelength_candidates:
                    st.info(f"Found {len(wavelength_candidates)} potential wavelength array(s)")
                    wl_options = {f"Array {i+1}: {c['size']} points, [{c['min']:.1f}-{c['max']:.1f}]": i 
                                for i, c in enumerate(wavelength_candidates)}
                    wl_choice = st.selectbox("Select wavelength array:", 
                                            options=list(wl_options.keys()))
                    wavelength_array = wavelength_candidates[wl_options[wl_choice]]['data']
                else:
                    st.warning("No wavelength array found. Will generate based on your input.")
                    wavelength_array = None
                
                # Manual wavelength input
                st.markdown("**Or specify wavelength range manually:**")
                wcol1, wcol2, wcol3 = st.columns(3)
                with wcol1:
                    start_wl = st.number_input("Start (nm)", value=200.0, step=10.0)
                with wcol2:
                    end_wl = st.number_input("End (nm)", value=800.0, step=10.0)
                with wcol3:
                    use_manual = st.checkbox("Use manual range", value=(wavelength_array is None))
            
            with col2:
                st.subheader("Absorbance Data")
                
                if absorbance_candidates:
                    st.info(f"Found {len(absorbance_candidates)} potential absorbance array(s)")
                    abs_options = {f"Array {i+1}: {c['size']} points, [{c['min']:.4f}-{c['max']:.4f}]": i 
                                 for i, c in enumerate(absorbance_candidates)}
                    abs_choice = st.selectbox("Select absorbance array:", 
                                             options=list(abs_options.keys()))
                    absorbance_array = absorbance_candidates[abs_options[abs_choice]]['data']
                else:
                    # Try to find any array that's not wavelength
                    other_arrays = [a for a in all_arrays if not a['is_monotonic']]
                    if other_arrays:
                        abs_options = {f"{a['name']}: {a['size']} points": i 
                                     for i, a in enumerate(other_arrays)}
                        abs_choice = st.selectbox("Select data array:", 
                                                 options=list(abs_options.keys()))
                        absorbance_array = other_arrays[abs_options[abs_choice]]['data']
                    else:
                        st.error("No absorbance data found!")
                        absorbance_array = None
            
            # Process and preview
            if absorbance_array is not None:
                # Determine wavelengths
                n_points = len(absorbance_array)
                
                if use_manual or wavelength_array is None:
                    wavelengths = np.linspace(start_wl, end_wl, n_points)
                    st.info(f"📏 Using manual wavelength range: {start_wl}-{end_wl} nm for {n_points} points")
                else:
                    wavelengths = wavelength_array
                    
                    # Check if sizes match
                    if len(wavelengths) != len(absorbance_array):
                        st.warning(f"⚠️ Size mismatch! Wavelength: {len(wavelengths)}, Absorbance: {len(absorbance_array)}")
                        st.info("Generating wavelength array to match absorbance data...")
                        wavelengths = np.linspace(start_wl, end_wl, n_points)
                
                # Create dataframe
                df = pd.DataFrame({
                    'wavelength_nm': wavelengths,
                    'absorbance': absorbance_array
                })
                
                # Preview plot
                st.header("📊 Data Preview")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Full spectrum
                ax1.plot(df['wavelength_nm'], df['absorbance'], 'b-', linewidth=1)
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Absorbance')
                ax1.set_title('Full Spectrum')
                ax1.grid(True, alpha=0.3)
                
                # Zoomed view (first 100 points)
                n_zoom = min(100, len(df))
                ax2.plot(df['wavelength_nm'][:n_zoom], df['absorbance'][:n_zoom], 'r.-', linewidth=1, markersize=2)
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Absorbance')
                ax2.set_title(f'Zoom (first {n_zoom} points)')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data points", len(df))
                with col2:
                    st.metric("λ range (nm)", f"{df['wavelength_nm'].min():.1f} - {df['wavelength_nm'].max():.1f}")
                with col3:
                    st.metric("Abs range", f"{df['absorbance'].min():.4f} - {df['absorbance'].max():.4f}")
                with col4:
                    interval = np.diff(df['wavelength_nm']).mean()
                    st.metric("λ interval", f"{interval:.2f} nm")
                
                # Export
                st.header("💾 Export")
                
                # Show first few rows
                st.subheader("Data preview (first 10 rows):")
                st.dataframe(df.head(10))
                
                # Download button
                csv = df.to_csv(index=False)
                filename = uploaded_file.name.replace('.spc', '_converted.csv')
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
                
                # Show instructions for verification
                with st.expander("✅ How to verify the wavelength is correct"):
                    st.markdown("""
                    1. **Check the wavelength range**: Does it match your instrument settings?
                    2. **Look for characteristic peaks**: 
                       - UV absorbers typically peak around 200-400 nm
                       - Visible dyes peak at their characteristic colors
                    3. **Compare with known samples**: If you have a standard, check if peaks align
                    4. **Check the interval**: Most UV-Vis uses 0.5, 1, or 2 nm intervals
                    
                    If the wavelength seems wrong, try:
                    - Using the manual wavelength range input
                    - Selecting a different array from the dropdown
                    - Checking your instrument's method settings for the actual scan range
                    """)
        else:
            st.error("Could not find any data arrays in this OLE file.")
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        os.unlink(tmp_path)
else:
    st.info("""
    ### 📝 Instructions
    
    1. **Upload your Shimadzu .spc file** using the button above
    2. **Review the detected arrays** - the tool will find potential wavelength and absorbance data
    3. **Verify or correct the wavelength range** if needed
    4. **Preview the spectrum** to ensure it looks correct
    5. **Download the CSV file** with corrected wavelengths
    
    ### ⚠️ Common Issues & Solutions
    
    **Wrong wavelength range?**
    - Check your instrument method settings for the actual scan range
    - Use the manual wavelength input to override detected values
    - Common ranges: 200-800 nm, 190-1100 nm, 200-600 nm
    
    **No wavelength array found?**
    - Some Shimadzu files only store absorbance data
    - Enter your scan range manually (check your method settings)
    
    **Multiple arrays found?**
    - Try different combinations until the spectrum looks correct
    - The largest array is usually the main data
    """)
