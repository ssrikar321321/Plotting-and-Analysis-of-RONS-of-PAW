#!/usr/bin/env python3
"""
Shimadzu OLE SPC to CSV Converter
Extracts spectroscopy data from Shimadzu UVProbe/LabSolutions .spc files
"""

import sys
import numpy as np
import pandas as pd
import olefile
import os

def extract_shimadzu_spc(filepath):
    """
    Extract data from Shimadzu OLE-format SPC files.
    Returns list of (wavelength, absorbance) arrays.
    """
    ole = olefile.OleFileIO(filepath)
    results = []
    
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    print(f"OLE streams found: {ole.listdir()}")
    
    # Iterate through all streams in the OLE file
    for stream_path in ole.listdir():
        stream_name = '/'.join(stream_path)
        
        try:
            stream_data = ole.openstream(stream_path).read()
            stream_size = len(stream_data)
            
            # Skip small streams (likely metadata)
            if stream_size < 100:
                continue
            
            print(f"\nTrying stream: {stream_name} ({stream_size} bytes)")
            
            # Try different interpretations
            found_data = False
            
            # Method 1: Try as float32 array with various offsets
            for offset in [0, 4, 8, 16, 32, 64, 128, 256, 512]:
                if stream_size <= offset:
                    continue
                
                trimmed = stream_data[offset:]
                
                # Try float32
                if len(trimmed) % 4 == 0:
                    try:
                        arr = np.frombuffer(trimmed, dtype='<f4')  # little-endian float32
                        
                        # Check if values are reasonable for spectroscopy
                        if arr.size > 10 and np.all(np.isfinite(arr)):
                            max_val = np.max(np.abs(arr))
                            if 0.0001 < max_val < 10000:  # Reasonable range
                                
                                # Try as XY pairs
                                if arr.size % 2 == 0:
                                    xy = arr.reshape(-1, 2)
                                    x = xy[:, 0]
                                    y = xy[:, 1]
                                    
                                    # Check if x values are wavelengths (monotonic, reasonable range)
                                    if (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)) and \
                                       100 < np.mean(x) < 2000:  # UV-Vis range
                                        results.append({
                                            'name': f"{os.path.basename(filepath)}_{stream_name}",
                                            'wavelength': x,
                                            'absorbance': y,
                                            'method': f'XY_pairs_offset_{offset}'
                                        })
                                        print(f"  ✓ Found XY data: {len(x)} points, λ={x.min():.1f}-{x.max():.1f}nm")
                                        found_data = True
                                        break
                                
                                # Try as Y-only data (wavelengths might be in another stream)
                                if not found_data and arr.size > 100:
                                    # Common wavelength ranges for UV-Vis
                                    for start, end in [(190, 1100), (200, 800), (200, 600)]:
                                        if arr.size > 50:
                                            wavelengths = np.linspace(start, end, arr.size)
                                            results.append({
                                                'name': f"{os.path.basename(filepath)}_{stream_name}_assumed_wavelength",
                                                'wavelength': wavelengths,
                                                'absorbance': arr,
                                                'method': f'Y_only_offset_{offset}'
                                            })
                                            print(f"  ✓ Found Y data: {arr.size} points (wavelength assumed {start}-{end}nm)")
                                            found_data = True
                                            break
                    except Exception as e:
                        continue
                
                if found_data:
                    break
                    
            # Method 2: Try float64 if float32 didn't work
            if not found_data and len(stream_data) % 8 == 0:
                try:
                    arr = np.frombuffer(stream_data, dtype='<f8')  # little-endian float64
                    if arr.size > 10 and np.all(np.isfinite(arr)) and np.max(np.abs(arr)) < 10000:
                        if arr.size % 2 == 0:
                            xy = arr.reshape(-1, 2)
                            x = xy[:, 0]
                            y = xy[:, 1]
                            
                            if (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)) and \
                               100 < np.mean(x) < 2000:
                                results.append({
                                    'name': f"{os.path.basename(filepath)}_{stream_name}",
                                    'wavelength': x,
                                    'absorbance': y,
                                    'method': 'XY_pairs_float64'
                                })
                                print(f"  ✓ Found float64 XY data: {len(x)} points")
                except:
                    pass
                    
        except Exception as e:
            print(f"  × Error processing stream: {e}")
    
    ole.close()
    return results

def save_to_csv(data_dict, output_path):
    """Save extracted data to CSV file."""
    df = pd.DataFrame({
        'wavelength_nm': data_dict['wavelength'],
        'absorbance': data_dict['absorbance']
    })
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python shimadzu_spc_converter.py <spc_file> [output.csv]")
        print("\nThis script extracts UV-Vis data from Shimadzu OLE-format SPC files.")
        print("If no output file is specified, it will create one with _extracted.csv suffix.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Extract data
    try:
        results = extract_shimadzu_spc(input_file)
        
        if not results:
            print("\n❌ No spectroscopy data found in the file.")
            print("This might not be a Shimadzu SPC file, or it uses an unsupported format.")
            print("Try exporting as CSV/TXT from your instrument software.")
            sys.exit(1)
        
        # Save results
        print(f"\n📊 Found {len(results)} dataset(s)")
        
        for i, data in enumerate(results):
            # Generate output filename
            if len(sys.argv) > 2 and len(results) == 1:
                output_file = sys.argv[2]
            else:
                base_name = os.path.splitext(input_file)[0]
                suffix = f"_extracted_{i+1}" if len(results) > 1 else "_extracted"
                output_file = f"{base_name}{suffix}.csv"
            
            # Save to CSV
            save_to_csv(data, output_file)
            
            # Show preview
            print(f"\nDataset {i+1}:")
            print(f"  Method: {data['method']}")
            print(f"  Points: {len(data['wavelength'])}")
            print(f"  Wavelength range: {data['wavelength'].min():.1f} - {data['wavelength'].max():.1f} nm")
            print(f"  Absorbance range: {data['absorbance'].min():.4f} - {data['absorbance'].max():.4f}")
        
        print("\n✅ Conversion complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
