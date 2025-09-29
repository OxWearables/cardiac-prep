#!/usr/bin/env python3
"""
Main script for the ECG and Accelerometer processing pipeline, can be run locally. 

This script discovers all .EDF files in a specified input directory,
distributes the processing of these files across all available CPU cores
using multiprocessing, and aggregates the results into a single summary CSV file.
"""
__author__ = "Awa Bator"

import os
import glob
import pandas as pd
import multiprocessing as mp
from proc_edf import procEDF_wrapper, init_worker
import time 

def main():
    """
    Main function to find EDF files and process them in parallel.
    """
    start_time = time.time()
    # TODO Point this to the directory containing your EDF files
    input_directory = "./input_data/"
    # TODO Point this to where you want summary CSV files to be saved
    output_directory = "./output/"
    # Number of CPU cores to use. os.cpu_count() uses all available cores.
    n_processes = os.cpu_count()

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all EDF files in the input directory
    edf_files = glob.glob(os.path.join(input_directory, "*.EDF" )) + glob.glob(os.path.join(input_directory, "*.edf"))
    
    if not edf_files:
        print(f"Error: No .edf or .EDF files found in '{input_directory}'")
        return
        
    print(f"Found {len(edf_files)} EDF files to process using {n_processes} cores.")

    ## Run Processing in Parallel 
    # This uses the same parallel processing logic as the original script
    with mp.Pool(processes=n_processes, initializer=init_worker) as pool:
        # pool.map applies the procEDF_wrapper function to each file in the edf_files list
        results = pool.map(procEDF_wrapper, edf_files)

    # Aggregate and Save Results 
    # Combine the summary DataFrames from all files into one
    df_info_all = pd.concat(results, ignore_index=True)

    # Save the aggregated summary file
    output_path = os.path.join(output_directory, "df_info_summary.csv.gz")
    df_info_all.to_csv(output_path, compression='gzip', index=False)

    end_time = time.time()
    total_duration = end_time - start_time
    print("\n-----------------------------------------")
    print(f"Processing complete!")
    print(f"   - Total files processed: {len(edf_files)}")
    print(f"   - Total time elapsed: {total_duration / 60:.2f} minutes ({total_duration:.2f} seconds)")
    print(f"   - Average time per file: {total_duration / len(edf_files):.2f} seconds")
    print(f"   - Aggregated summary saved to: {output_path}")
    print("-----------------------------------------")
    print(f"Aggregated summary saved to: {output_path}")
    print("Individual file outputs (plots, detailed CSVs) are in the main project directory.")


if __name__ == '__main__':
    main()
