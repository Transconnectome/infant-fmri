import os
import argparse
import glob
import nibabel as nb
import numpy as np
import torch
from nipype.interfaces import fsl

def preprocess_subject(subject_dir, output_dir, fsl_output_type='NIFTI_GZ'):
    """
    Preprocess a single subject:
    1. Load fMRI data
    2. Apply FSL BET (Skull Stripping)
    3. Save as .pt files for SwiFT/Titans
    """
    subject_id = os.path.basename(subject_dir)
    print(f"Processing {subject_id}...")
    
    # Find functional runs
    func_files = glob.glob(os.path.join(subject_dir, 'func', '*bold.nii.gz'))
    
    for func_file in func_files:
        run_id = os.path.basename(func_file).split('_')[2] # e.g. run-1
        
        # 1. Skull Stripping with FSL BET
        bet = fsl.BET()
        bet.inputs.in_file = func_file
        bet.inputs.frac = 0.5 # Fractional intensity threshold
        bet.inputs.vertical_gradient = 0
        bet.inputs.mask = True
        bet.inputs.output_type = fsl_output_type
        
        # Output filename for BET
        bet_out_file = os.path.join(output_dir, subject_id, f"{subject_id}_{run_id}_brain.nii.gz")
        os.makedirs(os.path.dirname(bet_out_file), exist_ok=True)
        bet.inputs.out_file = bet_out_file
        
        try:
            print(f"Running BET on {func_file} -> {bet_out_file}")
            bet.run()
        except Exception as e:
            print(f"Error running BET on {func_file}: {e}")
            continue

        # 2. Convert to Tensors (Frame-wise)
        # Load the skull-stripped file
        img = nb.load(bet_out_file)
        data = img.get_fdata() # (X, Y, Z, T)
        
        # Save each frame
        # Structure: output_dir/subject_id/frame_X.pt
        save_root = os.path.join(output_dir, 'img', subject_id) # Using 'img' subfolder to match Dataset expectation
        os.makedirs(save_root, exist_ok=True)
        
        # Calculate voxel mean/std for normalization if needed
        voxel_mean = np.mean(data, axis=3)
        voxel_std = np.std(data, axis=3)
        torch.save(torch.from_numpy(voxel_mean), os.path.join(save_root, 'voxel_mean.pt'))
        torch.save(torch.from_numpy(voxel_std), os.path.join(save_root, 'voxel_std.pt'))
        
        for t in range(data.shape[3]):
            frame_data = data[:, :, :, t]
            # Normalize? SwiFT usually expects raw or normalized. 
            # Saving as float16 to save space
            tensor = torch.from_numpy(frame_data).half()
            torch.save(tensor, os.path.join(save_root, f"frame_{t}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory of raw BIDS dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory for preprocessed tensors")
    parser.add_argument("--subject", type=str, help="Specific subject to process (optional)")
    args = parser.parse_args()
    
    subjects = glob.glob(os.path.join(args.root, 'sub-*'))
    if args.subject:
        subjects = [s for s in subjects if os.path.basename(s) == args.subject]
    
    for subj_dir in subjects:
        preprocess_subject(subj_dir, args.output)
