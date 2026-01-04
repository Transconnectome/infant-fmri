# Data Processing Guide: fMRI Analysis Pipeline for Infant Neurodevelopment

**Comprehensive guide for preprocessing, analysis, and prediction using SwiFT framework**

---

## 📋 Overview

This guide details the complete data processing pipeline for transforming raw neonatal fMRI data into neurodevelopmental predictions using the SwiFT framework. The pipeline handles data from acquisition through final clinical predictions.

### Pipeline Stages
```
Raw NIfTI fMRI → Preprocessing → Template Registration → Feature Extraction → SwiFT Analysis → Predictions
```

---

## 🗂️ Data Organization

### Input Data Structure

#### **dHCP Dataset Organization**
```
dHCP_data/
├── derivatives/
│   ├── sub-CC00060XX01/
│   │   └── ses-12000/
│   │       └── func/
│   │           ├── sub-CC00060XX01_ses-12000_task-rest_bold.nii.gz
│   │           ├── sub-CC00060XX01_ses-12000_task-rest_bold.json
│   │           └── ...
│   └── ...
├── participants.tsv           # Subject demographics
└── phenotype/
    └── bayley_scores.csv      # Developmental outcomes
```

#### **Required File Formats**
- **fMRI Data**: NIfTI format (.nii.gz)
- **Demographics**: TSV format with BIDS compliance
- **Outcomes**: CSV with Bayley-III scores
- **Splits**: Text files defining train/validation/test sets

### Output Data Structure

#### **Processed Data Organization**
```
processed_data/
├── {dataset_name}_MNI_to_TRs/
│   ├── img/                   # Processed volume data
│   │   ├── sub-001/
│   │   │   ├── frame_0.pt     # Individual volume tensors
│   │   │   ├── frame_1.pt
│   │   │   └── ...
│   │   └── ...
│   └── metadata/
│       └── metafile.csv       # Subject-target mappings
└── group_ica/
    ├── group_ica_25.nii.gz    # Group ICA maps
    ├── group_ica_100.nii.gz
    └── connectivity_maps/     # Subject-specific connectivity
        ├── sub-001_ic_maps.nii.gz
        └── ...
```

---

## 🔧 Preprocessing Pipeline

### Step 1: Environment Setup

#### **Install Required Dependencies**
```bash
# Create conda environment
conda env create -f envs/py39.yaml
conda activate py39

# Additional neuroimaging tools
conda install -c conda-forge fsl ants nibabel nilearn

# Verify installations
python -c "import nibabel, nilearn, ants; print('All packages loaded')"
```

#### **Set Environment Variables**
```bash
# FSL Setup
export FSLDIR=/path/to/fsl
export PATH=${FSLDIR}/bin:${PATH}

# ANTs Setup
export ANTSPATH=/path/to/ants/bin/
export PATH=${ANTSPATH}:${PATH}

# Data paths
export RAW_DATA_DIR=/path/to/raw/data
export PROCESSED_DATA_DIR=/path/to/processed/data
```

### Step 2: Initial Data Validation

#### **Data Quality Checks**
```python
import nibabel as nib
import numpy as np
from pathlib import Path

def validate_fmri_data(fmri_path):
    """Validate fMRI data quality and structure."""

    # Load data
    img = nib.load(fmri_path)
    data = img.get_fdata()

    # Check dimensions
    assert len(data.shape) == 4, f"Expected 4D data, got {data.shape}"

    # Check temporal length
    n_timepoints = data.shape[3]
    assert n_timepoints >= 100, f"Too few timepoints: {n_timepoints}"

    # Check for NaN/infinite values
    assert np.isfinite(data).all(), "Data contains NaN or infinite values"

    # Check signal range
    mean_signal = np.mean(data[data > 0])
    assert mean_signal > 0, "No positive signal detected"

    print(f"✅ Data validation passed: {data.shape}, mean signal: {mean_signal:.2f}")

# Example usage
validate_fmri_data("path/to/sub-001_task-rest_bold.nii.gz")
```

#### **Subject Metadata Processing**
```python
import pandas as pd

def process_subject_metadata(participants_file, bayley_file, output_file):
    """Combine demographics and outcome data."""

    # Load data
    participants = pd.read_csv(participants_file, sep='\t')
    bayley = pd.read_csv(bayley_file)

    # Merge datasets
    metadata = participants.merge(bayley, on='participant_id', how='inner')

    # Quality checks
    print(f"Total subjects: {len(participants)}")
    print(f"Subjects with Bayley scores: {len(bayley)}")
    print(f"Complete data: {len(metadata)}")

    # Save processed metadata
    metadata.to_csv(output_file, index=False)
    return metadata

# Process metadata
metadata = process_subject_metadata(
    "participants.tsv",
    "bayley_scores.csv",
    "processed_metadata.csv"
)
```

### Step 3: fMRI Preprocessing

#### **Template Registration Pipeline**
```python
import ants
from pathlib import Path

def register_to_template(input_fmri, template_path, output_dir):
    """Register fMRI data to neonatal template."""

    subject_id = Path(input_fmri).stem.split('_')[0]
    output_dir = Path(output_dir) / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Processing {subject_id}...")
    fmri_img = ants.image_read(str(input_fmri))
    template_img = ants.image_read(str(template_path))

    # Compute mean image for registration
    mean_img = fmri_img.mean(axis=3)

    # Register to template
    registration = ants.registration(
        fixed=template_img,
        moving=mean_img,
        type_of_transform='SyN',
        verbose=True
    )

    # Apply transformation to all timepoints
    n_timepoints = fmri_img.shape[3]
    registered_data = []

    for t in range(n_timepoints):
        # Extract timepoint
        timepoint = ants.slice_image(fmri_img, axis=3, idx=t)

        # Apply transformation
        registered_tp = ants.apply_transforms(
            fixed=template_img,
            moving=timepoint,
            transformlist=registration['fwdtransforms']
        )

        registered_data.append(registered_tp.numpy())

    # Stack registered timepoints
    registered_4d = np.stack(registered_data, axis=3)

    # Save registered data
    registered_img = ants.from_numpy(registered_4d)
    registered_img.set_spacing(template_img.spacing)
    registered_img.set_origin(template_img.origin)

    output_file = output_dir / f"{subject_id}_registered.nii.gz"
    ants.image_write(registered_img, str(output_file))

    return str(output_file)

# Example usage
template_path = "templates/dHCP_40w_T1.nii.gz"
registered_file = register_to_template(
    "sub-001_task-rest_bold.nii.gz",
    template_path,
    "registered_data"
)
```

#### **Volume Normalization and Conversion**
```python
import torch
import numpy as np
from scipy import stats

def process_and_save_volumes(registered_fmri, output_dir, normalize_method='zscore'):
    """Process fMRI volumes and save as individual tensors."""

    # Load registered data
    img = nib.load(registered_fmri)
    data = img.get_fdata()

    # Extract subject ID
    subject_id = Path(registered_fmri).stem.split('_')[0]
    subject_dir = Path(output_dir) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Background mask (non-zero voxels)
    background_mask = np.mean(data, axis=3) > 0

    n_timepoints = data.shape[3]

    for t in range(n_timepoints):
        volume = data[:, :, :, t].copy()

        if normalize_method == 'zscore':
            # Z-normalization of non-background voxels
            foreground_voxels = volume[background_mask]
            if len(foreground_voxels) > 0:
                mean_val = np.mean(foreground_voxels)
                std_val = np.std(foreground_voxels)
                if std_val > 0:
                    volume[background_mask] = (volume[background_mask] - mean_val) / std_val

        elif normalize_method == 'minmax':
            # Min-max normalization
            foreground_voxels = volume[background_mask]
            if len(foreground_voxels) > 0:
                min_val = np.min(foreground_voxels)
                max_val = np.max(foreground_voxels)
                if max_val > min_val:
                    volume[background_mask] = (volume[background_mask] - min_val) / (max_val - min_val)

        # Convert to float16 for storage efficiency
        volume_tensor = torch.from_numpy(volume.astype(np.float16))

        # Save individual volume
        output_file = subject_dir / f"frame_{t}.pt"
        torch.save(volume_tensor, output_file)

    print(f"✅ Processed {n_timepoints} volumes for {subject_id}")
    return subject_dir

# Process all subjects
def batch_process_volumes(registered_dir, output_dir, normalize_method='zscore'):
    """Batch process all registered volumes."""

    registered_files = list(Path(registered_dir).glob("*/*_registered.nii.gz"))

    for fmri_file in registered_files:
        try:
            process_and_save_volumes(
                str(fmri_file),
                output_dir,
                normalize_method
            )
        except Exception as e:
            print(f"❌ Error processing {fmri_file}: {e}")
            continue

    print(f"✅ Batch processing complete: {len(registered_files)} subjects")

# Run batch processing
batch_process_volumes(
    "registered_data",
    "processed_data/dHCP_MNI_to_TRs/img",
    normalize_method='zscore'
)
```

---

## 🧬 Group ICA Analysis

### Step 1: Subject Selection for ICA

#### **Select Healthy Development Subjects**
```python
import pandas as pd

def select_healthy_subjects(metadata_file, output_file, criteria=None):
    """Select subjects with typical development for Group ICA."""

    # Default criteria for healthy development
    if criteria is None:
        criteria = {
            'cognitive_score': 85,
            'language_score': 85,
            'motor_score': 85,
            'min_ga': 36,  # weeks
            'max_ga': 44   # weeks
        }

    # Load metadata
    df = pd.read_csv(metadata_file)

    # Apply selection criteria
    healthy_mask = (
        (df['cognitive_composite'] >= criteria['cognitive_score']) &
        (df['language_composite'] >= criteria['language_score']) &
        (df['motor_composite'] >= criteria['motor_score']) &
        (df['scan_ga'] >= criteria['min_ga']) &
        (df['scan_ga'] <= criteria['max_ga']) &
        (df['has_fmri'] == True)
    )

    healthy_subjects = df[healthy_mask]

    print(f"Total subjects: {len(df)}")
    print(f"Healthy subjects: {len(healthy_subjects)}")
    print(f"Selection rate: {len(healthy_subjects)/len(df)*100:.1f}%")

    # Randomly sample if too many subjects
    if len(healthy_subjects) > 100:
        healthy_subjects = healthy_subjects.sample(n=100, random_state=42)
        print(f"Randomly sampled: {len(healthy_subjects)} subjects")

    # Save selected subjects
    healthy_subjects.to_csv(output_file, index=False)
    return healthy_subjects

# Select healthy subjects for ICA
healthy_subjects = select_healthy_subjects(
    "processed_metadata.csv",
    "healthy_subjects_for_ica.csv"
)
```

### Step 2: Group ICA Computation

#### **MIGP and Group ICA Pipeline**
```bash
#!/bin/bash
# group_ica_pipeline.sh

# Set FSL environment
source ${FSLDIR}/etc/fslconf/fsl.sh

# Parameters
SUBJECT_LIST="healthy_subjects_for_ica.csv"
OUTPUT_DIR="group_ica_analysis"
N_COMPONENTS_LIST="25 100"
N_PCA=1200

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Create 4D merged data for selected subjects
echo "Creating merged 4D data..."
python scripts/create_merged_data.py \
    --subject_list ${SUBJECT_LIST} \
    --data_dir "processed_data/dHCP_MNI_to_TRs/img" \
    --output ${OUTPUT_DIR}/merged_data.nii.gz

# Run MIGP (MELODIC's Incremental Group-PCA)
echo "Running MIGP..."
melodic -i ${OUTPUT_DIR}/merged_data.nii.gz \
        -o ${OUTPUT_DIR}/migp_output \
        --migp \
        --migpN=${N_PCA} \
        --report

# Run Group ICA for different component numbers
for N_COMP in ${N_COMPONENTS_LIST}; do
    echo "Running Group ICA with ${N_COMP} components..."

    melodic -i ${OUTPUT_DIR}/migp_output/melodic_mix \
            -o ${OUTPUT_DIR}/group_ica_${N_COMP} \
            --dim=${N_COMP} \
            --report

    # Copy important outputs
    cp ${OUTPUT_DIR}/group_ica_${N_COMP}/melodic_IC.nii.gz \
       ${OUTPUT_DIR}/group_ica_${N_COMP}_maps.nii.gz
done

echo "Group ICA analysis complete!"
```

#### **Create Merged Data Script**
```python
# scripts/create_merged_data.py
import pandas as pd
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse

def create_merged_data(subject_list, data_dir, output_file, sequence_length=20):
    """Create merged 4D data for Group ICA analysis."""

    # Load subject list
    subjects_df = pd.read_csv(subject_list)
    subject_ids = subjects_df['participant_id'].tolist()

    # Collect all volume data
    all_volumes = []
    valid_subjects = []

    for subject_id in subject_ids:
        subject_dir = Path(data_dir) / subject_id

        if not subject_dir.exists():
            print(f"Warning: No data found for {subject_id}")
            continue

        # Load subject volumes
        volume_files = sorted(subject_dir.glob("frame_*.pt"))

        if len(volume_files) < sequence_length:
            print(f"Warning: Insufficient volumes for {subject_id}")
            continue

        # Load first sequence_length volumes
        subject_volumes = []
        for i in range(sequence_length):
            volume = torch.load(volume_files[i])
            subject_volumes.append(volume.numpy())

        # Stack volumes for this subject
        subject_4d = np.stack(subject_volumes, axis=3)  # (H, W, D, T)
        all_volumes.append(subject_4d)
        valid_subjects.append(subject_id)

    # Merge all subjects
    merged_data = np.concatenate(all_volumes, axis=3)  # (H, W, D, T_total)

    print(f"Merged data shape: {merged_data.shape}")
    print(f"Valid subjects: {len(valid_subjects)}")

    # Create NIfTI image
    affine = np.eye(4)  # Identity affine matrix
    merged_img = nib.Nifti1Image(merged_data, affine)

    # Save merged data
    nib.save(merged_img, output_file)

    # Save subject mapping
    mapping_file = Path(output_file).parent / "subject_mapping.csv"
    mapping_df = pd.DataFrame({
        'subject_id': valid_subjects,
        'timepoint_start': np.arange(len(valid_subjects)) * sequence_length,
        'timepoint_end': (np.arange(len(valid_subjects)) + 1) * sequence_length
    })
    mapping_df.to_csv(mapping_file, index=False)

    return merged_data, valid_subjects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_list', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence_length', type=int, default=20)

    args = parser.parse_args()

    create_merged_data(
        args.subject_list,
        args.data_dir,
        args.output,
        args.sequence_length
    )
```

### Step 3: Dual Regression and Connectivity Analysis

#### **Subject-Specific ICA Maps**
```python
import nibabel as nib
import numpy as np
from pathlib import Path

def dual_regression_analysis(group_ica_maps, subject_data_dir, output_dir, n_components=25):
    """Perform dual regression to get subject-specific ICA maps."""

    # Load group ICA maps
    group_maps = nib.load(group_ica_maps).get_fdata()  # (H, W, D, n_components)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject
    subject_dirs = list(Path(subject_data_dir).iterdir())

    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        print(f"Processing {subject_id}...")

        # Load subject data
        volume_files = sorted(subject_dir.glob("frame_*.pt"))

        if len(volume_files) == 0:
            continue

        # Stack subject volumes
        subject_volumes = []
        for vol_file in volume_files:
            volume = torch.load(vol_file).numpy()
            subject_volumes.append(volume)

        subject_4d = np.stack(subject_volumes, axis=3)  # (H, W, D, T)

        # Reshape for dual regression
        H, W, D, T = subject_4d.shape
        subject_2d = subject_4d.reshape(-1, T)  # (voxels, time)
        group_2d = group_maps.reshape(-1, n_components)  # (voxels, components)

        # Step 1: Spatial regression (get timeseries)
        # Solve: subject_2d = group_2d @ timeseries + noise
        try:
            timeseries = np.linalg.lstsq(group_2d, subject_2d, rcond=None)[0]  # (components, time)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix for {subject_id}")
            continue

        # Step 2: Temporal regression (get subject-specific maps)
        # Solve: subject_2d = spatial_maps @ timeseries + noise
        try:
            spatial_maps = np.linalg.lstsq(timeseries.T, subject_2d.T, rcond=None)[0]  # (components, voxels)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix for {subject_id}")
            continue

        # Reshape back to 3D
        spatial_maps_4d = spatial_maps.reshape(n_components, H, W, D).transpose(1, 2, 3, 0)

        # Save subject-specific maps
        output_file = output_dir / f"{subject_id}_ic_maps.nii.gz"
        affine = np.eye(4)
        maps_img = nib.Nifti1Image(spatial_maps_4d, affine)
        nib.save(maps_img, output_file)

        # Save timeseries
        timeseries_file = output_dir / f"{subject_id}_timeseries.npy"
        np.save(timeseries_file, timeseries)

# Run dual regression
dual_regression_analysis(
    "group_ica_analysis/group_ica_25_maps.nii.gz",
    "processed_data/dHCP_MNI_to_TRs/img",
    "group_ica_analysis/dual_regression_25"
)
```

#### **Seed-to-Voxel Connectivity**
```python
def compute_connectivity_maps(subject_timeseries, group_ica_maps, output_dir):
    """Compute seed-to-voxel connectivity maps."""

    # Load group ICA maps for reference
    group_maps = nib.load(group_ica_maps).get_fdata()
    H, W, D, n_components = group_maps.shape

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject's timeseries
    timeseries_files = list(Path(subject_timeseries).glob("*_timeseries.npy"))

    for ts_file in timeseries_files:
        subject_id = ts_file.stem.replace('_timeseries', '')
        print(f"Computing connectivity for {subject_id}...")

        # Load subject timeseries
        timeseries = np.load(ts_file)  # (components, time)

        # Load subject volume data for correlation
        subject_dir = Path("processed_data/dHCP_MNI_to_TRs/img") / subject_id
        volume_files = sorted(subject_dir.glob("frame_*.pt"))

        if len(volume_files) != timeseries.shape[1]:
            print(f"Warning: Timeseries length mismatch for {subject_id}")
            continue

        # Load and stack volumes
        subject_volumes = []
        for vol_file in volume_files:
            volume = torch.load(vol_file).numpy()
            subject_volumes.append(volume)

        subject_4d = np.stack(subject_volumes, axis=3)
        subject_2d = subject_4d.reshape(-1, timeseries.shape[1])

        # Compute seed-to-voxel correlations for each IC
        connectivity_maps = np.zeros((H * W * D, n_components))

        for ic in range(n_components):
            ic_timeseries = timeseries[ic, :]

            # Correlate IC timeseries with all voxels
            correlations = np.corrcoef(ic_timeseries, subject_2d)[0, 1:]
            connectivity_maps[:, ic] = correlations

        # Reshape and save
        connectivity_4d = connectivity_maps.reshape(H, W, D, n_components)

        output_file = output_dir / f"{subject_id}_connectivity.nii.gz"
        affine = np.eye(4)
        conn_img = nib.Nifti1Image(connectivity_4d, affine)
        nib.save(conn_img, output_file)

# Compute connectivity maps
compute_connectivity_maps(
    "group_ica_analysis/dual_regression_25",
    "group_ica_analysis/group_ica_25_maps.nii.gz",
    "group_ica_analysis/connectivity_maps"
)
```

---

## 🧠 SwiFT Model Training

### Step 1: Data Module Setup

#### **Create Metadata File**
```python
def create_swift_metadata(processed_metadata, volume_dir, output_file):
    """Create metadata file for SwiFT training."""

    # Load processed metadata
    df = pd.read_csv(processed_metadata)

    # Check for available processed data
    volume_path = Path(volume_dir)
    available_subjects = [d.name for d in volume_path.iterdir() if d.is_dir()]

    # Filter for subjects with processed data
    df_available = df[df['participant_id'].isin(available_subjects)].copy()

    # Rename columns to match SwiFT expectations
    df_swift = df_available.rename(columns={
        'participant_id': 'Subject',
        'cognitive_composite': 'cognitive',
        'language_composite': 'language',
        'motor_composite': 'motor',
        'scan_age': 'Age',
        'sex': 'Gender'
    })

    # Add required columns
    df_swift['Path'] = df_swift['Subject'].apply(lambda x: str(volume_path / x))

    # Save SwiFT-compatible metadata
    df_swift.to_csv(output_file, index=False)

    print(f"SwiFT metadata created: {len(df_swift)} subjects")
    return df_swift

# Create metadata for SwiFT
swift_metadata = create_swift_metadata(
    "processed_metadata.csv",
    "processed_data/dHCP_MNI_to_TRs/img",
    "processed_data/dHCP_MNI_to_TRs/metadata/metafile.csv"
)
```

### Step 2: Training Configuration

#### **Basic Training Commands**
```bash
#!/bin/bash
# training_scripts/train_swift.sh

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Multi-GPU training

# Data paths
DATA_PATH="processed_data/dHCP_MNI_to_TRs"
SPLITS_PATH="data/splits/dHCP"
OUTPUT_DIR="experiments/swift_training"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Single-label Raw fMRI training
echo "Training Single-label Raw fMRI model..."
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive \
    --downstream_task_type classification \
    --model swin4d_ver7 \
    --image_path ${DATA_PATH} \
    --batch_size 4 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --sequence_length 50 \
    --devices 4 \
    --default_root_dir ${OUTPUT_DIR}/single_raw \
    --loggername tensorboard \
    --augment_during_training

# Multi-label Raw fMRI training
echo "Training Multi-label Raw fMRI model..."
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive,language,motor \
    --downstream_task_type classification \
    --model swin4d_ver7 \
    --image_path ${DATA_PATH} \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --max_epochs 150 \
    --sequence_length 50 \
    --devices 4 \
    --default_root_dir ${OUTPUT_DIR}/multi_raw \
    --loggername tensorboard \
    --augment_during_training

echo "Training complete!"
```

#### **ICA Feature Training**
```bash
#!/bin/bash
# training_scripts/train_ica.sh

# ICA connectivity data path
ICA_DATA_PATH="group_ica_analysis/connectivity_maps"

# Single-label ICA training
echo "Training Single-label ICA model..."
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive \
    --downstream_task_type classification \
    --model swin4d_ver7 \
    --image_path ${ICA_DATA_PATH} \
    --use_ica_features \
    --ica_components 25 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --sequence_length 42 \
    --devices 4 \
    --default_root_dir experiments/ica_training/single \
    --loggername tensorboard

# Multi-label ICA training (best performance)
echo "Training Multi-label ICA model..."
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive,language,motor \
    --downstream_task_type classification \
    --model swin4d_ver7 \
    --image_path ${ICA_DATA_PATH} \
    --use_ica_features \
    --ica_components 25 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --max_epochs 150 \
    --sequence_length 42 \
    --devices 4 \
    --default_root_dir experiments/ica_training/multi \
    --loggername tensorboard \
    --augment_during_training

echo "ICA training complete!"
```

### Step 3: Hyperparameter Optimization

#### **Optuna Integration**
```python
# scripts/hyperparameter_optimization.py
import optuna
import subprocess
import json
from pathlib import Path

def objective(trial):
    """Objective function for hyperparameter optimization."""

    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    sequence_length = trial.suggest_categorical('sequence_length', [20, 50, 100])
    embed_dim = trial.suggest_categorical('embed_dim', [24, 36, 48])

    # Create training command
    cmd = [
        'python', 'project/main.py',
        '--dataset_name', 'dHCP',
        '--downstream_task', 'cognitive,language,motor',
        '--model', 'swin4d_ver7',
        '--learning_rate', str(learning_rate),
        '--batch_size', str(batch_size),
        '--sequence_length', str(sequence_length),
        '--embed_dim', str(embed_dim),
        '--max_epochs', '50',  # Shorter for optimization
        '--devices', '1',
        '--default_root_dir', f'experiments/optuna/trial_{trial.number}',
        '--loggername', 'tensorboard'
    ]

    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        # Parse validation accuracy from logs
        log_file = Path(f'experiments/optuna/trial_{trial.number}') / 'version_0' / 'metrics.csv'

        if log_file.exists():
            import pandas as pd
            metrics = pd.read_csv(log_file)
            val_acc = metrics['val_balanced_acc'].max()
            return val_acc
        else:
            return 0.0

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

# Run optimization
def run_optimization(n_trials=50):
    """Run hyperparameter optimization."""

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")

    # Save results
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    run_optimization(n_trials=30)
```

---

## 📊 Model Evaluation and Interpretation

### Step 1: Model Testing

#### **Evaluation Script**
```python
# scripts/evaluate_model.py
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from pathlib import Path

def evaluate_trained_model(checkpoint_path, test_data_path, output_dir):
    """Evaluate trained SwiFT model on test set."""

    # Load trained model
    from project.module.pl_classifier import LitClassifier
    model = LitClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Load test data
    from project.module.utils.data_module import fMRIDataModule
    data_module = fMRIDataModule(
        dataset_name='dHCP',
        image_path=test_data_path,
        batch_size=8,
        num_workers=4
    )
    data_module.setup('test')

    # Get predictions
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            images, targets = batch
            outputs = model(images)

            # Convert to probabilities for classification
            if model.task_type == 'classification':
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                preds = outputs

            predictions.append(preds.cpu().numpy())
            ground_truth.append(targets.cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # Compute metrics
    results = {}

    if model.task_type == 'classification':
        # Classification metrics
        for i, task in enumerate(['cognitive', 'language', 'motor']):
            if predictions.shape[1] > i:
                acc = accuracy_score(ground_truth[:, i], predictions[:, i])
                bal_acc = balanced_accuracy_score(ground_truth[:, i], predictions[:, i])
                auc = roc_auc_score(ground_truth[:, i], predictions[:, i])

                results[f'{task}_accuracy'] = acc
                results[f'{task}_balanced_accuracy'] = bal_acc
                results[f'{task}_auc'] = auc

    else:
        # Regression metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr

        for i, task in enumerate(['cognitive', 'language', 'motor']):
            if predictions.shape[1] > i:
                mae = mean_absolute_error(ground_truth[:, i], predictions[:, i])
                mse = mean_squared_error(ground_truth[:, i], predictions[:, i])
                corr, _ = pearsonr(ground_truth[:, i], predictions[:, i])

                results[f'{task}_mae'] = mae
                results[f'{task}_mse'] = mse
                results[f'{task}_correlation'] = corr

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions for further analysis
    np.save(output_path / 'predictions.npy', predictions)
    np.save(output_path / 'ground_truth.npy', ground_truth)

    return results

# Evaluate best model
results = evaluate_trained_model(
    "experiments/ica_training/multi/checkpoints/best.ckpt",
    "processed_data/dHCP_MNI_to_TRs",
    "evaluation_results"
)
```

### Step 2: Interpretation Analysis

#### **Integrated Gradients Analysis**
```python
# scripts/interpretation_analysis.py
from captum import IntegratedGradients
from captum.noise_tunnel import SmoothGrad
import torch
import numpy as np

def generate_attribution_maps(model, test_loader, output_dir, task_names):
    """Generate attribution maps using Integrated Gradients."""

    # Set up interpretation method
    ig = IntegratedGradients(model)
    sg = SmoothGrad(ig)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_attributions = {task: [] for task in task_names}

    model.eval()

    for batch_idx, (images, targets) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}...")

        # Only process correctly classified samples
        with torch.no_grad():
            predictions = model(images)
            pred_classes = (torch.sigmoid(predictions) > 0.5).float()
            correct_mask = (pred_classes == targets).all(dim=1)

        if not correct_mask.any():
            continue

        # Get correctly classified samples
        correct_images = images[correct_mask]
        correct_targets = targets[correct_mask]

        # Generate attributions for each task
        for task_idx, task_name in enumerate(task_names):
            # Create baseline (zeros)
            baseline = torch.zeros_like(correct_images)

            # Compute attributions
            attributions = sg.attribute(
                correct_images,
                baseline,
                target=task_idx,
                n_samples=25,
                stdevs=0.1
            )

            # Store attributions
            all_attributions[task_name].append(attributions.cpu().numpy())

    # Average attributions across subjects
    for task_name in task_names:
        if all_attributions[task_name]:
            task_attributions = np.concatenate(all_attributions[task_name], axis=0)
            mean_attribution = np.mean(task_attributions, axis=0)

            # Save attribution map
            np.save(output_dir / f'{task_name}_attribution.npy', mean_attribution)

            # Save as NIfTI for visualization
            import nibabel as nib
            affine = np.eye(4)
            attr_img = nib.Nifti1Image(mean_attribution[0], affine)  # Remove batch dimension
            nib.save(attr_img, output_dir / f'{task_name}_attribution.nii.gz')

    print("Attribution analysis complete!")

# Generate interpretation maps
from project.module.pl_classifier import LitClassifier

model = LitClassifier.load_from_checkpoint("experiments/ica_training/multi/checkpoints/best.ckpt")
test_loader = data_module.test_dataloader()

generate_attribution_maps(
    model,
    test_loader,
    "interpretation_results",
    ['cognitive', 'language', 'motor']
)
```

---

## 🔧 Troubleshooting and Optimization

### Common Issues and Solutions

#### **Memory Issues**
```bash
# Reduce batch size
--batch_size 2

# Use gradient checkpointing
--gradient_checkpointing True

# Use mixed precision training
--precision 16

# Reduce sequence length
--sequence_length 20
```

#### **Training Instability**
```bash
# Lower learning rate
--learning_rate 0.0001

# Add gradient clipping
--grad_clip True

# Use different optimizer
--optimizer SGD --momentum 0.9
```

#### **Data Loading Errors**
```python
# Check data paths
import os
print("Data directory exists:", os.path.exists("processed_data/dHCP_MNI_to_TRs"))

# Verify file format
import torch
test_file = "processed_data/dHCP_MNI_to_TRs/img/sub-001/frame_0.pt"
if os.path.exists(test_file):
    data = torch.load(test_file)
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
```

### Performance Optimization

#### **Data Loading Optimization**
```python
# Use more workers for data loading
--num_workers 8

# Pin memory for GPU training
--pin_memory True

# Use faster data format (already using .pt files)
```

#### **Training Optimization**
```bash
# Distributed training
--strategy ddp --devices 4

# Compile model (PyTorch 2.0+)
--compile True

# Use optimized attention
--use_flash_attention True
```

---

## 📚 Best Practices and Guidelines

### Data Processing Checklist

- [ ] Validate raw data quality and completeness
- [ ] Verify subject metadata and outcome measurements
- [ ] Apply consistent preprocessing pipeline
- [ ] Quality check registered data
- [ ] Validate ICA component biological plausibility
- [ ] Ensure balanced train/validation/test splits
- [ ] Document all processing parameters

### Training Best Practices

- [ ] Start with smaller models for debugging
- [ ] Use cross-validation for robust evaluation
- [ ] Monitor training/validation curves for overfitting
- [ ] Save checkpoints regularly
- [ ] Log hyperparameters and metrics
- [ ] Validate on held-out test set only once

### Clinical Translation Readiness

- [ ] Validate on external datasets
- [ ] Assess bias across demographic groups
- [ ] Compute confidence intervals for predictions
- [ ] Generate interpretable visualizations
- [ ] Document limitations and failure modes
- [ ] Prepare clinical user documentation

---

This comprehensive guide provides the foundation for processing neonatal fMRI data and training SwiFT models for neurodevelopmental prediction. Follow the pipeline systematically and adapt parameters based on your specific dataset characteristics.