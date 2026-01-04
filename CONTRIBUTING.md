# Contributing to SwiFT Infant Neurodevelopment Project

Welcome to the SwiFT infant neurodevelopment research project! This guide will help you contribute effectively to our codebase and research efforts.

---

## 🎯 Project Overview

SwiFT (Swin 4D fMRI Transformer) for infant neurodevelopment is a deep learning framework that predicts early neurodevelopmental outcomes from neonatal fMRI data using the Developing Human Connectome Project (dHCP) dataset.

**Key Research Goals:**
- Predict Bayley-III composite scores (cognitive, language, motor) from neonatal fMRI
- Enable early intervention through risk prediction at birth
- Advance understanding of early brain development patterns

---

## 🚀 Quick Start for Developers

### Prerequisites
- Python 3.9.12+
- CUDA-capable GPU (tested on 8x RTX 3090)
- 32GB+ RAM recommended
- Git with LFS support

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Transconnectome/infant-fmri.git
cd infant-fmri

# Create conda environment
conda env create -f envs/py39.yaml
conda activate py39

# Verify installation
python test/module_test_swin4d.py
```

### Development Dependencies
```bash
# Additional dev tools
pip install black isort flake8 pytest pytest-cov jupyter
```

---

## 📁 Repository Structure

```
infant-fmri/
├── project/                    # Main codebase
│   ├── main.py                # Training entry point
│   ├── module/
│   │   ├── pl_classifier.py   # PyTorch Lightning module
│   │   ├── models/            # Model architectures
│   │   │   ├── swin4d_transformer_ver7.py
│   │   │   ├── patchembedding.py
│   │   │   └── ...
│   │   └── utils/             # Utility modules
│   │       ├── data_module.py
│   │       ├── data_preprocess_and_load/
│   │       └── ...
├── paper/                     # Synchronized manuscript
│   ├── bookchapter.tex        # Main LaTeX source
│   └── img/                   # Research figures
├── data/splits/               # Dataset split definitions
├── pretrained_models/         # Model checkpoints
├── sample_scripts/            # Example training scripts
├── test/                      # Unit tests
├── interpretation/            # Model interpretability
└── envs/                      # Environment configs
```

---

## 🛠️ Development Workflow

### 1. Setting up Development Environment

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Install development hooks (recommended)
pre-commit install

# Run tests to ensure everything works
pytest test/
```

### 2. Code Style Guidelines

**Python Code Style:**
```bash
# Format code
black project/
isort project/

# Check linting
flake8 project/ --max-line-length=88 --ignore=E203,W503
```

**Key Conventions:**
- Use type hints for function signatures
- Follow PyTorch Lightning patterns for model modules
- Document complex functions with docstrings
- Keep functions focused and modular

### 3. Testing Requirements

**Unit Tests:**
```bash
# Run all tests
pytest test/ -v

# Run with coverage
pytest test/ --cov=project/ --cov-report=html
```

**Test Categories:**
- `test_model_*.py`: Model architecture tests
- `test_data_*.py`: Data loading and preprocessing tests
- `test_training_*.py`: Training pipeline tests

**Required Test Coverage:**
- New model components: 90%+ coverage
- Data processing functions: 85%+ coverage
- Utility functions: 80%+ coverage

---

## 🧪 Model Development

### Adding New Model Components

1. **Create model in `project/module/models/`:**
```python
# example_model.py
import torch
import torch.nn as nn
from .utils import ModelUtils

class NewModelComponent(nn.Module):
    """Brief description of model component.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension

    Example:
        >>> model = NewModelComponent(128, 64)
        >>> output = model(input_tensor)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

2. **Add corresponding test:**
```python
# test/test_new_model.py
import torch
import pytest
from project.module.models.example_model import NewModelComponent

def test_new_model_component():
    model = NewModelComponent(128, 64)
    x = torch.randn(4, 128)
    output = model(x)
    assert output.shape == (4, 64)
```

3. **Integration with main framework:**
```python
# Update pl_classifier.py if needed
# Update main.py argument parsing
# Update configuration files
```

### Data Processing Contributions

**Adding New Datasets:**

1. **Create dataset class in `project/module/utils/data_preprocess_and_load/datasets.py`:**
```python
class NewDataset(BaseDataset):
    """New dataset for fMRI analysis."""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path

    def __getitem__(self, idx: int):
        # Implement data loading logic
        pass
```

2. **Add preprocessing pipeline:**
```python
# Add to preprocessing.py
def preprocess_new_dataset(input_path: str, output_path: str):
    """Preprocess new dataset to SwiFT format."""
    pass
```

3. **Update data module:**
```python
# Modify data_module.py to include new dataset
```

---

## 📊 Experiment Management

### Training Configuration

**Standard Training Script:**
```bash
# Basic training
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive \
    --model swin4d_ver7 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --max_epochs 100 \
    --devices 1

# Multi-label with ICA
python project/main.py \
    --dataset_name dHCP \
    --downstream_task cognitive,language,motor \
    --model swin4d_ver7 \
    --use_ica_features \
    --learning_rate 0.0005 \
    --batch_size 4 \
    --max_epochs 150
```

### Hyperparameter Guidelines

**Recommended Starting Points:**
```python
# Model Architecture
embed_dim = [24, 36, 48]  # Start with 24
depths = [2, 2, 6, 2]     # Standard configuration
num_heads = [3, 6, 12, 24]  # Proportional to embed_dim
window_size = [4, 4, 4, 4]  # Standard 4D windows

# Training
learning_rate = [0.001, 0.0005, 0.0001]  # Start with 0.001
batch_size = [4, 8, 16]  # Limited by GPU memory
sequence_length = [20, 50, 100]  # 50 often optimal
```

### Logging and Monitoring

**Neptune Integration:**
```bash
# Set up Neptune logging
export NEPTUNE_API_TOKEN="your-token"

python project/main.py \
    --loggername neptune \
    --project_name your-username/infant-fmri \
    --downstream_task cognitive
```

**TensorBoard Alternative:**
```bash
# Local logging
python project/main.py \
    --loggername tensorboard \
    --default_root_dir ./experiments/
```

---

## 📝 Paper and Documentation

### Overleaf Synchronization

**Setup (if not already configured):**
```bash
# Check current sync status
./sync_paper.sh status

# Pull latest changes from Overleaf
./sync_paper.sh pull

# After making local changes to paper/
git add paper/
git commit -m "Update paper content"
./sync_paper.sh push
```

**Contribution Workflow:**
1. Pull latest changes: `./sync_paper.sh pull`
2. Make edits in `paper/bookchapter.tex`
3. Test LaTeX compilation locally
4. Commit changes: `git add paper/ && git commit -m "description"`
5. Push to Overleaf: `./sync_paper.sh push`

### Documentation Updates

**Required Documentation for New Features:**
1. **Code Documentation**: Inline docstrings and comments
2. **API Documentation**: Update relevant `.md` files
3. **Usage Examples**: Add to sample scripts or notebooks
4. **Test Coverage**: Comprehensive unit tests

---

## 🔬 Research Contributions

### Experimental Design

**Adding New Experiments:**

1. **Hypothesis Definition:**
   - Clear research question
   - Expected outcomes
   - Statistical analysis plan

2. **Implementation:**
   - Extend existing model or create new variant
   - Implement proper evaluation metrics
   - Design appropriate baselines

3. **Validation:**
   - Cross-validation strategy
   - Statistical significance testing
   - Ablation studies

### Interpretability Analysis

**Adding New Interpretation Methods:**

```python
# interpretation/new_method.py
import torch
from captum import IntegratedGradients

class NewInterpretationMethod:
    """New method for model interpretation."""

    def __init__(self, model):
        self.model = model

    def attribute(self, inputs, targets):
        """Generate attribution maps."""
        # Implementation here
        pass
```

---

## 🧬 Clinical Integration

### Biomarker Validation

**Requirements for Clinical Features:**
1. **Neurobiological Plausibility**: Literature support
2. **Statistical Validation**: Proper significance testing
3. **Clinical Relevance**: Connection to developmental outcomes
4. **Interpretability**: Explainable to clinicians

### Validation Studies

**Clinical Validation Checklist:**
- [ ] Appropriate patient cohort
- [ ] Ethical approval documentation
- [ ] Clinical endpoint definition
- [ ] Statistical power analysis
- [ ] Bias assessment and mitigation

---

## 📋 Pull Request Guidelines

### PR Requirements

**Before Submitting:**
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changes tested on sample data
- [ ] No performance regressions

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Performance benchmarks run

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

### Review Process

1. **Automatic Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainer review for quality and consistency
3. **Research Review**: Scientific validity for research contributions
4. **Final Testing**: Integration testing before merge

---

## 🆘 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Research questions and general help
- **Direct Contact**: Reach out to maintainers for urgent issues

### Common Issues

**GPU Memory Issues:**
```bash
# Reduce batch size
--batch_size 2

# Use gradient checkpointing
--gradient_checkpointing

# Mixed precision training
--precision 16
```

**Data Loading Errors:**
```bash
# Check data path
--image_path /path/to/preprocessed/data

# Verify split files
ls data/splits/dHCP/
```

---

## 🏆 Recognition

### Contribution Types

**Code Contributions:**
- New model architectures
- Performance improvements
- Bug fixes and optimizations

**Research Contributions:**
- Novel experiments and analyses
- Clinical validation studies
- Interpretability improvements

**Documentation Contributions:**
- Tutorial creation
- API documentation
- Clinical guides

**Community Contributions:**
- Issue triage and support
- Code reviews
- Testing and validation

---

## 📄 License and Attribution

This project is released under [LICENSE]. Contributors retain copyright over their contributions while granting the project rights to use and distribute the work.

**Citation Requirements:**
```bibtex
@article{styll2024swift,
  title={Swin fMRI Transformer Predicts Early Neurodevelopmental Outcomes from Neonatal fMRI},
  author={Styll, Patrick and Kim, Dowon and Cha, Jiook},
  journal={[Journal Name]},
  year={2024}
}
```

---

Thank you for contributing to advancing early neurodevelopmental prediction and improving outcomes for infants worldwide! 🍼🧠✨