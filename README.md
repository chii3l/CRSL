# CRSL

Official research code for **CRSL: Observation-Constrained Component-Resolved
Spectral Lifting for PPG-Based Concentration Estimation**.

CRSL augments an unchanged temporal backbone with a wavelength-domain path. It
lifts sparse multichannel PPG observations into smooth role-specific latent
spectra, projects the fused spectrum through a fixed dataset-specific
observation matrix, and applies a gated target-specific correction to the
baseline prediction.

## Repository layout

~~~text
CRSL/
|-- src/
|   |-- models/                 # CRSL and temporal backbones
|   |-- run_ablation.py         # main deep-learning entry point
|   |-- run_ml_h_baselines.py   # classical ML and spectral-feature baselines
|   |-- oximetry_data.py        # phone-camera SpO2 loader
|   |-- openox_data.py          # OpenOx SaO2/SpO2 loader
|   +-- load_data.py            # private glucose dataset loader
|-- data/                       # local datasets; ignored by Git
|-- outputs/                    # generated runs; ignored by Git
|-- docs/
|   |-- DATASETS.md
|   +-- TRAINING.md
|-- THIRD_PARTY.md
+-- requirements.txt
~~~

Experimental outputs, trained weights, private records, and manuscript files
are intentionally excluded from this public package.

## Installation

Python 3.10 or 3.11 is recommended. Install the PyTorch build appropriate for
your CUDA version, then install the remaining dependencies:

~~~bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## Data

The glucose dataset is **not publicly distributed** because it contains
participant-related information governed by privacy and ethics restrictions.
Qualified researchers may request access from the corresponding author and may
be asked to complete an appropriate data-use agreement.

The other datasets must be obtained from their official sources:

- Phone-camera oximetry: <https://github.com/ubicomplab/oximetry-phone-cam-data>
- OpenOximetry Repository (OpenOx v1.1.1):
  <https://physionet.org/content/openox-repo/1.1.1/>

See [docs/DATASETS.md](docs/DATASETS.md) for the expected local directory
layout. Dataset licenses and access terms remain those of the original owners.

## Quick check

The dry run validates experiment selection and configuration without loading
data or starting training:

~~~bash
python -m src.run_ablation --dataset glucose --experiments R H4R --dry-run
~~~

Full commands for glucose, phone-camera SpO2, OpenOx SaO2, all paper backbones,
and the compact ablation are provided in
[docs/TRAINING.md](docs/TRAINING.md).

## Main experiment groups

| Group | Contents |
|---|---|
| h4a0_paper_backbones | Ten paper baselines and their CRSL variants |
| external_h4_all | The same baseline/CRSL pairs for an external dataset |
| resnet_h4_compact_ablation | ResNet1D baseline, full CRSL, and six ablations |

All generated files are written under outputs/.

## Reproducibility notes

- Use --deterministic-training and an explicit --training-seed for paired runs.
- --n-split 5 runs the five-fold protocol used by the paper scripts.
- Phone-camera windows default to 10 s with a 10 s stride, so adjacent windows
  do not overlap.
- The hardware-response workbook in src/Spectral Distribution Curves.xlsx is
  code-side configuration and contains no participant records.

## Citation

The citation entry will be added after publication. Until then, please cite the
associated manuscript and this repository URL.

## License

No open-source license has been selected yet. Add the intended license before
making the repository public; third-party code and datasets remain subject to
their original licenses.