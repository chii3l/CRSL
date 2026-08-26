# Datasets

No participant data are bundled with this repository.

## Private multiband glucose dataset

The glucose cohort is unavailable for unrestricted public download because the
records contain participant-related information governed by privacy and ethics
requirements. Qualified researchers may contact the corresponding author to
request access. Approval and a data-use agreement may be required.

Place the author-provided processed archive in data/glucose/. The loader expects
the following files:

~~~text
BG_Min_Max.npy
data_raw.npy
Y_full.npy
X_full_modal1.npy
X_full_modal2.npy
X_full_modal3.npy
X_full_modal4.npy
X_full_modal5.npy
~~~

The distributed archive defines synchronized PPG, environmental, demographic,
and hardware-related modalities. Do not commit any of these arrays to Git.

## Phone-camera oximetry

Official source:
<https://github.com/ubicomplab/oximetry-phone-cam-data>

Clone or download the official repository into data/phone_camera/. The current
loader expects the original preprocessed HDF5 layout, including:

~~~text
data/phone_camera/data/preprocessed/all_uw_data.h5
~~~

The experiments use the Masimo Radical-7 field (spo2_5) as the SpO2 reference.
Refer to the source repository for collection details, terms, and citation.

## OpenOximetry Repository

Official source:
<https://physionet.org/content/openox-repo/1.1.1/>

Download OpenOx v1.1.1 according to the PhysioNet instructions and place either
the repository root or the 1.1.1 directory under data/openox/. The loader uses
red/infrared PPG and supports blood-gas SaO2, continuous SpO2, or a synchronized
dual-target task.

Dataset access, redistribution, and citation requirements are controlled by
PhysioNet and the dataset authors.