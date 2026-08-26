# Training

Run commands from the repository root. Replace --gpu 0 as needed; use --gpu -1
to expose CPU-only execution.

## Configuration check

~~~bash
python -m src.run_ablation --dataset glucose --experiments h4a0_paper_backbones --dry-run
~~~

## Private glucose dataset

All ten paper baselines and paired CRSL variants, five folds:

~~~bash
python -m src.run_ablation --dataset glucose --experiments h4a0_paper_backbones --gpu 0 --epochs 500 --n-split 5 --batch-size 256 --modal-mask 11100 --ppg-mask 111111 --glucose-split record --glucose-test-size 0.2 --glucose-random-state 2 --glucose-aux-spo2 --data-path data/glucose --load-cache --deterministic-training --training-seed 2026
~~~

Compact ResNet1D ablation:

~~~bash
python -m src.run_ablation --dataset glucose --experiments resnet_h4_compact_ablation --gpu 0 --epochs 500 --n-split 5 --batch-size 256 --modal-mask 11100 --ppg-mask 111111 --glucose-split record --glucose-aux-spo2 --data-path data/glucose --load-cache --deterministic-training --training-seed 2026
~~~

## Phone-camera SpO2

This setup uses non-overlapping 10 s windows and the Masimo Radical-7 reference:

~~~bash
python -m src.run_ablation --dataset oximetry --experiments external_h4_all --gpu 0 --epochs 500 --n-split 5 --batch-size 64 --learning-rate 0.001 --modal-mask 10000 --ppg-mask 111000 --oximetry-root data/phone_camera --oximetry-split random --oximetry-random-test-size 0.2 --oximetry-random-state 2 --oximetry-window-sec 10 --oximetry-stride-sec 10 --oximetry-target spo2_5 --oximetry-normalize window_minmax --deterministic-training --training-seed 2026
~~~

## OpenOx SaO2

~~~bash
python -m src.run_ablation --dataset openox_sao2 --experiments external_h4_all --gpu 0 --epochs 500 --n-split 5 --batch-size 256 --learning-rate 0.001 --modal-mask 10000 --ppg-mask 110000 --openox-root data/openox --openox-split random --openox-test-size 0.2 --openox-random-state 2 --openox-window-sec 6 --openox-sampling-rate 50 --openox-output-len 300 --openox-normalize window_minmax --openox-detrend --openox-detrend-cutoff 0.3 --openox-scale none --load-cache --deterministic-training --training-seed 2026
~~~

Use --dataset openox_spo2 for continuous SpO2 or --dataset
openox_sao2_spo2 for synchronized dual-target output.

## Machine-learning baselines

~~~bash
python -m src.run_ml_h_baselines --datasets glucose --ml-models rf extratrees gbr linear_svr --feature-sets raw raw_h --n-split 5 --glucose-spo2-target --data-path data/glucose --load-cache
~~~

## Outputs

Each deep-learning run is saved immediately under outputs/YYYY-MM-DD/.
Manifests are also written to outputs/, allowing interrupted experiment groups
to be audited without bundling results into the repository.