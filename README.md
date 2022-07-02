# Code Style Transfer & Probing
UCSC IBM Capstone dedicated to probing large language models for code style.


## Usage
### Extract Metrics from Py150k
```bash
python extract_metrics.py py150k
```

### Training
> Modify the setup in `config.py` before starting the training
```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python train.py
```