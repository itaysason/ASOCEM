## ASOCEM

### Installation

```sh
conda env create -f environment.yml
source activate ASOCEM
```

### Training the model

```sh
python ASOCEM.py ASOCEM --in_dir _ --out_dir _ --particle_size _ --downsample_size _ --window_size _ --contamination_criterion size/power --algorithm regular/fast --n_cores _
```
