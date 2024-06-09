```sh
conda create -n harvard_gf python=3.10  pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate harvard_gf
conda install scikit-learn fairlearn scikit-image pandas -c conda-forge
pip install blobfile
```

Data is stored one level higher relative to this repository, in `quant_notes/data_compr`. Names of subfolders must be changed to `train` and `test` respectively. Validation is currently unused.

Next, run `scripts/train_glaucoma_fair_fin.sh`. This should train the 2D model for 10 epochs. Results can be found in results folder, currently does not save model weights.