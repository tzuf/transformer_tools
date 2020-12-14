Transformer Tools
======================

Some utilities for working with transformers.

Basic setup
----------------------------

We suggest using [**conda**](https://docs.conda.io/en/latest/miniconda.html) for creating a python environment, and doing the following:
```
conda create -n transformer_tools python=3.6.7
conda activate transformer_tools ## after setting up above
pip install -r requirements.txt
```

Running T5
----------------------------
One main utility here is the T5 model, to run this and see all of its
options, do the following:
```
./run.sh {T5Generator,T5Classifier} --help 
```

Setting up on beaker
---------------------------

Do the following:
```
./create_beaker_image.sh
```
