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

The following trains a T5(-large) classifier model on a version of the babi
data:
```
./run.sh  T5Classifier \
          --dtype mcqa \
          --output_dir _runs/example \
          --data_dir etc/data/mix_babi \
          --num_train_epochs "8" \
          --model_name_or_path  t5-large \
          --tokenizer_name_or_path t5-large \
          --learning_rate "0.0005" \
          --train_batch_size "16" \
          --seed "42" \
          --max_seq_len "250" \
          --max_answer, "5" \
          --early_stopping \
          --dev_eval \
          --patience "4" \
          --num_beams "2" \
          --print_output \
          --no_repeat_ngram_size "0" 
```

Running Transformer Taggers
----------------------------

TODO: add documentation


Setting up on beaker
---------------------------

Do the following:
```
./create_beaker_image.sh
```
