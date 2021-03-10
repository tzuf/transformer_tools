Transformer Tools
======================

Some utilities for working with transformers.

Basic setup
----------------------------

We suggest using [**conda**](https://docs.conda.io/en/latest/miniconda.html) for creating a python environment, and doing the following:
```bash
conda create -n transformer_tools python=3.6.7
conda activate transformer_tools ## after setting up above
pip install -r requirements.txt
python -m spacy download en  ## (optional for some demos)
```
Below are some of the current use cases. 

Running T5 for QA
----------------------------
One main utility here is the T5 model, to run this and see all of its
options, do the following:
```bash
./run.sh {T5Generator,T5Classifier} --help 
```

The following trains a T5(-large) classifier model on a version of the babi
data:
```bash
./run.sh  T5Classifier \
          --dtype mcqa \
          --output_dir _runs/example \
          --data_dir etc/data/mix_babi \
          --num_train_epochs "12" \
          --model_name_or_path  t5-large \
          --tokenizer_name_or_path t5-large \
          --learning_rate "0.0005" \
          --train_batch_size "16" \
          --seed "42" \
          --max_seq_len "250" \
          --max_answer, "10" \
          --early_stopping \
          --dev_eval \
          --patience "4" \
          --num_beams "2" \
          --print_output \
          --no_repeat_ngram_size "0" \
          --T5_type T5ClassificationMultiQA \
          --data_builder  multi_qa
```
and will place the output into `_runs/example` (to see the final
output, check `_runs/example/dev_eval.tsv`, also to check that
everything looks correct). The final scores will be stored in
`metrics.json`, which gives me the following after running the
experiment above:
```json
{
  "best_dev_score": 0.9195979899497487,
  "dev_eval": 0.9195979899497487
}
```


In the `data_dir` it will expect 2 (optionally 3) files:
`{train,test,dev}.jsonl`, where each line has the following format
(for QA type tasks):
```json
{
    "id": "4297_dev",
    "question" :
        {
            "stem": "Bill got the milk there. Following that he
            discarded the milk there. Mary is not in ..."
        },
    "answerKey": -1,
    "output": "no",
    "prefix": "answer:"
}
```
Where `id` is the example id, `question` and `stem`contain the input
question, `output` contains the target output, and `prefix` pertains
to the target mode (`answerKey` can be safely ignored).

I put an auxliary script in `bin/babi2json.py` to convert babi files
to json. For example,:
```
python bin/babi2json.py --data_loc etc/data/babi_datasets/mix-1-13/ --odir etc/data/babi_datasets/mix-1-13/
```

**A few things to watch out for**: `max_seq_len` and `max_answer`
correspond to the token number of tokens allowed on encoder and
decoder side, respectively. When running experiments, you can see how
many examples are getting truncated (in the data above, I think it
truncates around 16 training examples, which one should be careful
about). In principle, `t5-large` can be replaced with a larger T5
model, e.g., `t5-3b` (which is the largest T5 model that can find on a
single GPU), but so far I've only been able to get this to work with
`gradient_accumulation` (e.g., by adding
`--gradient_accumulation_steps "8"` above) and a `train_batch_size` of 1. 


**Using a trained model directly** From the terminal, you can use the
code from above for testing by setting `--target_model` to the
target model directory and using the `--no_train` directive. Here's
the full call:
```bash
./run.sh  T5Classifier \
          --dtype mcqa \
          --output_dir _runs/example_test \
          --data_dir etc/data/mix_babi \
          --model_name_or_path  t5-large \
          --tokenizer_name_or_path t5-large \
          --max_seq_len "250" \
          --max_answer, "10" \
          --dev_eval \
          --no_training \         # <-------- important!
          --target_model /mode/dir # <-------- important!
          --num_beams "2" \
          --print_output \
          --no_repeat_ngram_size "0" \
          --T5_type T5ClassificationMultiQA \
          --data_builder  multi_qa \
```

Here is how it can be done through the terminal (see also `notebooks/load_t5_babi-model.ipynb`):
```python
>>> from transformer_tools import LoadT5Classifier,get_config
>>> gen_config = get_config("transformer_tools.T5Classifier")
>>> gen_config.target_model = "path/to/output/directory/above"
>>> gen_config.max_answer = 200
>>> model = LoadT5Generator(gen_config)
>>> model.query("target query here..")
```


Polarity Projection Models 
----------------------------
This library also has functionality for sequence tagging using pre-trained
encoders (e.g., BERT, RoBERTa, ..).  The current version relies on the
[**Simple Transformers**](https://www.google.com/search?q=simple+transformers&oq=simple&aqs=chrome.1.69i57j69i59l2j69i60l3j69i65j69i60.2021j0j4&sourceid=chrome&ie=UTF-8)
library.

The code below will train a *BERT-base* polarity tagger.
```bash
./run.sh Tagger \
  --output_dir _runs/example_tagger \
  --data_dir  etc/data/polarity \ ## uses example polarity data here
  --num_train_epochs "3" \
  --learning_rate "0.00001" \
  --train_batch_size "16" \
  --label_list "B-up;B-down;B-=" \ ## set of target labels
  --dev_eval \
  --print_output \ ## will print the model predictions, etc..
  --model_name bert \
  --model_type bert-base-uncased \
  --tagger_model arrow_tagger
```
As above, type `./run.sh Tagger --help` to see the full list of
details. The target files have the same format as above:
```json
{
     "1148_sick.matched.txt_first_arrows"
    "question" :
        {
            "stem": "A man and a woman are hiking through a wooded area"
        },
    "output": "u u u u u u u u u u u" # alternatively, "↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓"
}
```
where the `output` field shows the target arrow annotations (`u/up` =
up; `d/down` = down, `=` = equals. Should also work with symbols `↑`
and `↓`, if you prefer).


Setting up on beaker (AI2 Internal)
---------------------------

Do the following:
```bash
./create_beaker_image.sh
```

An example beaker experiment is included in
`etc/beaker_templates/run_t5_babi_v1.yaml`. To launch, run:
```
beaker experiment create -n "beaker_babi_run" -f etc/beaker_templates/run_t5_babi_v1.yaml
```

Setting up with wandb
---------------------------

Here's an example of running a T5 with wandb on the backend:
```python
python  -m  transformer_tools T5Classifier \
        --output_dir /output \
         --data_dir  /inputs \
         --dev_eval \
         --wdir /output \
         --T5_type T5ClassificationMultiQA \
         --data_builder  multi_qa \
         --wandb_project "t5_model_runs" \ #<---- name of project
         --wandb_name "t5_small_backup_test" \ #<----- name of wandbexp. 
         --wandb_entity "eco-semantics" \ #<--- project id
         --save_wandb_model \ #<--- backup the resulting model on wandb
         --wandb_api_key "xxxxxxxxxxxxxxxxxxxx" #<--- wandb api key (if needed)
```
**NOTE**: `wandb_api_key` is not safe to broadcast like this; if not
running in a cloud environment, it is best to set this as an
environment variable.

Example notebooks, bAbi/polarity demo
---------------------------
Example notebooks are included in `notebooks`.

I hacked out a quick `bAbi` demo interface, specifically for debugging
our newest multi-task models (requires `streamlit`, which can be
installed via `pip`). To run the demo, do:
```
./babi_demo.sh
```
This will require having access to the `eco-semantics` wandb account,
and might require you to set your `WANDB_API_KEY` (via `export WANDB_API_KEY=xxxxxxxxxxxxxx`).

There is also a polarity demo that can be run in the same fashion:
```
./polarity_demo.sh 
```

Notebook for Colab
-----------------------------
This is now in `notebooks/load_polarity_tagger_wandb_colab.ipynb`.
`requirements_colab.txt` is a bit different from `requirements.txt` as I removed a few packages. 

Note: when installing the packages, colab will give several Errors about python package conflict; Just ignore them.


