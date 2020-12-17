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


**Using a trained model directly** This can be done by doing the
following:
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

Example notebooks
---------------------------


An example beaker experiment is included in
`etc/beaker_templates/run_t5_babi_v1.yaml`. To launch, run:
```
beaker experiment create -n "beaker_babi_run" -f etc/beaker_templates/run_t5_babi_v1.yaml
```
