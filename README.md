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
```
{
  "best_dev_score": 0.914572864321608,
  "dev_eval": 0.914572864321608
}
```


In the `data_dir` it will expect 2 (optionally 3) files:
`{train,test,dev}.jsonl`, where each line has the following format
(for QA type tasks):
```
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

Running Transformer Taggers
----------------------------

TODO: add documentation


Setting up on beaker (AI2 Internal)
---------------------------

Do the following:
```
./create_beaker_image.sh
```
