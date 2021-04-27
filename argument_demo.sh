#MODEL_ID="aaac/aaac_model_runs/multi_angle_exp_model:v0"
MODEL_ID="aaac/finetune/model-3bfu0wcn:v0"


if [ ! $WANDB_API_KEY ]; then
    echo "PLEASE SET WANDB_API_KEY!"
    exit
fi

streamlit run bin/argument_demo.py --server.port 9050 -- \
          --wandb_model $MODEL_ID \
          --wandb_entity aaac
