if [ ! $WANDB_API_KEY ]; then
    echo "PLEASE SET WANDB_API_KEY!"
    exit
fi

streamlit run bin/t5_demo.py -- \
          --wandb_model "eco-semantics/t5_model_runs/t5_b_c1k_multi_v3_model:v0" \
          --wandb_entity eco-semantics
