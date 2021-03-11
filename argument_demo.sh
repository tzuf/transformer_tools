if [ ! $WANDB_API_KEY ]; then
    echo "PLEASE SET WANDB_API_KEY!"
    exit
fi

streamlit run bin/argument_demo.py -- \
          --wandb_model "aaac/aaac_model_runs/multi_angle_exp_model:v0" \
          --wandb_entity aaac
