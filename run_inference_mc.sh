# --------------------------
# Setting
GPUID=3

# --------------------------
# Run
source activate scoreMRI

rm log_mc_GPU$GPUID

CUDA_VISIBLE_DEVICES=$GPUID nohup python inference_multi-coil_hybrid_on_valset.py >> log_mc_GPU$GPUID &
