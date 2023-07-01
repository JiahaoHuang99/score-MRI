# --------------------------
# Setting
GPUID=2

# --------------------------
# Run
source activate scoreMRI

rm log_sc_GPU$GPUID

CUDA_VISIBLE_DEVICES=$GPUID nohup python inference_single-coil_on_valset.py >> log_sc_GPU$GPUID &
