CHECKPOINT_NAME="Llama-3.2-1B-Instruct-s2t"  
CHECKPOINT_DIR="./checkpoints/${CHECKPOINT_NAME}" 
BASE_MODEL="./hf_hub/Llama-3.2-1B-Instruct"
DATA_PATH="./playground/data.json"
SPEECH_FOLDER="./playground/"
SPEECH_ENCODER="./models/speech_encoder/whisper-large-v3"

CUDA_VISIBLE_DEVICES=0 python ./omni_speech/merge_lora_model.py --model-path $CHECKPOINT_DIR/lora_checkpoints --model-base $BASE_MODEL --save-model-path $CHECKPOINT_DIR