#!/bin/bash

# 1. Modality Argument
MOD=$1
if [[ -z "$MOD" ]]; then echo "Usage: bash run_llm_eval.sh [V|A|AV]"; exit 1; fi

# 2. Paths
CKP_DIR="/data/ssd3/data_rishabh/llama-avsr-ckps"
BASE_OUT_PATH="/data/ssd3/data_rishabh/experiments_autoavsr_wild_multi_llm"
SUMMARY_FILE="${BASE_OUT_PATH}/summary_llm_${MOD}.csv"

# 3. Specific Flags for each Modality
LLM_MODEL="meta-llama/Meta-Llama-3.1-8B"
AVHUBERT_PATH="/data/ssd2/data_rishabh/avhubert_prep/ckps/large_vox_iter5.pt"

case $MOD in
    "A")
        M_NAME="audio"
        CKP="${CKP_DIR}/ASR_Whisper-M_Llama3.1-8B_lrs3vox_down3_seed42.pth"
        EXTRA_FLAGS="--audio-encoder-name openai/whisper-medium.en \
                     --unfrozen_modules peft_llm \
                     --downsample-ratio-audio 3"
        ;;
    "V")
        M_NAME="video"
        CKP="${CKP_DIR}/VSR_AVH-L_Llama3.1-8B_lrs3vox_down3_seed42.pth"
        EXTRA_FLAGS="--pretrain-avhubert-enc-video-path $AVHUBERT_PATH \
                     --use-lora-avhubert True \
                     --unfrozen_modules peft_llm lora_avhubert \
                     --downsample-ratio-video 3"
        ;;
    "AV")
        M_NAME="audiovisual"
        CKP="${CKP_DIR}/AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth"
        EXTRA_FLAGS="--audio-encoder-name openai/whisper-medium.en \
                     --pretrain-avhubert-enc-video-path $AVHUBERT_PATH \
                     --unfrozen_modules peft_llm \
                     --downsample-ratio-audio 4 \
                     --downsample-ratio-video 2"
        ;;
esac

# 4. Dataset List (Including LRS2 and LRS3)
DATASETS=(
    "LRS2|/data/ssd2/data_rishabh/lrs2_rf/labels/"
    "LRS3|/data/ssd2/data_rishabh/lrs3/metadata/"
    "LombardGrid_front|/data/ssd3/data_rishabh/LombardGrid_Clean/meta/front/"
    "LombardGrid_side|/data/ssd3/data_rishabh/LombardGrid_Clean/meta/side/"
    "LombardGrid_Combined|/data/ssd3/data_rishabh/LombardGrid_Clean/meta/combined/"
    "GRID|/data/ssd3/data_rishabh/Grid_clean/meta"
    "RoomReader_conversational|/data/ssd3/data_rishabh/RoomReader_lips/meta/conversational"
    "RoomReader_individual|/data/ssd3/data_rishabh/RoomReader_lips/meta/individual"
    "RoomReader_combined|/data/ssd3/data_rishabh/RoomReader_lips/meta/combined"
    "TCD_TIMIT_lipspeakers_30degcam|/data/ssd3/data_rishabh/tcd_timit/meta/lipspeakers_30degcam"
    "TCD_TIMIT_lipspeakers_straightcam|/data/ssd3/data_rishabh/tcd_timit/meta/lipspeakers_straightcam"
    "TCD_TIMIT_lipspeakers|/data/ssd3/data_rishabh/tcd_timit/meta/lipspeakers"
    "TCD_TIMIT_volunteers_30degcam|/data/ssd3/data_rishabh/tcd_timit/meta/volunteers_30degcam"
    "TCD_TIMIT_volunteers_straightcam|/data/ssd3/data_rishabh/tcd_timit/meta/volunteers_straightcam"
    "TCD_TIMIT_volunteers|/data/ssd3/data_rishabh/tcd_timit/meta/volunteers"
    "TCD_TIMIT_combined|/data/ssd3/data_rishabh/tcd_timit/meta/combined"
)

# Initialize CSV
mkdir -p "$BASE_OUT_PATH"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Dataset,Modality,WER,CER" > "$SUMMARY_FILE"
fi

export PYTHONPATH=$PYTHONPATH:.

for ENTRY in "${DATASETS[@]}"; do
    NAME="${ENTRY%%|*}"
    META="${ENTRY##*|}"
    
    # 1. Search for 'test' CSV
    TEST_FILE=$(ls ${META}/*test*transcript_lengths_seg16s*.csv 2>/dev/null | head -n 1)
    
    # 2. Clean ROOT_DIR logic (strips /meta, /labels, or /metadata)
    ROOT_DIR="${META%/meta*}"
    ROOT_DIR="${ROOT_DIR%/labels*}"
    ROOT_DIR="${ROOT_DIR%/metadata*}"

    if [ -z "$TEST_FILE" ]; then continue; fi

    OUT_DIR="${BASE_OUT_PATH}/${NAME}/${MOD}"
    mkdir -p "$OUT_DIR"
    LOG="${OUT_DIR}/eval.log"

    echo "------------------------------------------------"
    echo "RUNNING: $NAME ($MOD)"
    echo "ROOT:    $ROOT_DIR"
    echo "CSV:     $(basename $TEST_FILE)"
    echo "------------------------------------------------"

    python eval.py \
        --exp-name "AVSR_inference" \
        --modality "$M_NAME" \
        --pretrained-model-path "$CKP" \
        --root-dir "$ROOT_DIR" \
        --test-file "$TEST_FILE" \
        --llm-model "$LLM_MODEL" \
        --add_PETF_LLM lora \
        --reduction_lora 64 \
        --alpha 8 \
        --max-dec-tokens 32 \
        --num-beams 15 \
        $EXTRA_FLAGS 2>&1 | tee "$LOG"

    # Extract Results
    WER_RAW=$(grep -oP "'test_wer': \K[0-9.]+" "$LOG" | head -1)
    [ -n "$WER_RAW" ] && WER=$(awk -v v="$WER_RAW" 'BEGIN {printf "%.2f", v*100}') || WER="N/A"

    CER_RAW=$(grep -oP "'test_cer': \K[0-9.]+" "$LOG" | head -1)
    [ -n "$CER_RAW" ] && CER=$(awk -v v="$CER_RAW" 'BEGIN {printf "%.2f", v*100}') || CER="N/A"

    echo "${NAME},${MOD},${WER},${CER}" >> "$SUMMARY_FILE"
done