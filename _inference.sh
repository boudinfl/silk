# script for inference
DOMAIN=$1

BASE_MODEL="bart-base"
REFERENCE="data/${DOMAIN}/test.jsonl"
OUTPUT_DIR="data/${DOMAIN}/outputs/"
NUM_BEAMS=20
NUM_SEQ=20
BATCH_SIZE=4

mkdir -p ${OUTPUT_DIR}

# keyBART model
OUTPUT="${OUTPUT_DIR}/keybart.jsonl"
if [ -f "$OUTPUT" ]; then
    echo "$OUTPUT exists."
else 
    python3 inference-keybart.py --model "models/keybart" \
                          --input ${REFERENCE} \
                          --output $OUTPUT \
                          --num_beams 50 \
                          --num_return_sequences 50 \
                          --batch_size 2
fi

# base model
OUTPUT="${OUTPUT_DIR}/"${BASE_MODEL##*/}".jsonl"
if [ -f "$OUTPUT" ]; then
    echo "$OUTPUT exists."
else 
    python3 inference2.py --model "models/${BASE_MODEL}" \
                          --input ${REFERENCE} \
                          --output $OUTPUT \
                          --num_beams ${NUM_BEAMS} \
                          --num_return_sequences ${NUM_SEQ} \
                          --batch_size ${BATCH_SIZE}
fi

# ft models
for MODEL in models/${BASE_MODEL}-${DOMAIN}*; do
    echo $MODEL
    OUTPUT="${OUTPUT_DIR}/"${MODEL##*/}".jsonl"

    if [ -f "$OUTPUT" ]; then
        echo "$OUTPUT exists."
    else 
        #echo $OUTPUT
        python3 inference2.py --model $MODEL \
                              --input ${REFERENCE} \
                              --output $OUTPUT \
                              --num_beams ${NUM_BEAMS} \
                              --num_return_sequences ${NUM_SEQ} \
                              --batch_size ${BATCH_SIZE}
    fi
done