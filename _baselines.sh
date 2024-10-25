for DOMAIN in "kp20k" "nlp" "astro" "paleo"; do
    OUTPUT_DIR="data/${DOMAIN}/outputs"
    mkdir -p ${OUTPUT_DIR}

    INPUT="data/${DOMAIN}/test.jsonl"

    # MPRank
    OUTPUT="${OUTPUT_DIR}/test.mprank.jsonl"
    if [ -f "$OUTPUT" ]; then
        echo "$OUTPUT exists."
    else 
        python3 baselines/mprank.py -i $INPUT -o $OUTPUT
    fi

    # YAKE
    OUTPUT="${OUTPUT_DIR}/test.yake.jsonl"
    if [ -f "$OUTPUT" ]; then
        echo "$OUTPUT exists."
    else 
        python3 baselines/yake-baseline.py -i $INPUT -o $OUTPUT
    fi

done