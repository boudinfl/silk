# Script for fine-tuning models
DOMAIN=$1

PATH_SILVER="data/${DOMAIN}/train.jsonl"

#BASE_MODEL="bart-base"
#BATCH_SIZE=16
#GRADIENT_ACCUMULATION_STEPS=1

EPOCHS=3
LR='1e-6'

PATH_SELF_LEARNING="data/${DOMAIN}/train.${BASE_MODEL}.jsonl"
CUT_OFF=2000

if [ -f "$PATH_SELF_LEARNING" ]; then
   echo "$PATH_SELF_LEARNING exists."
else
    python3 self-supervised.py --model "models/${BASE_MODEL}" \
                               --input ${PATH_SILVER} \
                               --output ${PATH_SELF_LEARNING} \
                               --batch_size 4 \
                               --cut_off ${CUT_OFF}
fi

for FEW_SHOT in 500 1000 2000; do
    OUTPUT="models/${BASE_MODEL}-${DOMAIN}-${FEW_SHOT}-${EPOCHS}epochs.lr${LR}/"
    if [ -d "$OUTPUT" ]; then
        echo "${OUTPUT} already exists"
    else
        python3 fine-tuning.py --input ${PATH_SILVER} \
                               --output ${OUTPUT} \
                               --batch_size ${BATCH_SIZE} \
                               --num_train_epochs ${EPOCHS} \
                               --few_shot ${FEW_SHOT} \
                               --model "models/${BASE_MODEL}" \
                               --learning_rate ${LR} \
                               --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    fi

    OUTPUT="models/${BASE_MODEL}-${DOMAIN}-self-${FEW_SHOT}-${EPOCHS}epochs.lr${LR}/"
    if [ -d "$OUTPUT" ]; then
        echo "${OUTPUT} already exists"
    else
        python3 fine-tuning.py --input ${PATH_SELF_LEARNING} \
                               --output ${OUTPUT} \
                               --batch_size ${BATCH_SIZE} \
                               --num_train_epochs ${EPOCHS} \
                               --few_shot ${FEW_SHOT} \
                               --model "models/${BASE_MODEL}" \
                               --learning_rate ${LR} \
                               --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    fi
done
                       


