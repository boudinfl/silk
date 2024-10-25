DOMAIN=$1
REFERENCE="data/${DOMAIN}/test.jsonl"

echo "Evaluation for [${DOMAIN}]"
echo "|-> using reference file: ${REFERENCE}"

for MODEL in data/${DOMAIN}/outputs/*.jsonl; do
    python3 evaluate2.py -r ${REFERENCE} -s ${MODEL} --output_scores
done