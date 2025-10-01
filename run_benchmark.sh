#!/bin/bash
QUESTIONS="yesno_questions.json"
ANSWERS="yesno_answers.json"
IMAGES="Images_LR"

# List of models
MODELS=("llava:7b" "llava-llama3:latest" "gemma3:4b" "qwen2.5vl:3b")

for MODEL in "${MODELS[@]}"; do
    # replace ":" and "/" with "_" for safe filenames
    SAFE_MODEL=$(echo $MODEL | tr ':/' '_')
    OUTPUT="benchmark_results_YESNO${SAFE_MODEL}.csv"

    # clear old results
    rm -f $OUTPUT

    # how many questions in the dataset
    TOTAL=$(jq ".questions | length" $QUESTIONS)

    echo "=== Benchmarking $MODEL on $TOTAL yesno questions ==="

    for ((i=0; i<100; i++)); do
        echo ">>> Running question $i / 100"
        python3 llm_accuracy_benchmarking.py \
            --questions $QUESTIONS \
            --answers $ANSWERS \
            --image-dir $IMAGES \
            --model $MODEL \
            --output $OUTPUT \
            --index $i
    done
done

