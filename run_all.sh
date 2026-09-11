#!/bin/bash

# Define the python interpreter to use
PYTHON_CMD="/opt/anaconda3/envs/dsde-ice/bin/python3"

MODEL=${1:-all}

case "$MODEL" in
    fivegram)
        echo "Starting Fivegram Model..."
        $PYTHON_CMD fivegram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fivegram.csv
        ;;

    qwen_optimized)
        echo "Starting Optimized Qwen Model..."
        $PYTHON_CMD qwen_optimized.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_qwen_optimized.csv \
            --batch_size 16 \
            --device auto
        ;;

    trigram)
        echo "Starting Trigram Model..."
        $PYTHON_CMD trigram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_trigram.csv \
        ;;

    fourgram)
        echo "Starting Fourgram Model..."
        $PYTHON_CMD fourgram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fourgram.csv \
        ;;

    qwen)
        echo "Starting Qwen Model..."
        $PYTHON_CMD qwen_model.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_qwen.csv \
            --batch_size 4 \
            --device auto
        ;;
    hybrid)
        echo "Starting Hybrid Model..."
        $PYTHON_CMD hybrid_model.py \
            --dev dev_set_final.csv \
            --output predictions_hybrid.csv \
            --limit 1000 \
            --alpha 0.5 \
            --top_k 20
        
        echo "--------------------------------------------------"
        echo "Starting Hybrid Model..."
        $PYTHON_CMD hybrid_model.py \
            --dev dev_set_final.csv \
            --output predictions_hybrid.csv \
            --limit 1000 \
            --alpha 0.8 \
            --top_k 50
        ;;

    smoothed)
        echo "Starting Smoothed Fourgram Model ..."
        $PYTHON_CMD smoothed_fourgram.py \
            --dev test_set_no_answer_final.csv \
            --output predictions_smoothed_test.csv \
            --method absolute_discounting
        ;;

    interpolated)
        echo "Starting Interpolated Fourgram Model ..."
        $PYTHON_CMD smoothed_fourgram.py \
            --dev dev_set_final.csv \
            --output predictions_interpolated.csv \
            --method linear
        ;;

    comp)
        echo "--------------------------------------------------"
        echo "Starting Fourgram Model..."
        $PYTHON_CMD fourgram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fourgram.csv

        echo "--------------------------------------------------"
        echo "Starting Fivegram Model..."
        $PYTHON_CMD fivegram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fivegram.csv

        echo "--------------------------------------------------"
        echo "Starting Smoothed Fourgram Model..."
        $PYTHON_CMD smoothed_fourgram.py \
            --dev dev_set_final.csv \
            --output predictions_smoothed.csv \
            --method absolute_discounting


        echo "Starting Interpolated Fourgram Model ..."
        $PYTHON_CMD smoothed_fourgram.py \
            --dev dev_set_final.csv \
            --output predictions_interpolated.csv \
            --method linear

        ;;

    all)
        echo "Starting Trigram Model..."
        $PYTHON_CMD trigram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_trigram.csv

        echo "--------------------------------------------------"
        echo "Starting Fourgram Model..."
        $PYTHON_CMD fourgram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fourgram.csv

        echo "--------------------------------------------------"
        echo "Starting Fivegram Model..."
        $PYTHON_CMD fivegram.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_fivegram.csv

        echo "--------------------------------------------------"
        echo "Starting Qwen Model..."
        $PYTHON_CMD qwen_model.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_qwen.csv \
            --batch_size 16 \
            --device auto

        echo "--------------------------------------------------"
        echo "Starting Optimized Qwen Model..."
        $PYTHON_CMD qwen_optimized.py \
            --train train_final.src.tok \
            --dev dev_set_final.csv \
            --output predictions_qwen_optimized.csv \
            --batch_size 16 \
            --device auto

        echo "--------------------------------------------------"
        echo "Starting Hybrid Model..."
        $PYTHON_CMD hybrid_model.py \
            --dev dev_set_final.csv \
            --output predictions_hybrid.csv \
            --batch_size 16 \
            --device auto \
            --top_k 100

        echo "--------------------------------------------------"
        echo "Starting Smoothed Fourgram Model..."
        $PYTHON_CMD smoothed_fourgram.py \
            --dev dev_set_final.csv \
            --output predictions_smoothed.csv \
            --method absolute_discounting

        echo "--------------------------------------------------"
        echo "All models finished running."
        ;;

    *)
        echo "Usage: $0 {trigram|fourgram|fivegram|qwen|qwen_optimized|hybrid|smoothed|interpolated|all}"
        exit 1
        ;;
esac