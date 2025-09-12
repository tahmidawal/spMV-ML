#!/bin/bash

# Local training script for GCN model
echo "Starting GCN training locally..."
echo "================================"
echo "Configuration:"
echo "  - Model: GCN (Graph Convolutional Network)"
echo "  - Hidden Dim: 128"
echo "  - Layers: 4"
echo "  - Epochs: 200"
echo "  - Batch Size: 8"
echo "  - Learning Rate: 0.001"
echo "  - Dropout: 0.1"
echo "================================"

# Run the training
python 0911_train_gnn_experiment.py \
    --conv-type GCN \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --hidden-dim 128 \
    --num-layers 4 \
    --dropout 0.1 \
    --weight-decay 1e-4

echo "Training completed!"
