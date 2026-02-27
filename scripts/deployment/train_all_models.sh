#!/bin/bash
# Train all VAE models for J1-J2 analysis
# Automatically detects available data and trains accordingly
#
# Usage: bash scripts/deployment/train_all_models.sh

set -e

echo "=============================================="
echo "J1-J2 VAE Training Pipeline"
echo "=============================================="
echo ""

# Configuration
DATA_DIR="results/groundstates"
OUTPUT_DIR="results/trained_models"
CONFIG="configs/laptop_config.yaml"

mkdir -p "$OUTPUT_DIR"

# Activate Python venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for GPU
echo "Checking hardware..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "PyTorch check failed"
echo ""

# ============================================
# Detect available data and train
# ============================================

echo "Scanning for available data..."
echo ""

# L=4: Full wavefunction VAE
if [ -f "$DATA_DIR/groundstates_L4.h5" ]; then
    echo "=========================================="
    echo "Training VAE for L=4 (full wavefunction)"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/L4/latent_representations.h5" ]; then
        echo "L=4 already trained, skipping (delete $OUTPUT_DIR/L4 to retrain)"
    else
        python scripts/deployment/train_vae.py \
            --input "$DATA_DIR/groundstates_L4.h5" \
            --config "$CONFIG" \
            --output_dir "$OUTPUT_DIR/L4"
        echo "L=4 training complete!"
    fi
    echo ""
else
    echo "L=4 data not found, skipping"
fi

# L=5: Full wavefunction VAE
if [ -f "$DATA_DIR/groundstates_L5.h5" ]; then
    echo "=========================================="
    echo "Training VAE for L=5 (full wavefunction)"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/L5/latent_representations.h5" ]; then
        echo "L=5 already trained, skipping"
    else
        python scripts/deployment/train_vae.py \
            --input "$DATA_DIR/groundstates_L5.h5" \
            --config "$CONFIG" \
            --output_dir "$OUTPUT_DIR/L5"
        echo "L=5 training complete!"
    fi
    echo ""
else
    echo "L=5 data not found, skipping"
fi

# L=6: RDM-based VAE
if [ -f "$DATA_DIR/groundstates_L6_rdm.h5" ]; then
    echo "=========================================="
    echo "Training VAE for L=6 (RDM features)"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/L6/latent_representations.h5" ]; then
        echo "L=6 already trained, skipping"
    else
        python scripts/deployment/train_vae_rdm.py \
            --input "$DATA_DIR/groundstates_L6_rdm.h5" \
            --config "$CONFIG" \
            --output_dir "$OUTPUT_DIR/L6"
        echo "L=6 training complete!"
    fi
    echo ""
else
    echo "L=6 data not found, skipping"
fi

# L=8: RDM-based VAE
if [ -f "$DATA_DIR/groundstates_L8_rdm.h5" ]; then
    echo "=========================================="
    echo "Training VAE for L=8 (RDM features)"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/L8/latent_representations.h5" ]; then
        echo "L=8 already trained, skipping"
    else
        python scripts/deployment/train_vae_rdm.py \
            --input "$DATA_DIR/groundstates_L8_rdm.h5" \
            --config "$CONFIG" \
            --output_dir "$OUTPUT_DIR/L8"
        echo "L=8 training complete!"
    fi
    echo ""
else
    echo "L=8 data not found, skipping"
fi

# ============================================
# Run analysis on whatever is available
# ============================================
echo "=========================================="
echo "Running phase analysis..."
echo "=========================================="

# Check if any models exist
if ls "$OUTPUT_DIR"/L*/latent_representations.h5 1> /dev/null 2>&1; then
    python scripts/deployment/analyze_phases.py \
        --model_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --output_dir "results/analysis"
else
    echo "No trained models found, skipping analysis"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo ""
echo "Trained models:"
ls -la "$OUTPUT_DIR"/L*/latent_representations.h5 2>/dev/null || echo "  None"
echo ""
echo "Analysis results:"
ls -la results/analysis/*.json 2>/dev/null || echo "  None"
