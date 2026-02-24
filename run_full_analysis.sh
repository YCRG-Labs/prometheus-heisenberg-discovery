#!/bin/bash
# Run complete J1-J2 analysis

set -e

echo "=========================================="
echo "Starting J1-J2 Heisenberg Analysis"
echo "=========================================="
echo ""
echo "This will run the complete analysis pipeline."
echo "Expected runtime: 4-20 hours depending on hardware"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Activate environment
source venv/bin/activate

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Analysis started at: $(date)" | tee logs/analysis_${TIMESTAMP}.log

# Run pipeline
echo ""
echo "Running main pipeline..."
python main_pipeline.py --config configs/vm_config.yaml 2>&1 | tee -a logs/analysis_${TIMESTAMP}.log

# Create completion marker
touch ANALYSIS_COMPLETE
echo "Analysis completed at: $(date)" | tee -a logs/analysis_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Results are in: ./output/"
echo "Logs are in: ./logs/"
echo ""
