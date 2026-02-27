#!/bin/bash
# Generate all ground state data for J1-J2 analysis
# Run this on the CPU VM with Julia + ITensors installed
#
# Usage: bash scripts/deployment/generate_all_data.sh [--fast]
#   --fast: Use fewer points and lower bond dim for quick test

set -e

echo "=============================================="
echo "J1-J2 Heisenberg Ground State Data Generation"
echo "=============================================="
echo ""

# Configuration - can override with --fast flag
if [[ "$1" == "--fast" ]]; then
    N_POINTS=21
    BOND_DIM=100
    echo "FAST MODE: $N_POINTS points, bond_dim=$BOND_DIM"
else
    N_POINTS=41
    BOND_DIM=200
    echo "FULL MODE: $N_POINTS points, bond_dim=$BOND_DIM"
fi

OUTPUT_DIR="results/groundstates"
J2_MIN=0.0
J2_MAX=1.0

mkdir -p "$OUTPUT_DIR"

# Activate Python venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Function to combine temp files into single HDF5
combine_temp_files() {
    local L=$1
    python << EOF
import h5py
from pathlib import Path

L = $L
output_dir = Path("results/groundstates")
temp_dir = output_dir / f"temp_L{L}"
output_file = output_dir / f"groundstates_L{L}_rdm.h5"

if not temp_dir.exists():
    print(f"No temp dir for L={L}")
    exit(0)

temp_files = sorted(temp_dir.glob("J2_*.h5"))
if not temp_files:
    print(f"No temp files for L={L}")
    exit(0)

print(f"Combining {len(temp_files)} files for L={L}...")

with h5py.File(output_file, "w") as out_f:
    out_f.attrs["L"] = L
    out_f.attrs["n_points"] = len(temp_files)
    
    for temp_file in temp_files:
        j2_key = temp_file.stem
        with h5py.File(temp_file, "r") as in_f:
            grp = out_f.create_group(j2_key)
            for key in in_f.keys():
                grp.create_dataset(key, data=in_f[key][()])
            for attr_key in in_f.attrs:
                grp.attrs[attr_key] = in_f.attrs[attr_key]

print(f"Saved {output_file}")
EOF
}

# ============================================
# L=4: Exact diagonalization with QuSpin
# ============================================
echo "=========================================="
echo "L=4: Exact Diagonalization (QuSpin)"
echo "=========================================="

if [ ! -f "$OUTPUT_DIR/groundstates_L4.h5" ]; then
    python scripts/deployment/generate_groundstates.py \
        --L 4 \
        --use_quspin \
        --n_points $N_POINTS \
        --output "$OUTPUT_DIR/groundstates_L4.h5"
    echo "L=4 complete!"
else
    echo "L=4 already exists, skipping"
fi
echo ""

# ============================================
# L=5: Exact diagonalization with QuSpin  
# ============================================
echo "=========================================="
echo "L=5: Exact Diagonalization (QuSpin)"
echo "=========================================="

if [ ! -f "$OUTPUT_DIR/groundstates_L5.h5" ]; then
    python scripts/deployment/generate_groundstates.py \
        --L 5 \
        --use_quspin \
        --n_points $N_POINTS \
        --output "$OUTPUT_DIR/groundstates_L5.h5"
    echo "L=5 complete!"
else
    echo "L=5 already exists, skipping"
fi
echo ""

# ============================================
# L=6: DMRG with RDM extraction (parallel)
# ============================================
echo "=========================================="
echo "L=6: DMRG with RDM extraction"
echo "=========================================="

L=6
TEMP_DIR="$OUTPUT_DIR/temp_L${L}"
mkdir -p "$TEMP_DIR"

# Use GNU parallel if available, otherwise sequential
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for speedup..."
    
    # Generate J2 values file
    > /tmp/j2_values_L${L}.txt
    for i in $(seq 0 $((N_POINTS - 1))); do
        J2=$(echo "scale=6; $J2_MIN + $i * ($J2_MAX - $J2_MIN) / ($N_POINTS - 1)" | bc)
        J2_KEY=$(printf "J2_%.3f" $J2)
        TEMP_FILE="$TEMP_DIR/${J2_KEY}.h5"
        if [ ! -f "$TEMP_FILE" ]; then
            echo "$J2 $TEMP_FILE" >> /tmp/j2_values_L${L}.txt
        fi
    done
    
    N_TODO=$(wc -l < /tmp/j2_values_L${L}.txt)
    if [ "$N_TODO" -gt 0 ]; then
        echo "Computing $N_TODO points for L=$L..."
        cat /tmp/j2_values_L${L}.txt | parallel --colsep ' ' -j 4 \
            "julia scripts/deployment/j1j2_dmrg_rdm.jl $L {1} $BOND_DIM {2}"
    else
        echo "All L=$L points already computed"
    fi
else
    # Sequential fallback
    for i in $(seq 0 $((N_POINTS - 1))); do
        J2=$(echo "scale=6; $J2_MIN + $i * ($J2_MAX - $J2_MIN) / ($N_POINTS - 1)" | bc)
        J2_KEY=$(printf "J2_%.3f" $J2)
        TEMP_FILE="$TEMP_DIR/${J2_KEY}.h5"
        
        if [ -f "$TEMP_FILE" ]; then
            echo "Skipping L=$L, J2=$J2 (already done)"
            continue
        fi
        
        echo "Computing L=$L, J2=$J2 ($((i+1))/$N_POINTS)..."
        julia scripts/deployment/j1j2_dmrg_rdm.jl $L $J2 $BOND_DIM "$TEMP_FILE"
    done
fi

combine_temp_files 6
echo "L=6 complete!"
echo ""

# ============================================
# L=8: DMRG with RDM extraction (parallel)
# ============================================
echo "=========================================="
echo "L=8: DMRG with RDM extraction"
echo "=========================================="

L=8
TEMP_DIR="$OUTPUT_DIR/temp_L${L}"
mkdir -p "$TEMP_DIR"

if command -v parallel &> /dev/null; then
    > /tmp/j2_values_L${L}.txt
    for i in $(seq 0 $((N_POINTS - 1))); do
        J2=$(echo "scale=6; $J2_MIN + $i * ($J2_MAX - $J2_MIN) / ($N_POINTS - 1)" | bc)
        J2_KEY=$(printf "J2_%.3f" $J2)
        TEMP_FILE="$TEMP_DIR/${J2_KEY}.h5"
        if [ ! -f "$TEMP_FILE" ]; then
            echo "$J2 $TEMP_FILE" >> /tmp/j2_values_L${L}.txt
        fi
    done
    
    N_TODO=$(wc -l < /tmp/j2_values_L${L}.txt)
    if [ "$N_TODO" -gt 0 ]; then
        echo "Computing $N_TODO points for L=$L..."
        cat /tmp/j2_values_L${L}.txt | parallel --colsep ' ' -j 2 \
            "julia scripts/deployment/j1j2_dmrg_rdm.jl $L {1} $BOND_DIM {2}"
    else
        echo "All L=$L points already computed"
    fi
else
    for i in $(seq 0 $((N_POINTS - 1))); do
        J2=$(echo "scale=6; $J2_MIN + $i * ($J2_MAX - $J2_MIN) / ($N_POINTS - 1)" | bc)
        J2_KEY=$(printf "J2_%.3f" $J2)
        TEMP_FILE="$TEMP_DIR/${J2_KEY}.h5"
        
        if [ -f "$TEMP_FILE" ]; then
            echo "Skipping L=$L, J2=$J2 (already done)"
            continue
        fi
        
        echo "Computing L=$L, J2=$J2 ($((i+1))/$N_POINTS)..."
        julia scripts/deployment/j1j2_dmrg_rdm.jl $L $J2 $BOND_DIM "$TEMP_FILE"
    done
fi

combine_temp_files 8
echo "L=8 complete!"
echo ""

echo "=============================================="
echo "Data generation complete!"
echo "=============================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.h5 2>/dev/null || echo "No HDF5 files found"
echo ""
echo "Transfer to GPU machine:"
echo "  scp $OUTPUT_DIR/groundstates_L*.h5 gpu-vm:~/prometheus/results/groundstates/"
