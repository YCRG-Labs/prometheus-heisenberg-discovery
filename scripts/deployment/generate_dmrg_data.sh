#!/bin/bash
# Generate DMRG ground state data with RDM features for L=6 and L=8
# Saves L=6 immediately when done so training can start
#
# Usage: bash scripts/deployment/generate_dmrg_data.sh [--fast]

set -e

echo "=============================================="
echo "J1-J2 DMRG Data Generation (L=6, L=8)"
echo "=============================================="
echo ""

# Configuration
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

# Function to combine temp files into single HDF5
combine_temp_files() {
    local L=$1
    echo "Combining L=$L temp files..."
    python3 << EOF
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
print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")
EOF
}

# Function to run DMRG for one L value
run_dmrg_for_L() {
    local L=$1
    local TEMP_DIR="$OUTPUT_DIR/temp_L${L}"
    mkdir -p "$TEMP_DIR"
    
    echo ""
    echo "=========================================="
    echo "L=$L: DMRG with RDM extraction"
    echo "=========================================="
    
    # Count how many already done
    DONE_COUNT=$(ls "$TEMP_DIR"/J2_*.h5 2>/dev/null | wc -l || echo 0)
    echo "Already completed: $DONE_COUNT / $N_POINTS"
    
    for i in $(seq 0 $((N_POINTS - 1))); do
        J2=$(echo "scale=6; $J2_MIN + $i * ($J2_MAX - $J2_MIN) / ($N_POINTS - 1)" | bc)
        J2_KEY=$(printf "J2_%.3f" $J2)
        TEMP_FILE="$TEMP_DIR/${J2_KEY}.h5"
        
        if [ -f "$TEMP_FILE" ]; then
            continue
        fi
        
        echo ""
        echo "[$((i+1))/$N_POINTS] L=$L, J2=$J2"
        julia scripts/deployment/j1j2_dmrg_rdm.jl $L $J2 $BOND_DIM "$TEMP_FILE"
    done
    
    # Combine into single file
    combine_temp_files $L
    
    echo ""
    echo "L=$L COMPLETE!"
    echo "Output: $OUTPUT_DIR/groundstates_L${L}_rdm.h5"
    echo ""
}

# ============================================
# Run L=6 first, save immediately
# ============================================
run_dmrg_for_L 6

echo "=============================================="
echo "L=6 data ready for training!"
echo "You can now transfer and start training:"
echo "  scp $OUTPUT_DIR/groundstates_L6_rdm.h5 gpu-vm:~/prometheus/results/groundstates/"
echo "=============================================="
echo ""

# ============================================
# Run L=8
# ============================================
run_dmrg_for_L 8

echo ""
echo "=============================================="
echo "All DMRG data generation complete!"
echo "=============================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/groundstates_L*_rdm.h5 2>/dev/null || echo "No files found"
