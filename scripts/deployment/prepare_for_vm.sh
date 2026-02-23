#!/bin/bash
# Prepare repository for VM deployment
# This script creates a clean package ready for VM upload

set -e

echo "=========================================="
echo "Preparing Repository for VM Deployment"
echo "=========================================="
echo ""

# Create deployment package directory
PACKAGE_DIR="j1j2_vm_package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="j1j2_heisenberg_${TIMESTAMP}.tar.gz"

echo "Creating package directory..."
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy essential files
echo "Copying source files..."
cp -r src $PACKAGE_DIR/
cp -r tests $PACKAGE_DIR/
cp -r configs $PACKAGE_DIR/
cp -r notebooks $PACKAGE_DIR/
cp -r paper $PACKAGE_DIR/

# Copy scripts
echo "Copying scripts..."
cp main_pipeline.py $PACKAGE_DIR/
cp run_vm_analysis.py $PACKAGE_DIR/
cp run_sequential_analysis.py $PACKAGE_DIR/
cp run_ed_only.py $PACKAGE_DIR/
cp validate_setup.py $PACKAGE_DIR/
cp document_hardware.py $PACKAGE_DIR/

# Copy VM setup files
cp vm_setup.sh $PACKAGE_DIR/
cp VM_DEPLOYMENT_GUIDE.md $PACKAGE_DIR/
cp VM_QUICK_START.md $PACKAGE_DIR/

# Copy documentation
echo "Copying documentation..."
cp README.md $PACKAGE_DIR/
cp REPRODUCIBILITY.md $PACKAGE_DIR/
cp REPRODUCIBILITY_CHECKLIST.md $PACKAGE_DIR/
cp PIPELINE_USAGE.md $PACKAGE_DIR/
cp requirements.txt $PACKAGE_DIR/
cp LICENSE $PACKAGE_DIR/

# Copy configuration files
cp pytest.ini $PACKAGE_DIR/
cp .gitignore $PACKAGE_DIR/

# Create empty directories
echo "Creating directory structure..."
mkdir -p $PACKAGE_DIR/data
mkdir -p $PACKAGE_DIR/output
mkdir -p $PACKAGE_DIR/logs
mkdir -p $PACKAGE_DIR/checkpoints/ed_checkpoints
mkdir -p $PACKAGE_DIR/checkpoints/qvae_models

# Create .gitkeep files
touch $PACKAGE_DIR/data/.gitkeep
touch $PACKAGE_DIR/output/.gitkeep
touch $PACKAGE_DIR/logs/.gitkeep

# Copy existing data if available
if [ -f "data/j1j2_data.h5" ]; then
    echo "Copying existing data..."
    cp data/j1j2_data.h5 $PACKAGE_DIR/data/
fi

# Copy existing checkpoints if available
if [ -d "checkpoints/ed_checkpoints" ] && [ "$(ls -A checkpoints/ed_checkpoints)" ]; then
    echo "Copying ED checkpoints..."
    cp -r checkpoints/ed_checkpoints/* $PACKAGE_DIR/checkpoints/ed_checkpoints/
fi

# Create README for package
cat > $PACKAGE_DIR/README_VM.txt << 'EOF'
J1-J2 Heisenberg Prometheus Framework - VM Package
===================================================

This package contains everything needed to run the complete analysis on a VM.

QUICK START:
1. Extract this archive on your VM
2. Run: chmod +x vm_setup.sh && ./vm_setup.sh
3. Run: tmux new -s analysis
4. Run: ./run_full_analysis.sh

DOCUMENTATION:
- VM_QUICK_START.md - Fast track guide (start here!)
- VM_DEPLOYMENT_GUIDE.md - Comprehensive deployment guide
- README.md - Project overview
- REPRODUCIBILITY.md - Reproducibility guidelines

REQUIREMENTS:
- Ubuntu 22.04 or 20.04 LTS
- 16+ CPU cores (32+ recommended)
- 32+ GB RAM (64+ GB recommended)
- 100 GB storage
- Optional: NVIDIA GPU with 8+ GB VRAM

ESTIMATED RUNTIME:
- With 16 cores + GPU: 6-9 hours
- With 32 cores + GPU: 4-6 hours
- CPU only (16 cores): 9-13 hours

For questions or issues, see VM_DEPLOYMENT_GUIDE.md
EOF

# Create archive
echo ""
echo "Creating archive..."
tar -czf $ARCHIVE_NAME $PACKAGE_DIR

# Calculate size
SIZE=$(du -h $ARCHIVE_NAME | cut -f1)

# Cleanup
rm -rf $PACKAGE_DIR

echo ""
echo "=========================================="
echo "Package Created Successfully!"
echo "=========================================="
echo ""
echo "Archive: $ARCHIVE_NAME"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "  1. Upload to VM:"
echo "     scp $ARCHIVE_NAME user@vm-ip:~/"
echo ""
echo "  2. On VM, extract:"
echo "     tar -xzf $ARCHIVE_NAME"
echo "     cd $PACKAGE_DIR"
echo ""
echo "  3. Run setup:"
echo "     chmod +x vm_setup.sh"
echo "     ./vm_setup.sh"
echo ""
echo "  4. Start analysis:"
echo "     tmux new -s analysis"
echo "     ./run_full_analysis.sh"
echo ""
echo "See VM_QUICK_START.md for detailed instructions"
echo ""
