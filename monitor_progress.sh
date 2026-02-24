#!/bin/bash
# Monitor analysis progress

echo "=========================================="
echo "J1-J2 Analysis Progress Monitor"
echo "=========================================="
echo ""

# Check if analysis is running
if pgrep -f "main_pipeline.py" > /dev/null; then
    echo "✓ Analysis is RUNNING"
else
    echo "✗ Analysis is NOT running"
fi

echo ""
echo "Latest log entries:"
echo "-------------------"
tail -n 20 logs/j1j2_prometheus.log

echo ""
echo "Checkpoints:"
echo "------------"
ls -lh checkpoints/ed_checkpoints/ 2>/dev/null || echo "No ED checkpoints yet"
ls -lh checkpoints/qvae_models/ 2>/dev/null || echo "No Q-VAE models yet"

echo ""
echo "Output files:"
echo "-------------"
ls -lh output/ 2>/dev/null || echo "No output files yet"

echo ""
echo "System resources:"
echo "-----------------"
echo "CPU usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk: $(df -h . | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{print "  GPU: " $1 "%, Memory: " $2 "/" $3 " MB"}'
fi
