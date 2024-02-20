#!/bin/bash
source /scratch_net/biwidl210/kcekmeceli/conda/etc/profile.d/conda.sh
conda activate dinov2
python -u dinov2/MedDino/train_seg_head_lit.py "$@"
