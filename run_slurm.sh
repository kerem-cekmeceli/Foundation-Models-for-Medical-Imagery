#!/bin/bash
source /scratch_net/biwidl210/kcekmeceli/conda/etc/profile.d/conda.sh
conda activate dinov2
python -u /scratch_net/biwidl210/kcekmeceli/FoundationModels/MedicalSegmentation/train.py "$@"
