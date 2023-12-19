#!/bin/bash
source /scratch_net/biwidl210/kcekmeceli/conda/etc/profile.d/conda.sh
conda activate dynov2
python -u test.py "$@"
