#!/bin/bash
export SCHRODINGER=/home/zdy/schrodinger2021-3

# 只添加 Schrödinger 的 Python 包，不加 lib
export PYTHONPATH=$SCHRODINGER/internal/lib/python3.8/site-packages:$PYTHONPATH

# 激活 Conda
source /home/zdy/anaconda3/etc/profile.d/conda.sh
conda activate FlexPose

# 直接用 Conda 的 python
python /home/zdy/Project2/data_processing/phy_geo.py --config /home/zdy/Project2/config.yaml
