#!/bin/bash

#SBATCH -J wubaodong-job
#SBATCH -o /mnt/petrelfs/gaopeng/zst/logs/ar-image-generation-decoding-toolkit/gen-%j.out
#SBATCH -e /mnt/petrelfs/gaopeng/zst/logs/ar-image-generation-decoding-toolkit/gen-%j.err
srun python /mnt/petrelfs/gaopeng/zst/ar-image-generation-decoding-toolkit/generate.py \
    --model_path /mnt/petrelfs/gaopeng/zst/models/Lumina-mGPT-7B-1024 \
    --save_path /mnt/petrelfs/gaopeng/zst/ar-image-generation-decoding-toolkit/results \
    --temperature 1.0 \
    --top_k 4000 \
    --cfg 4.0 \
    --n 1 \
    --width 1024 \
    --height 1024 \
    --lp_list 4

# sbatch -p lumina --gres=gpu:1 --cpus-per-task 8 -n1 --ntasks-per-node=1 --quotatype=spot --job-name ar /mnt/petrelfs/gaopeng/zst/ar-image-generation-decoding-toolkit/scripts/run.sh