CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="MyData" \
    --resume="resume/best_checkpoint.pth" \
    --vis_dir="results/MyData"