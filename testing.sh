dataset=$1
dataset_base=$2 
workspace_base=$3 
export CUDA_VISIBLE_DEVICES=$4

python train_llff.py  -s $dataset_base --model_path $workspace_base -r 2 --eval --n_sparse 250  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
            --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
            --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
            --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
            --scaling_lr 0.003 \
            --shape_pena 0.002 --opa_pena 0.001 \
            --test_iterations 100 1000 2000 3000 4500 6000 10000 --save_iterations 100 500 1000 3000 6000 10000\
            --near 10 \
            --dataset $dataset \


python render.py -s $dataset_base --model_path $workspace_base -r 2 --near 10  
python spiral.py -s $dataset_base --model_path $workspace_base -r 2 --near 10 

python metrics.py --model_path $workspace_base 
