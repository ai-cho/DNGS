# dataset=$1 
# workspace=$2 
# export CUDA_VISIBLE_DEVICES=$3


# python train_llff.py  -s $dataset --model_path $workspace -r 2 --eval --n_sparse 250  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
#             --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
#             --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
#             --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
#             --scaling_lr 0.003 \
#             --shape_pena 0.002 --opa_pena 0.001 \
#             --test_iterations 100 1000 2000 3000 4500 6000 10000 --save_iterations 100 500 1000 3000 6000 10000\
#             --near 10

# # set a larger "--error_tolerance" may get more smooth results in visualization

            
# python render.py -s $dataset --model_path $workspace -r 8 --near 10  
# python spiral.py -s $dataset --model_path $workspace -r 8 --near 10 


# python metrics.py --model_path $workspace 

dataset_base=$1 
workspace_base=$2 
export CUDA_VISIBLE_DEVICES=$3

for SCENE in "aloe" "art" "flowers" "garbage" "picnic" "roses" ; do
    workspace=$workspace_base/$SCENE
    dataset=$dataset_base/$SCENE
    echo "training $workspace"
    python train_llff.py  -s $dataset --model_path $workspace -r 2 --eval --n_sparse 250  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
                --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
                --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
                --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
                --scaling_lr 0.003 \
                --shape_pena 0.002 --opa_pena 0.001 \
                --test_iterations 100 1000 2000 3000 4000 5000 6000 10000 --save_iterations 100 500 1000 3000 6000 10000\
                # --near 10

    python render.py -s $dataset --model_path $workspace -r 2 --near 10  
    python spiral.py -s $dataset --model_path $workspace -r 2 --near 10 

    python metrics.py --model_path $workspace 
done

SCENE="century"
workspace=$workspace_base/$SCENE
dataset=$dataset_base/$SCENE

echo "training $SCENE"
    python train_llff.py  -s $dataset --model_path $workspace -r 1 --eval --n_sparse 250  --rand_pcd --iterations 6000 --lambda_dssim 0.2 \
                --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
                --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
                --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
                --scaling_lr 0.003 \
                --shape_pena 0.002 --opa_pena 0.001 \
                --test_iterations 100 1000 2000 3000 4500 6000 10000 --save_iterations 100 500 1000 3000 6000 10000\
                --near 10
            
python render.py -s $dataset --model_path $workspace -r 1 --near 10  
python spiral.py -s $dataset --model_path $workspace -r 1 --near 10 

python metrics.py --model_path $workspace 