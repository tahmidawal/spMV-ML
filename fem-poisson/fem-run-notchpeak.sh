#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --mem=200GB
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1472227/fem-poisson/output.txt
#SBATCH -e /uufs/chpc.utah.edu/common/home/u1472227/fem-poisson/error.txt

##SBATCH --account=soc-np
##SBATCH --partition=soc-np

#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gres=gpu:a100:1

#SBATCH --nodelist=notch372


#SBATCH --mail-user=u1472227@utah.edu  
#SBATCH --mail-type=FAIL


set -e

module load gcc/11.2.0
module load python/3.10.3
module load cuda/12.5.0
module load cudnn/9.2.0.82-12-gpu
# module load ninja/1.11.1
# module load intel-oneapi-advisor/2023.1.0

source "$PWD/.venv/bin/activate"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
# export JAX_ENABLE_X64=1


mesh_file_path="/scratch/general/vast/u1472227/meshes/fem/unstructured-tube-250250.msh"
surface_tags="31"

# mesh_file_path="/scratch/general/vast/u1472227/meshes/fem/structured-box-250047.msh"
# surface_tags="13"

# batch_size=4


for p_order in {1..3}
# for p_order in 4;
do
    # echo "running mat-free version for p = $p_order"
    
    # rm -rf ./jax_cache
    # mkdir -p jax_cache
    # python main_mat_free.py $p_order $mesh_file_path $surface_tags

    # nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true \
    #     --force-overwrite true -o ./profiling/unstructured-tube-250250-p-$p_order-mat-free python main_mat_free.py $p_order $mesh_file_path $surface_tags


    # echo "running structured-mat-free version for p = $p_order"
    
    # rm -rf ./jax_cache
    # mkdir -p jax_cache
    # python main_structured_mat_free.py $p_order $mesh_file_path $surface_tags

    # nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true \
    #     --force-overwrite true -o ./profiling/unstructured-tube-250250-p-$p_order-structured-mat-free python main_structured_mat_free.py $p_order $mesh_file_path $surface_tags

    for batch_size in 2 4 8; do 

    echo "running mat-free-batched version for p = $p_order, batch_size = $batch_size"
    
    rm -rf ./jax_cache
    mkdir -p jax_cache
    python main_mat_free_batched.py $p_order $batch_size $mesh_file_path $surface_tags

    nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true \
        --force-overwrite true -o ./profiling/unstructured-tube-250250-p-$p_order-mat-free-batched-$batch_size python main_mat_free_batched.py $p_order $batch_size $mesh_file_path $surface_tags

    done


    # echo "running block-mat version for p = $p_order"
    
    # rm -rf ./jax_cache
    # mkdir -p jax_cache

    # python main_block_assembled.py $p_order $mesh_file_path $surface_tags

    # nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true \
    #     --force-overwrite true -o ./profiling/unstructured-tube-250250-p-$p_order-block-mat python main_block_assembled.py $p_order $mesh_file_path $surface_tags


    # echo "running global-asssembled version for p = $p_order"
    
    # rm -rf ./jax_cache
    # mkdir -p jax_cache

    # python main_global_assembled.py $p_order $mesh_file_path $surface_tags

    # nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true \
    #     --force-overwrite true -o ./profiling/unstructured-tube-250250-p-$p_order-global-asmbl python main_global_assembled.py $p_order $mesh_file_path $surface_tags

done

# rm -rf ./jax_cache

# mkdir -p jax_cache

# export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# python main.py compile

# advisor --collect=roofline --project-dir=./advi_results -- python main.py compile


# nvprof -m all --openacc-profiling off -o ./profiling/fem-poisson-profiling-nvprof python main.py compile


# nsys profile \
#     --trace=cuda,nvtx --stats true --force-overwrite true -o ./profiling/fem-poisson-profiling python main.py compile


# nsys profile --cuda-graph-trace node --trace=cuda,nvtx --stats true --force-overwrite true -o ./profiling/fem-poisson-profiling python main.py compile

# ncu --verbose --replay-mode application \
#     --force-overwrite --mode=launch-and-attach \
#     --target-processes all -o ./profiling/fem-poisson-profiling-ncu python main.py compile

# ncu --verbose --replay-mode application --graph-profiling node \
#     --force-overwrite --mode=launch-and-attach --set full \
#     --section SpeedOfLight_HierarchicalTensorRooflineChart --target-processes all -o ./profiling/fem-poisson-profiling-ncu python main.py compile


echo "=============== FEM Poisson Completed ! ==================="
