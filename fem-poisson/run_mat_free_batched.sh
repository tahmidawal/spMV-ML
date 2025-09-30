set -e

p_order=3
mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/unstructured-tube-2684.msh"
# mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/structured-box-24389.msh"
# mesh_file_path="/home/budvin/research/Partitioning/Meshes/10k_hex/1582416_sf_hexa.mesh"
surface_tags="31"
# surface_tags="13"

batch_size=10




rm -rf ./jax_cache

mkdir -p jax_cache

python main_mat_free_batched.py $p_order $batch_size $mesh_file_path $surface_tags



# nsys profile --stats true --force-overwrite true --trace=cuda,nvtx \
#     --cuda-graph-trace=node -o nsys-report-mat-free-batched \
#     python main_mat_free_batched.py $p_order $batch_size $mesh_file_path $surface_tags




