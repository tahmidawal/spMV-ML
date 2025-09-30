set -e

p_order=3
mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/unstructured-tube-2684.msh"
# mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/structured-box-250047.msh"
# mesh_file_path="/home/budvin/research/Partitioning/Meshes/10k_hex/1582416_sf_hexa.mesh"
surface_tags="31"
# surface_tags="13"


cd mat_free_poisson_v2/kron_gemm_kernel
cmake -DN=$(( $p_order + 1 )) -DCMAKE_BUILD_TYPE=Release\
    -DJAXLIB_INCLUDE_DIR="/home/budvin/research/fem-poisson/.venv/lib64/python3.10/site-packages/jaxlib/include" \
    -DCMAKE_INSTALL_PREFIX="build/install" -B build -S . 
cmake --build build
cmake --install build

cd ../../

rm -rf ./jax_cache

mkdir -p jax_cache

python main_mat_free_v2.py $p_order $mesh_file_path $surface_tags



# nsys profile --stats true --force-overwrite true --trace=cuda,nvtx \
#     --cuda-graph-trace=node -o nsys-report-mat-free_v2 \
#     python main_mat_free_v2.py $p_order $mesh_file_path $surface_tags




