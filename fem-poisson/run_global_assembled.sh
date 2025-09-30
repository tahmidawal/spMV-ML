set -e

p_order=3
mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/unstructured-tube-2684.msh"
surface_tags="31"
# mesh_file_path="/home/budvin/research/fem-poisson/tmp-test/meshes/structured-box-2744.msh"
# surface_tags="13"



rm -rf ./jax_cache

mkdir -p jax_cache

python main_global_assembled.py $p_order $mesh_file_path $surface_tags



# nsys profile --stats true --force-overwrite true --trace=cuda,nvtx \
#     --cuda-graph-trace=node -o nsys-report-global-assembl \
#     python main_global_assembled.py $p_order $mesh_file_path $surface_tags







    # /home/budvin/research/fem-poisson/tmp-test/meshes/box-mesh-1.msh 29 
    # /home/budvin/research/fem-poisson/tmp-test/meshes/extruded-mesh-1.msh 21 
    # /home/budvin/research/fem-poisson/tmp-test/meshes/small-mesh.msh 27 
    # /home/budvin/research/fem-poisson/tmp-test/meshes/unstructured-2920.msh 21 
    # /home/budvin/research/fem-poisson/tmp-test/meshes/unstructured2.msh 31 

    # /home/budvin/research/fem-poisson/tmp-test/meshes/shell-mesh-large.msh 27 
    # /home/budvin/research/fem-poisson/tmp-test/meshes/regular-box.msh 13 

