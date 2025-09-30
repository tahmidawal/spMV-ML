# import numpy as np
import jax.numpy as jnp


k = 2*jnp.pi
u_sol_func = lambda x, y, z : jnp.sin(k*x)*jnp.sin(k*y)*jnp.sin(k*z)
source_func = lambda x, y, z : 3*k*k*jnp.sin(k*x)*jnp.sin(k*y)*jnp.sin(k*z)
boundary_func = u_sol_func
# u_sol_func = lambda x, y, z : (x**2)*(y**2)*(z**2)
# source_func = lambda x, y, z : -2*(x**2)*(y**2) -2*(y**2)*(z**2) -2* (x**2)*(z**2)


# u_sol_func = lambda x, y, z : jnp.sin(k*x)*jnp.sin(k*y)*jnp.sin(k*z)*(x**2 + y**2 + z**2)
# source_func = lambda x, y, z : -(3*jnp.sin(k*x)*jnp.sin(k*y)*jnp.sin(k*z)*(2-k*k*(x**2 + y**2 + z**2))) \
#                                - jnp.sin(k*y)*jnp.sin(k*z)*4*k*x*jnp.cos(k*x) \
#                                - jnp.sin(k*x)*jnp.sin(k*z)*4*k*y*jnp.cos(k*y) \
#                                - jnp.sin(k*x)*jnp.sin(k*y)*4*k*z*jnp.cos(k*z) 
                               
# boundary_func = u_sol_func

coeff_func = lambda x, y, z : 1