import jax
import numpy as np



def run_IIA_kernel(A, xs, N, eN):

  out_type = jax.ShapeDtypeStruct((np.uint64((N**3)*eN),), A.dtype)
  ys = jax.ffi.ffi_call("IIA_kernel", out_type)(A, xs, eN=np.uint64(eN))
  return ys



def run_IAI_kernel(A, xs, N, eN):

  out_type = jax.ShapeDtypeStruct((np.uint64((N**3)*eN),), A.dtype)
  ys = jax.ffi.ffi_call("IAI_kernel", out_type)(A, xs, eN=np.uint64(eN))
  return ys


def run_AII_kernel(A, xs, N, eN):

  out_type = jax.ShapeDtypeStruct((np.uint64((N**3)*eN),), A.dtype)
  ys = jax.ffi.ffi_call("AII_kernel", out_type)(A, xs, eN=np.uint64(eN))
  return ys