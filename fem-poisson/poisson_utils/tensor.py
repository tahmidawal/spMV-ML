# import numpy as np
import jax.numpy as jnp
import mat_free_poisson_v2.kron_gemm_kernel.wrapper as gemm_wrapper

def ABCx(A, B, C, x, n):
    """
    Calculates (A⊗B⊗C)x using matmul operations
    A: shape (n, n)
    B: shape (n, n)
    C: shape (n, n)
    x: shape (n^3,)
    """
    w1 = jnp.matmul(C, x.reshape(n,n**2, order='F'))
    w2 = jnp.matmul(w1.reshape(n**2,n, order='C'), jnp.transpose(B))

    return jnp.matmul(A, w2.reshape(n,n**2, order='F')).reshape(n**3,order='C')


    # ABC = np.kron(A, np.kron(B, C))

    # return ABC @ x
    

def IIAx_mat(A, x, n):
    """
    Calculates (I⊗I⊗A)x using matmul operations
    A: shape (n, n)
    x: shape (n^3,)
    """
    return jnp.matmul(A, x.reshape(n, n**2, order='F')).reshape(n**3,order='F')

def IIAx_tensor(A, x, n):
    """
    Calculates (I⊗I⊗A)x using kron operations. 
    (explicitly forms I⊗I⊗A matrix)
    A: shape (n, n)
    x: shape (n^3,)
    """
    I = jnp.identity(n)
    return jnp.matvec(jnp.kron(jnp.kron(I, I), A), x)

def IAIx_mat(A, x, n):
    """
    Calculates (I⊗A⊗I)x using matmul operations
    A: shape (n, n)
    x: shape (n^3,)
    """
    x_batch =  x.reshape(n,n,n, order='F')
    Ax_batch = jnp.einsum('ij,bjk->bik', A, x_batch)
    return Ax_batch.reshape(n**3,order='F')

def IAIx_tensor(A, x, n):
    """
    Calculates (I⊗A⊗I)x using kron operations. 
    (explicitly forms I⊗A⊗I matrix)
    A: shape (n, n)
    x: shape (n^3,)
    """
    I = jnp.identity(n)
    return jnp.matvec(jnp.kron(jnp.kron(I, A), I), x)


def AIIx_mat(A, x, n):
    """
    Calculates (A⊗I⊗I)x using matmul operations
    A: shape (n, n)
    x: shape (n^3,)
    """
    return jnp.matmul(x.reshape(n**2, n, order="F"), A.transpose()).reshape(n**3,order='F')

def AIIx_tensor(A, x, n):
    """
    Calculates (A⊗I⊗I)x using kron operations. 
    (explicitly forms A⊗I⊗I matrix)
    A: shape (n, n)
    x: shape (n^3,)
    """
    I = jnp.identity(n)
    return jnp.matvec(jnp.kron(jnp.kron(A, I), I), x)


def grad3(A, x, n):
    """
    Applies 1d gradient operator (A) in all 3 dimensions. Uses matmul impl. of
    tensor products.
    A: shape (n, n)
    x: shape (n^3,)
    """
    dx = IIAx_mat(A, x, n)
    dy = IAIx_mat(A, x, n)
    dz = AIIx_mat(A, x, n)

    return dx, dy, dz

def kron_kron(A, B, C):
    """
    Calculates (A⊗B⊗C) using jnp.kron
    """
    return jnp.kron(jnp.kron(A, B), C)




## ============ batched routines ==============


def ABCx_batched(A, B, C, x, n, batch):
    """
    Batched version of (A⊗B⊗C)x
    A, B, C: shape (n, n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """

    x3 = x.reshape(batch, n, n, n)  
    y = jnp.einsum('ia,jb,kc,mabc->mijk', A, B, C, x3)
    return y.reshape(batch, n**3)



def IIAx_batched(A, x, n, batch):
    """
    Batched version of (I⊗I⊗A)x
    A: shape (n, n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """
   
    x3 = x.reshape((batch, n, n, n), order='F') 
    y3 = jnp.einsum('ia,bajk->bijk', A, x3)
    return y3.reshape((batch, n**3), order='F')



def IAIx_batched(A, x, n, batch):
    """
    Batched version of (I⊗A⊗I)x
    A: shape (n, n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """
    
    x3 = x.reshape((batch, n, n, n), order='F')
    y3 = jnp.einsum('jl,bilk->bijk', A, x3)
    return y3.reshape((batch, n**3), order='F')


def AIIx_batched(A, x, n, batch):
    """
    Batched version of (A⊗I⊗I)x
    A: shape (n, n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """
    x3 = x.reshape((batch, n, n, n), order='F')
    y3 = jnp.einsum('bijk,mk->bijm', x3, A)
    return y3.reshape((batch, n**3), order='F')



def grad3_batched(A, x, n, batch):
    """
    Applies 1d gradient operator (A) in all 3 dimensions for batched x input. 
    A: shape (n, n)
    x: shape (batch, n^3)
    """
    dx = IIAx_batched(A, x, n, batch)
    dy = IAIx_batched(A, x, n, batch)
    dz = AIIx_batched(A, x, n, batch)

    return dx, dy, dz



# =========== batched routines with explicitly formed I⊗A for batching where I is batchxbatch identity matrix

def reshape_xs_for_IIA(xs, n):
    b = xs.shape[0]
    n_squared = n**2
    
    # Reshape to (b, n^2, n) to group elements by their position modulo n
    reshaped = xs.reshape(b, n_squared, n)
    
    # Transpose to (b, n, n^2) to separate the n groups
    transposed = jnp.transpose(reshaped, (0, 2, 1))
    
    # Reshape to final form: (b*n, n^2)
    result = transposed.reshape(b * n, n_squared)
    return result

def reshape_result_for_IIA(ys, n):
    b_times_n, n_squared = ys.shape
    b = b_times_n // n
    
    # Reshape to (b, n, n^2)
    reshaped = ys.reshape(b, n, n_squared)
    
    # Transpose to (b, n^2, n) to group elements back
    transposed = jnp.transpose(reshaped, (0, 2, 1))
    
    # Reshape to final form: (b, n^3)
    original_shaped = transposed.reshape(b, n**3)
    
    return original_shaped

def reshape_xs_for_IAI(xs, n):
    b = xs.shape[0]
    
    # Reshape each batch from (n^3,) to (n, n^2)
    # For n=2: (8,) -> (2, 4)
    reshaped = xs.reshape(b, n, n**2)
    
    # Further reshape to (n, n, n) to create n x n x n structure
    # For n=2: (2, 4) -> (2, 2, 2)
    blocks = reshaped.reshape(b, n, n, n)
    
    # Transpose to group blocks properly: (b, n, n, n) -> (b, n, n, n)
    # We want to extract blocks in the pattern shown
    transposed = jnp.transpose(blocks, (0, 2, 1, 3))
    
    # Reshape to (b*n, n^2)
    result = transposed.reshape(b * n, n**2)
    
    return result

def reshape_result_for_IAI(ys, n):
    b_times_n, n_squared = ys.shape
    b = b_times_n // n
    
    # Reshape to (b, n, n^2)
    reshaped = ys.reshape(b, n, n_squared)
    
    # Reshape to (b, n, n, n)
    blocks = reshaped.reshape(b, n, n, n)
    
    # Transpose back: (b, n, n, n) -> (b, n, n, n)
    transposed = jnp.transpose(blocks, (0, 2, 1, 3))
    
    # Reshape to (b, n, n^2) then (b, n^3)
    intermediate = transposed.reshape(b, n, n**2)
    original = intermediate.reshape(b, n**3)
    
    return original


def reshape_xs_for_AII(xs, n):
    b = xs.shape[0]
    
    # Reshape from (b, n^3) to (b, n, n^2)
    reshaped = xs.reshape(b, n, n**2)
    
    # Reshape to final form: (b*n, n^2)
    result = reshaped.reshape(b * n, n**2)
    
    return result

def reshape_result_for_AII(ys, n):
    b_times_n, n_squared = ys.shape
    b = b_times_n // n
    
    # Reshape to (b, n, n^2)
    reshaped = ys.reshape(b, n, n_squared)
    
    # Reshape to final form: (b, n^3)
    original = reshaped.reshape(b, n**3)
    
    return original

def ABCx_batched_bigABC(bigA, bigB, bigC, x, n):
    """
    Batched version of (A⊗B⊗C)x
    bigA, bigB, bigC: shape (batch*n, batch*n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """

    y = AIIx_batched_bigA(bigA, 
                          IAIx_batched_bigA(bigB, 
                                            IIAx_batched_bigA(bigC, x, n), n), n)
    return y



def IIAx_batched_bigA(bigA, x, n):
    """
    Batched version of (I⊗I⊗A)x
    bigA: shape (batch*n, batch*n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """

    xs_reshaped = reshape_xs_for_IIA(x, n)

    # Matmul (b*n, b*n) @ (b*n, n²)ᵗ
    y_mat = jnp.matmul(bigA, xs_reshaped)  # (b*n², n)

    return reshape_result_for_IIA(y_mat, n)



def IAIx_batched_bigA(bigA, x, n):
    """
    Batched version of (I⊗A⊗I)x
    bigA: shape (batch*n, batch*n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """
    
    xs_reshaped = reshape_xs_for_IAI(x, n)

    # Matmul (b*n, b*n) @ (b*n, n²)ᵗ
    y_mat = jnp.matmul(bigA, xs_reshaped)  # (b*n², n)

    return reshape_result_for_IAI(y_mat, n)


def AIIx_batched_bigA(bigA, x, n):
    """
    Batched version of (A⊗I⊗I)x
    bigA: shape (batch*n, batch*n)
    x: shape (batch, n^3)
    Returns: shape (batch, n^3)
    """
    xs_reshaped = reshape_xs_for_AII(x, n)

    # Matmul (b*n, b*n) @ (b*n, n²)ᵗ
    y_mat = jnp.matmul(bigA, xs_reshaped)  # (b*n², n)

    return reshape_result_for_AII(y_mat, n)



def grad3_batched_bigA(bigA, x, n):
    """
    Applies 1d gradient operator (A) in all 3 dimensions for batched x input. 
    bigA: shape (batch*n, batch*n)
    x: shape (batch, n^3)
    """
    dx = IIAx_batched_bigA(bigA, x, n)
    dy = IAIx_batched_bigA(bigA, x, n)
    dz = AIIx_batched_bigA(bigA, x, n)

    return dx, dy, dz

# ============== v2 version functions ================


def IIA_xs_v2(A, xs):
    """
    Apply (I ⊗ I ⊗ A) to a batched input vector.
    The batch is assumed to be ordered such that the first axis (x) is the 
    fastest varying.

    Args:
        A   : A square matrix of shape (n, n)
        xs  : 1D array of shape (n^3 * b,), where b is the batch size.

    Returns:
        ys  : 1D array of shape (n^3 * b,), representing the result of applying 
              (I ⊗ I ⊗ A) to each vector in the batch, with the same layout as
              `xs`.
    """
    N = A.shape[0]
    eN = xs.shape[0] // (N**3)

    ys = gemm_wrapper.run_IIA_kernel(A, xs, N, eN)
    return ys



def IAI_xs_v2(A, xs):
    """
    Apply (I ⊗ A ⊗ I) to a batched input vector.
    The batch is assumed to be ordered such that the first axis (x) is the 
    fastest varying.

    Args:
        A   : A square matrix of shape (n, n)
        xs  : 1D array of shape (n^3 * b,), where b is the batch size.

    Returns:
        ys  : 1D array of shape (n^3 * b,), representing the result of applying 
              (I ⊗ A ⊗ I) to each vector in the batch, with the same layout as
              `xs`.
    """
    N = A.shape[0]
    eN = xs.shape[0] // (N**3)

    ys = gemm_wrapper.run_IAI_kernel(A, xs, N, eN)

    return ys


def AII_xs_v2(A, xs):
    """
    Apply (A ⊗ I ⊗ I) to a batched input vector.
    The batch is assumed to be ordered such that the first axis (x) is the 
    fastest varying.

    Args:
        A   : A square matrix of shape (n, n)
        xs  : 1D array of shape (n^3 * b,), where b is the batch size.

    Returns:
        ys  : 1D array of shape (n^3 * b,), representing the result of applying 
              (A ⊗ I ⊗ I) to each vector in the batch, with the same layout as
              `xs`.
    """
    N = A.shape[0]
    eN = xs.shape[0] // (N**3)

    ys = gemm_wrapper.run_AII_kernel(A, xs, N, eN)

    return ys



def ABC_xs_v2(A, B, C, xs):
    """
    Apply (A ⊗ B ⊗ C) to a batched input vector.
    The batch is assumed to be ordered such that the first axis (x) is the 
    fastest varying.

    Args:
        A   : A square matrix of shape (n, n)
        B   : A square matrix of shape (n, n)
        C   : A square matrix of shape (n, n)
        xs  : 1D array of shape (n^3 * b,), where b is the batch size.

    Returns:
        ys  : 1D array of shape (n^3 * b,), representing the result of applying 
              (A ⊗ B ⊗ C) to each vector in the batch, with the same layout as
              `xs`.
    """
    ys = AII_xs_v2(A,IAI_xs_v2(B,IIA_xs_v2(C, xs)))
    return ys



def grad3_v2(A, x):
    """
    Applies 1d gradient operator (A) in all 3 dimensions for batched x input. 

    Args:
        A   : A square matrix of shape (n, n)
        xs  : 1D array of shape (n^3 * b,), where b is the batch size.
    """
    dx = IIA_xs_v2(A, x)
    dy = IAI_xs_v2(A, x)
    dz = AII_xs_v2(A, x)

    return dx, dy, dz