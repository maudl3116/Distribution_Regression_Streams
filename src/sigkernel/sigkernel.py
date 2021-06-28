import numpy as np
import torch
import torch.cuda
from numba import cuda

#from cython_backend import sig_kernel_batch_varpar, sig_kernel_Gram_varpar
from .cuda_backend import compute_sig_kernel_batch_varpar_from_increments_cuda, compute_sig_kernel_Gram_mat_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
# ===========================================================================================================
class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def __init__(self, add_time = 0):
        self.add_time = add_time

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        
        k = torch.bmm(X, Y.permute(0,2,1))

        if self.add_time!=0:
            fact = 1./self.add_time
            time_cov = fact*torch.arange(X.shape[1],device=X.device, dtype=X.dtype)[:,None]*fact*torch.arange(Y.shape[1],device=Y.device, dtype=Y.dtype)[None,:]
            k += time_cov[None,:,:]

        return k

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        
        K = torch.einsum('ipk,jqk->ijpq', X, Y)
        
        if self.add_time!=0:
            fact = 1./self.add_time
            time_cov = fact*torch.arange(X.shape[1],device=X.device, dtype=X.dtype)[:,None]*fact*torch.arange(Y.shape[1],device=Y.device, dtype=Y.dtype)[None,:]
            K += time_cov[None,None,:,:]
        return K


class RBFKernel():
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma, add_time = 0):
        self.sigma = sigma
        self.add_time = add_time

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))

        if self.add_time!=0:
            fact = 1./self.add_time
            time_component =(fact*torch.arange(X.shape[1],device=X.device, dtype=X.dtype)[:,None]-fact*torch.arange(Y.shape[1],device=Y.device, dtype=Y.dtype)[None,:])**2
            dist +=time_component[None,:,:]

        return torch.exp(-dist/self.sigma)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))

        if self.add_time:
            fact = 1./self.add_time
            time_component = (fact*torch.arange(X.shape[1],device=X.device, dtype=X.dtype)[:,None]-fact*torch.arange(Y.shape[1],device=Y.device, dtype=Y.dtype)[None,:])**2
            dist +=time_component[None,None,:,:]
 
        return torch.exp(-dist/self.sigma)
# ===========================================================================================================



# ===========================================================================================================
# Signature Kernel
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self,static_kernel, dyadic_order, _naive_solver=False, static_kernel_1=None, dyadic_order_1=None):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self.dyadic_order_1 = dyadic_order_1
        self._naive_solver = _naive_solver
        self.static_kernel_1 = static_kernel_1

    def compute_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        return _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver)

    def compute_Gram(self, X, Y, sym=False, return_sol_grid=False):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order, sym, self._naive_solver, return_sol_grid)

    def compute_Gram_rank_1(self, K_XX, K_XY, K_YY, lambda_, sym=False, inspect=False,centered=False):
        """Input: 
                  - K_XX: torch tensor of shape (batch_X, batch_X, length_X, length_X),
                  - K_YY: torch tensor of shape (batch_Y, batch_Y, length_Y, length_Y),
                  - K_XX: torch tensor of shape (batch_X, batchi
                  _Y, length_X, length_Y),
           Output: 
                  - matrix k(X^1[i]_T,Y^1[j]_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGramRank1.apply(K_XX, K_XY, K_YY, self.static_kernel_1, self.dyadic_order_1, lambda_, sym, self._naive_solver, inspect, centered)

    def compute_distance(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """
        
        assert not Y.requires_grad, "the second input should not require grad"

        k_XX = self.compute_kernel(X, X)
        k_YY = self.compute_kernel(Y, Y)
        k_XY = self.compute_kernel(X, Y)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY) 

    def compute_mmd(self, X, Y, estimator='b'):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True)
        K_YY = self.compute_Gram(Y, Y, sym=True)
        K_XY = self.compute_Gram(X, Y, sym=False)
        
        if estimator=='b':
            return torch.mean(K_XX) + torch.mean(K_YY) -2*torch.mean(K_XY)
        else:
            K_XX_m = (torch.sum(K_XX)-torch.sum(torch.diag(K_XX)))/(K_XX.shape[0]*(K_XX.shape[0]-1.))
            K_YY_m = (torch.sum(K_YY)-torch.sum(torch.diag(K_YY)))/(K_YY.shape[0]*(K_YY.shape[0]-1.))

            return K_XX_m + K_YY_m - 2.*torch.mean(K_XY) 

    def compute_mmd_rank_1(self, X, Y, lambda_, estimator='b', inspect=False, centered=False):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD rank-1 signature distance between samples X and samples Y
        """
        assert not X.requires_grad and not Y.requires_grad, "does not support automatic differentiation yet"

        # Compute Gram matrices order 0. Ex K_XY_0[i,j,p,q]= k( X[i,:,:p], Y[j,:, :q])
        K_XX_0 = self.compute_Gram(X, X, sym=True, return_sol_grid=True)   # shape (batch_X, batch_X, length_X, length_X)
        K_YY_0 = self.compute_Gram(Y, Y, sym=True, return_sol_grid=True)   # shape (batch_Y, batch_Y, length_Y, length_Y)
        K_XY_0 = self.compute_Gram(X, Y, sym=False, return_sol_grid=True)  # shape (batch_X, batch_Y, length_X, length_Y)

        # Compute Gram matrices rank 1. Ex K_XY_1[i,j]= k( X^1[i], Y^1[j] ) where X^1[i] = t -> E[k(X,.) | F_t](omega_i)
        K_XX_1 = self.compute_Gram_rank_1(K_XX_0, K_XX_0, K_XX_0, lambda_, sym=True, inspect=inspect,centered=centered)       # shape (batch_X, batch_X)
        K_YY_1 = self.compute_Gram_rank_1(K_YY_0, K_YY_0, K_YY_0, lambda_, sym=True, inspect=inspect,centered=centered)       # shape (batch_Y, batch_Y)
        K_XY_1 = self.compute_Gram_rank_1(K_XX_0, K_XY_0, K_YY_0, lambda_, sym=False, inspect=inspect,centered=centered)      # shape (batch_X, batch_Y)

        if inspect:
            K_XX_1, G_X = K_XX_1[0],  K_XX_1[1]
            K_YY_1, G_Y = K_YY_1[0],  K_YY_1[1]
            K_XY_1, G_XY = K_XY_1[0],  K_XY_1[1]
        # return K_XX_1, K_YY_1, K_XY_1
        if estimator=='b':
            if inspect:
                return torch.mean(K_XX_1) + torch.mean(K_YY_1) - 2.*torch.mean(K_XY_1), G_X, G_Y, G_XY
            return torch.mean(K_XX_1) + torch.mean(K_YY_1) - 2.*torch.mean(K_XY_1)
        else:
            K_XX_m = (torch.sum(K_XX_1)-torch.sum(torch.diag(K_XX_1)))/(K_XX_1.shape[0]*(K_XX_1.shape[0]-1.))
            K_YY_m = (torch.sum(K_YY_1)-torch.sum(torch.diag(K_YY_1)))/(K_YY_1.shape[0]*(K_YY_1.shape[0]-1.))
            if inspect:
                return K_XX_m + K_YY_m - 2.*torch.mean(K_XY_1), G_X, G_Y, G_XY
            return K_XX_m + K_YY_m - 2.*torch.mean(K_XY_1) 


class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""
 
    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, _naive_solver=False):

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X,Y)
        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM+1,NN+1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'
            
            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            K[:,0,:] = 1.
            K[:,:,0] = 1. 

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K), _naive_solver)
            K = K[:,:-1,:-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G_static,K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return K[:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):
    
        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=1),dim=2)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM+2, NN+2), device=G_static_rev.device, dtype=G_static_rev.dtype) 
            K_rev[:,0,:] = 1.
            K_rev[:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM,NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:,:-1,:-1]      

        # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        K_rev = flip(flip(K_rev,dim=1),dim=2)
        KK = K[:,:-1,:-1] * K_rev[:,1:,1:]   
        
        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.batch_kernel(Xh,Y) 
        G_h = G_h.reshape(A,M,D,N)
        G_h = G_h.permute(0,1,3,2) 

        Diff_1 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_1 =  tile( tile(Diff_1,1,2**dyadic_order)/float(2**dyadic_order),2, 2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_2 += - G_h[:,:-1,1:,:] + G_h[:,:-1,:-1,:] + (G_static[:,:-1,1:])[:,:,:,None] - (G_static[:,:-1,:-1])[:,:,:,None]
        Diff_2 = tile(tile(Diff_2,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (KK[:,:,:,None] * Diff_1)/h
        grad_2 = (KK[:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=2)
        grad_1 = torch.sum(grad_1.reshape(A,M-1,2**dyadic_order,D),axis=2)
        grad_2 = torch.sum(grad_2,axis=2)
        grad_2 = torch.sum(grad_2.reshape(A,M-1,2**dyadic_order,D),axis=2)

        grad_prev = grad_1[:,:-1,:] + grad_2[:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:,1:,:]],dim=1)   # /
        grad_incr = grad_prev - grad_1[:,1:,:]
        grad_points = torch.cat([(grad_2[:,0,:]-grad_1[:,0,:])[:,None,:],grad_incr,grad_1[:,-1,:][:,None,:]],dim=1)

        if Y.requires_grad:
            grad_points*=2

        return grad_output[:,None,None]*grad_points, None, None, None, None


class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False, _naive_solver=False, return_sol_grid=False):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X,Y)
        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G), _naive_solver)
            G = G[:,:,:-1,:-1]

        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_static_.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G,G_static)      
        ctx.sym = sym
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        if not return_sol_grid:
            return G[:,:,-1,-1]
        else:
            return G[:,:,::2**dyadic_order,::2**dyadic_order]


    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=2),dim=3)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G_rev[:,:,0,:] = 1.
            G_rev[:,:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G_rev), _naive_solver)

            G_rev = G_rev[:,:,:-1,:-1]

        # if on CPU
        else:
            G_rev = torch.tensor(sig_kernel_Gram_varpar(G_static_rev.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        G_rev = flip(flip(G_rev,dim=2),dim=3)
        GG = G[:,:,:-1,:-1] * G_rev[:,:,1:,1:]     

        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.Gram_matrix(Xh,Y) 
        G_h = G_h.reshape(A,B,M,D,N)
        G_h = G_h.permute(0,1,2,4,3) 

        Diff_1 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_1 =  tile(tile(Diff_1,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_2 += - G_h[:,:,:-1,1:,:] + G_h[:,:,:-1,:-1,:] + (G_static[:,:,:-1,1:])[:,:,:,:,None] - (G_static[:,:,:-1,:-1])[:,:,:,:,None]
        Diff_2 = tile(tile(Diff_2,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (GG[:,:,:,:,None] * Diff_1)/h
        grad_2 = (GG[:,:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=3)
        grad_1 = torch.sum(grad_1.reshape(A,B,M-1,2**dyadic_order,D),axis=3)
        grad_2 = torch.sum(grad_2,axis=3)
        grad_2 = torch.sum(grad_2.reshape(A,B,M-1,2**dyadic_order,D),axis=3)

        grad_prev = grad_1[:,:,:-1,:] + grad_2[:,:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:,:,1:,:]], dim=2)   # /
        grad_incr = grad_prev - grad_1[:,:,1:,:]
        grad_points = torch.cat([(grad_2[:,:,0,:]-grad_1[:,:,0,:])[:,:,None,:],grad_incr,grad_1[:,:,-1,:][:,:,None,:]],dim=2)

        if sym:
            grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None
        else:
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None


class _SigKernelGramRank1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K_XX, K_XY, K_YY, static_kernel, dyadic_order,lambda_, sym=False, _naive_solver=False,inspect=False,centered=False):

        A = K_XX.shape[0]
        B = K_YY.shape[0]
        M = K_XX.shape[2]
        N = K_YY.shape[2] 

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^1[i]_s,Y^1[j]_t)
        G_base = innerprodCKME(K_XX, K_XY, K_YY, lambda_, static_kernel, sym=sym,centered=centered)          # <--------------------- this is the only change compared to rank 0

        G_base_ = G_base[:,:,1:,1:] + G_base[:,:,:-1,:-1] - G_base[:,:,1:,:-1] - G_base[:,:,:-1,1:] 
        
        G_base_ = tile(tile(G_base_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if K_XX.device.type=='cuda':

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=G_base.device, dtype=G_base.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_base_.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G), _naive_solver)
            G = G[:,:,:-1,:-1]

        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_base_.detach().numpy(), sym, _naive_solver), dtype=G_base.dtype, device=G_base.device)
        if inspect:
            return G[:,:,-1,-1], G_base
        return G[:,:,-1,-1]
        # return G_base_ 


    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None

# ===========================================================================================================



# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================
# innerprodCKME with torch.cholesky_inverse
# ===========================================================================================================
def innerprodCKME(K_XX, K_XY, K_YY, lambda_, static_kernel, sym=False, centered=False):
    m  = K_XX.shape[0]
    n  = K_YY.shape[0]
    LX = K_XX.shape[2]
    LY = K_YY.shape[2]

    H_X = torch.eye(m,device=K_XX.device,dtype=K_XX.dtype) - (1./m)*torch.ones((m,m),device=K_XX.device,dtype=K_XX.dtype)       # centering matrix
    H_Y = torch.eye(n,device=K_YY.device,dtype=K_YY.dtype) - (1./n)*torch.ones((n,n),device=K_YY.device,dtype=K_YY.dtype)       # centering matrix

    G = torch.zeros((m, n, LX, LY), dtype=K_XX.dtype, device=K_XX.device)

    if centered:
        # H_X K_XY H_Y batchwise , K_XY is (M,N,L,L) and H_Y is (N,N). But to compute batchwise, one needs K to be (L,L,N,M) -> L,L,N,M
        # K_XY.t() is (L,L,N,M) x H_X -> (L,L,N,M)
        # can be multiplied left H_Y -> (L,L,N,M)
        K_XY = torch.matmul(H_Y,torch.matmul(K_XY.T,H_X)).T     
        K_XX = torch.matmul(H_X,torch.matmul(K_XX.T,H_X)).T  
        K_YY = torch.matmul(H_Y,torch.matmul(K_YY.T,H_Y)).T 

    # inv_X = torch.zeros((m, m, LX), dtype=K_XX.dtype, device=K_XX.device)
    # for p in range(LX):
    #     inv_X[:,:,p] = torch.cholesky_inverse(K_XX[:,:,p,p] + lambda_*torch.eye(m,dtype=K_XX.dtype, device=K_XX.device))

    # if sym:
    #     inv_Y = inv_X
    # else:
    #     inv_Y = torch.zeros((n, n, LY), dtype=K_YY.dtype, device=K_YY.device)
    #     for q in range(LY):
    #         inv_Y[:,:,q] = torch.cholesky_inverse(K_YY[:,:,q,q] + lambda_*torch.eye(n,dtype=K_YY.dtype, device=K_YY.device))


    to_inv_XX = torch.zeros((m, m, LX), dtype=K_XX.dtype, device=K_XX.device)
    for p in range(LX):
        to_inv_XX[:,:,p] = K_XX[:,:,p,p]
    to_inv_XX += lambda_*torch.eye(m,dtype=K_XX.dtype, device=K_XX.device)[:,:,None]
    to_inv_XX = to_inv_XX.T
    inv_X = torch.linalg.inv(to_inv_XX)
    inv_X = inv_X.T
    
    if sym:
        inv_Y = inv_X
    else:
        to_inv_YY = torch.zeros((n, n, LY), dtype=K_YY.dtype, device=K_YY.device)
        for q in range(LY):
            to_inv_YY[:,:,q] = K_YY[:,:,q,q]
        to_inv_YY += lambda_*torch.eye(n,dtype=K_YY.dtype, device=K_YY.device)[:,:,None]
        to_inv_YY = to_inv_YY.T
        inv_Y = torch.linalg.inv(to_inv_YY)
        inv_Y = inv_Y.T


    for p in range(LX): # TODO: to optimize (e.g. when X=Y)
        WX = inv_X[:,:,p]
        WX_ = torch.matmul(K_XX[:,:,p,p].t(),WX)
        for q in range(LY):
            WY = inv_Y[:,:,q]  
            WY_ = torch.matmul(WY,K_YY[:,:,q,q]    )
            if isinstance(static_kernel, LinearKernel):
                G[:,:,p,q] = torch.matmul(WX_,torch.matmul(K_XY[:,:,-1,-1],WY_))
                if static_kernel.add_time!=0:
                    fact = 1./static_kernel.add_time
                    G[:,:,p,q] +=fact*p*fact*q
            else:
                # WX_r = torch.matmul(inv_X[:,:,p],K_XX[:,:,q,q])   
                # WY_l = torch.matmul(K_YY[:,:,p,p].t(),inv_Y[:,:,q])  
                G_cross  = -2*torch.matmul(WX_,torch.matmul(K_XY[:,:,-1,-1],WY_))
                G_row = torch.diag(torch.matmul(WX_,torch.matmul(K_XX[:,:,-1,-1],WX_.t())))[:,None]
                G_col = torch.diag(torch.matmul(WY_.t(),torch.matmul(K_YY[:,:,-1,-1],WY_)))[None,:]
                dist  = G_cross + G_row + G_col
                if static_kernel.add_time!=0:
                    fact = 1./static_kernel.add_time
                    dist += (fact*p-fact*q)**2
                G[:,:,p,q] = torch.exp(-dist/static_kernel.sigma)
    return G
# ===========================================================================================================

# ===========================================================================================================
# Hypothesis test functionality
# ===========================================================================================================
def c_alpha(m, alpha):
    return 4. * np.sqrt(-np.log(alpha) / m)

def hypothesis_test(y_pred, y_test, static_kernel, confidence_level=0.99, dyadic_order=0, mmd_order=0,lambda_=0):
    """Statistical test based on MMD distance to determine if 
       two sets of paths come from the same distribution.
    """

    k_sig = SigKernel(static_kernel, dyadic_order)

    m = max(y_pred.shape[0], y_test.shape[0])
    
    if mmd_order == 0:
        TU = k_sig.compute_mmd(y_pred,y_test)  
    else:
        TU = k_sig.compute_mmd_rank_1(y_pred, y_test, lambda_)  
  
    c = torch.tensor(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level*100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level*100}% confidence')
# ===========================================================================================================








# ===========================================================================================================
# Deprecated implementation (just for testing)
# ===========================================================================================================
def SigKernel_naive(X, Y, static_kernel, dyadic_order=0, _naive_solver=False):

    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^i_t)
    G_static = static_kernel.batch_kernel(X,Y)
    G_static = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
    G_static = tile(tile(G_static,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,i,j].clone()

            k_10 = K_XY[:, i + 1, j].clone()
            k_01 = K_XY[:, i, j + 1].clone()
            k_00 = K_XY[:, i, j].clone()

            if _naive_solver:
                K_XY[:, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)
            
    return K_XY[:, -1, -1]


class SigLoss_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigLoss_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self,X,Y):

        k_XX = SigKernel_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_YY = SigKernel_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_XY = SigKernel_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY)


def SigKernelGramMat_naive(X,Y,static_kernel,dyadic_order=0,_naive_solver=False):

    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A,B, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:,:, 0, :] = 1.
    K_XY[:,:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^j_t)
    G_static = static_kernel.Gram_matrix(X,Y)
    G_static = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
    G_static = tile(tile(G_static,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,:,i,j].clone()

            k_10 = K_XY[:, :, i + 1, j].clone()
            k_01 = K_XY[:, :, i, j + 1].clone()
            k_00 = K_XY[:, :, i, j].clone()

            if _naive_solver:
                K_XY[:, :, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, :, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, :, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:,:, -1, -1]


class SigMMD_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigMMD_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self, X, Y):

        K_XX = SigKernelGramMat_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        K_YY = SigKernelGramMat_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)  
        K_XY = SigKernelGramMat_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        
        return torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY) 
