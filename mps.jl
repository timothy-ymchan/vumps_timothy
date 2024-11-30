using TensorKit
using KrylovKit
using TensorOperations


function uMPS_normalize(A::AbstractTensorMap)
    """
        uMPS_normalize(A::AbstractTensorMap)

    Normalize uMPS so that the largest eigenvalue is 1

    ## Parameters
    - `A::AbstractTensorMap`: Tensor on the top. Assume leg index (chi,phy,chi') <- {0}

    ## Return 
    - `An::AbstractTensorMap`: Normalized uMPS
    
    """
    # Remark: T and T^tr have the same eigenspectrum, so it does not matter which way we compute the transfer matrix
    T = transfer_mat(A,A) 
    # Currently only implemented for trivial sector
    Tmat = block(T,Trivial())
    vals, _, _ = eigsolve(Tmat) # By default returns the largest, see the documentation for KrylovKit

    return A/sqrt(vals[1]) # Julia array indices start from 1
end

function fix_point(T::AbstractTensorMap;ϵ_schur::Real=1e-15,ϵ_eig::Real=1e-8)
    # Treat T as a matrix, compute fix point tensors along the prescribed direction up to error ϵ
    # Compute fix point up to a phase
    # Again only implemented for trivial sector
    Tmat = block(T,Trivial())
    vals, vecs , _ = eigsolve(Tmat;tol=ϵ_schur) 
    
    ϵ = abs(abs(vals[1])-1)
    if ϵ > ϵ_eig
        throw("Eigenvalue not equal to 1 up to prescribed precision $ϵ_eig (Current precision: $ϵ)")
    end
    return vals[1],vecs[1] # Return fix point value and fix point eigenvalue 
end

function transfer_mat(A::AbstractTensorMap,B::AbstractTensorMap,direction::Symbol=:left)::AbstractTensorMap
    """
        transfer_mat(A,B [,direction])

    Compute transfer matrix 

    ## Parameters
    - `A::AbstractTensorMap`: Tensor on the top. Assume leg index (chi,phy,chi') <- {0}
    - `B::AbstractTensorMap`: Tensor on the bottom. Assume leg index (chi,phy,chi') <- {0}
    - `direction::Symbol`: The input direction of the transfer matrix. Either `:left` or `:right`

    ## Return 
    - `T::AbstractTensorMap`: Transfer matrix computed from `A[i,m,j]*conj(B[k,m,l])`
    
    """
    @tensor T[i,j,k,l] := A[i,m,k]*conj(B[j,m,l])

    if direction == :left 
        return permute(T,(3,4,),(1,2,))
    elseif direction == :right
        return permute(T,(1,2,),(3,4,))
    else
        throw("$direction is not a valid direction!")
    end

end


# Algorithm taken from https://arxiv.org/abs/1810.07006
function _left_orthonormal_form(A::AbstractTensorMap,L0::AbstractTensorMap,η::Real=1e-10,maxiter::Integer=1000)
    L = normalize(L0)
    Lold = L
    AL,L = leftorth(ncon([L,A],[[-1,1],[1,-2,-3]]))
    λ = norm(L)
    normalize!(L)

    δ = norm(L-Lold)

    iter = 1
    while δ > η && iter <= maxiter
        # Implement arnoldi method later, previously it made it worse
        # 
        # 
        # 
        
        Lold = L # Do QR Decomposition
        AL,L = leftorth(ncon([L,A],[[-1,1],[1,-2,-3]]))
        λ = norm(L)
        normalize!(L)
        δ = norm(L-Lold)

        iter += 1
    end
    return AL,L,λ
end

function _right_orthonormal_form(A::AbstractTensorMap,R0::AbstractTensorMap,η::Real=1e-10,maxiter::Integer=1000)
    AR,R,λ = _left_orthonormal_form(permute(A,(3,2,1),()),permute(R0,(2,),(1,)),η,maxiter)
    return permute(AR,(3,2,1),()), permute(R,(2,),(1,)),λ
end