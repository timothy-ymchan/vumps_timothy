using TensorKit
using KrylovKit
using TensorOperations
include("../mps.jl")

function main()

    # Generate a random uMPS for testing
    Hib_phy = ℂ^3
    Hib_chi = ℂ^2
    A = uMPS_normalize(randn(Hib_chi⊗Hib_phy⊗Hib_chi'))
    fix_point(transfer_mat(A,A))
    L0 = rand(Hib_chi,Hib_chi)
    R0 = rand(Hib_chi',Hib_chi')

    # Testing left orthonormal form code
    AL,L,λ1 = _left_orthonormal_form(A,L0,1e-15);
    _check_left_canonical(A,AL,L;verbose=true);

    # Testing right orthonormal form code
    AR,R,λ2 = _right_orthonormal_form(A,R0,1e-15);
    _check_right_canonical(A,AR,R;verbose=true);

    # Testing mixed orthonormal form code
    ALm,ARm,C,λ3 = _mixed_canonical_form(A,L0,R0,1e-15);
    _check_mixed_canonical(ALm,ARm,C;verbose=true);
    
end

