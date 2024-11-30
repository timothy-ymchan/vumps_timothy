using TensorKit 

function A4_hamiltonian(λ,μ)
    V = ℂ^3 # Spin 1 system
    # Spin matrices 
    Sx = TensorMap(zeros,ComplexF64,V←V)
    Sy = TensorMap(zeros,ComplexF64,V←V)
    Sz = TensorMap(zeros,ComplexF64,V←V)
    eye = id(V)
    
    Sx.data .= ([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] ./ sqrt(2))
    Sy.data .= ([0.0 -1.0*im 0.0; 1.0*im 0 -1.0*im; 0.0 1.0*im 0.0]) ./ sqrt(2)
    Sz.data .= [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
    
    H0 = Sx⊗Sx + Sy⊗Sy + Sz⊗Sz # The Heisenberg term
    H1 = (Sx⊗Sx)^2 + (Sy⊗Sy)^2 + (Sz⊗Sz)^2 # The square terms
    H2 = (Sx*Sy)⊗Sz + (Sz*Sx)⊗Sy + (Sy*Sz)⊗Sx +
         (Sy*Sx)⊗Sz + (Sx*Sz)⊗Sy + (Sz*Sy)⊗Sx + 
         Sx⊗(Sy*Sz) + Sz⊗(Sx*Sy) + Sy⊗(Sz*Sx) + 
         Sx⊗(Sz*Sy) + Sz⊗(Sy*Sx) + Sy⊗(Sx*Sz) # The complicated term

    H = H0 + μ*H1 + λ*H2 # The A4 Hamiltonian 
    return H
end