using TensorKit

function AKLT_state(χ::Int)
    if χ < 2
        throw("χ=$χ is too small, you need at least χ=2 to store AKLT states faithfully")
    end
    d = 3 
    Hib_phy = ℂ^d # Physical hilbert space 
    Hib_χ = ℂ^χ # Bond hilbert space
    M= zeros(ComplexF64,Hib_χ⊗Hib_phy⊗Hib_χ')
    
    # Spin -1
    M[1,1,1] = M[2,1,1] = M[2,1,2] = 0
    M[1,1,2] = -sqrt(2.0/3.0)

    # Spin 0
    M[1,2,1] = sqrt(1.0/3.0)
    M[2,2,2] = -sqrt(1.0/3.0)
    M[1,2,2] = M[2,2,1] = 0

    # Spin 1
    M[1,3,1] = M[1,3,2] = M[2,3,2] = 0
    M[2,3,1] = sqrt(2.0/3.0)
    
    return M
end