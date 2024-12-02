using TensorKit 
using TensorOperations

# Algorithm 1 of PhysRevB.97.045145
function effective_hamiltonian_two_site(h::AbstractTensorMap, AL::AbstractTensorMap, AR::AbstractTensorMap, L::AbstractTensorMap, R::AbstractTensorMap,ϵs::Real)
    """
        effective_hamiltonian(h::AbstractTensorMap, AL::AbstractTensorMap, AR::AbstractTensorMap, L::AbstractTensorMap, R::AbstractTensorMap,ϵs::Real)
    
    Compute effective hamiltonian tensors (h,AL,AR,HL,HR) from two_site hamiltonian h

    ## Parameters 
    `h::AbstractTensorMap`: Two site hamiltonian. Leg conventions (H_phy,H_phy)<- (H_phy,H_phy)
    `ϵs::Real`: Numerical precision for the hamiltonian solved 

    ## Returns
    `H_eff_terms`: A 5-tuple containing tensors (h,AL,AR,HL,HR) in order

    ## Remarks
    As discussed in PhysRevB.97.045145 we can in principle compute `H_AC` and `H_C` formally by inverting a power series. But this is numerically unstable. Instead, we will use the consistency conditions and solve for them iteratively. This introduces the error toloerance `ϵs`.

    """
    # hL is defined to be (Eq. 12)
    #   ->-         ->- (AL ) - (AL )->-
    #  |            |     ↓      ↓
    # (hL)    =     |   (    h     )
    #  |            |     ↓      ↓ 
    #   ->-         ->- (AL') - (AL')->-
    # Again that the index conventions for TensorKit is (codomain) ← (domain), that's why the indices of h looks so weird

    @tensor hL[i,j] := AL[a,m,b]*AL[b,n,i]*h[o,p,m,n]*conj(AL[a,o,c])*conj(AL[c,p,j]) 
    @tensor hR[i,j] := AR[i,m,b]*AR[b,n,a]*h[o,p,m,n]*conj(AR[j,o,c])*conj(AR[c,p,a])

    # Permute legs to fit the leg convention (i.e. arrows flows from left to right)
    hL = permute(hl,(1,2),())
    hR = permute(hR,(),(1,2))

    # Solving the system of equations for (HL| and |HR) (i.e. Eq. 15 of the paper)
    # (HL| [1 - TL + |R)(1|] = (hL| - (hL|R) (1|
    # [1 - TR + |1)(L|] |HR) = |hR) - |1)(L|hR)
    # We will use the method outlined in Appendix D
end
