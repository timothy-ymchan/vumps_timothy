# Two-site VUMPS 

## Problem description
We want to find an algorithm that does the following:
### Input
- Hamiltonian of form 
$$H=\sum_n h_{n,n+1}$$
- uMPS of given bond dimensions $A \in \mathbb C^{\chi}\otimes \mathbb C^{phy} \otimes \mathbb C^{\chi'}$

### Ouput
- Approximate solution to the minimization problem 
$$\min_{A\in \text{NuMPS}} \frac{\braket{\psi(\bar A)|H|\psi(A)}}{\braket{\psi(\bar A)|\psi(A)}}$$
- where $\bar A$ simply denotes the conjugate of uMPS $A$ and the braket notation here refers to the inner product in thermodynamic limit 
- NuMPS denotes the set of all uMPS normalized in thermodynamic limit 

## Approach (Conceptual)
We can differentiate the equation directly (at least formally), and obtain the optimal condition. Importantly, this lead to the following condition
$$\bra{\partial_{\bar i} \psi(\bar A^*)} (H-E(\bar A^*,A^*))\ket{\psi(A^*)}$$
where $A^*$ denotes the optimal solution to the variational problem. Geometrically, if we imagine embedding our manifold of uMPS states into the larger, full Hilbert space, this condition is simply saying that the residue vector 

$$(H-E(\bar A^*,A^*))\ket{\psi(A^*)} \in T_p(A^*)^\perp$$

i.e. the residue is orthogonal to the tangent space at point $A^*$, which make sense because if it wasn't perpendicular, we can move a little bit along the tagent plane and get a smaller residue. It is also called a **Galerkin condition** (Actually I have not idea what that is, but the paper is using it, so I'll use it). Pictorially,

![galerkin condition](img/galerkin_condition.png)


## Approach (In detail)
Now we want to derive the equation associated with this optimization procedure and cast everything in terms of the uMPS parameters inputs $A_L, A_R, C$ 