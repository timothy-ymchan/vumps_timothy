# Notes 

## Arnoldi method
Reference: Wikipedia; The arnoldi algorithm(https://acme.byu.edu/00000179-aa18-d402-af7f-abf806ad0000/arnoldi2020-pdf
- Find approximate eigenvalue and eigenvector by constructing orthonormal basis of the Krylov subspace
- Basically just GM orthonormalization procedure
- Intuitively we expect the Krylov subspace to be a good approximation for eigenvectors with large eigenvalues
- To see why, assume our operator is unitarily diagonalizable, then 
$$A = \sum_{i} \lambda_i \ket{i}\bra{i}$$
- where we have ordered our eigenvectors such that $|\lambda_i|\geq |\lambda_j|$ for all $i>j$
- Then $A^n\ket{b}$ will be dominated by the eigenvector with the largest eigenvalue, and the Krylov subspace will very roughly be the subspace spanned by the first $n$ largest eigenvalues