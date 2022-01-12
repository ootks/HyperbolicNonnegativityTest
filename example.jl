include("nonnegativity_test.jl")


# Uses the determinant to show that a sum-of-squares polynomial is nonnegative.
n = 4
@polyvar x[1:n, 1:n]
hyperbolic =
    1.0*MultivariatePolynomials.LinearAlgebra.det([x[min(i,j),max(i,j)] for i=1:n, j=1:n])
candidate = 5.0*(x[1,2]-x[1,3])^2+x[1,3]^2+x[1,4]^2
direction = [i == j ? 1. : 0. for i=1:n for j=1:i]
F = [convert(Polynomial{true, Float64}, (i > 1 || j == 1) ? 0 : x[i,j]) for j=1:n for i=1:j]
g = [convert(Polynomial{true, Float64}, i) for i=[0,0,1,0]]

println(nonnegativity_test(candidate, hyperbolic, direction, F, g))
