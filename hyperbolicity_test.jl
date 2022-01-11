using Hypatia
using LinearAlgebra
using DynamicPolynomials

include("hyperbolicity.jl")
# The Motzkin Polynomial
#q = a^4*b^2+a^2*b^4+c^6-3.0*a^2*b^2*c^2

# The Bezoutian matrix of the hyperbolic polynomial 'hyperbolic' with
# respect to the directions 'direction' and 'u'
function bezoutian(hyperbolic::Polynomial{true, T}, direction::Vector{T},
                   u::Vector) where T <: Real
    # Could be confusing if t or s are already variables in hyperbolic
    # but the library seems to handle this fine.
    @polyvar t s
    shift = variables(hyperbolic) + t * direction
    px = hyperbolic(variables(hyperbolic) => shift)
    d_upx = u'differentiate(px, variables(hyperbolic))

    Bezoutian = div(px * subs(d_upx, t=>s) - d_upx  * subs(px, t=>s), t-s)

    deg = maxdegree(hyperbolic)
    BezoutianMatrix = zeros(Polynomial{true, T}, deg, deg)

    b_terms = terms(Bezoutian)
    for term in b_terms
        i = maxdegree(gcd(t^deg, term))
        j = maxdegree(gcd(s^deg, term))
        BezoutianMatrix[i+1,j+1] += div(term, t^i*s^j)
    end
    return BezoutianMatrix
end

# Produces linear constraints for checking that 
# constant is a constant polynomial
# parameterized is a 
# Both should be homogeneous
function equality(parameterized::Polynomial{true, T},
                     constant::Polynomial{true, T}, 
                     extra_variables::Vector{PolyVar{true}}) where T <: Real
    q_terms = terms(constant)
    monomials = [monomial(term) for term in q_terms]
    q_coefficients = [coefficient(term) for term in q_terms]

    p_terms = terms(parameterized)
    p_coefficients = zeros(Polynomial{true, T}, length(monomials))
    zero_monomials = []
    zero_coefficients = []

    for p_term in p_terms
        is_present = false
        for (i, monomial) in enumerate(monomials)
            if divides(monomial, p_term)
                p_coefficients[i] += div(p_term, monomial)
                is_present = true
                break
            end
        end
        if is_present
            continue
        end
        is_present = false
        for (i, monomial) in enumerate(zero_monomials)
            if divides(monomial, p_term)
                zero_coefficients[i] += div(p_term, monomial)
                is_present = true
                break
            end
        end
        if is_present
            continue
        end
        monomial = subs(p_term, extra_variables => [1 for v in extra_variables])
        push!(zero_monomials, monomial)
        push!(zero_coefficients, div(p_term, monomial))
    end
    return (q_coefficients, p_coefficients, zero_coefficients)
end
n = 4
@polyvar x[1:n, 1:n] u[1:n,1:n]
p = MultivariatePolynomials.LinearAlgebra.det([x[min(i,j),max(i,j)] for i=1:n, j=1:n])
q = x[1,2]^2+x[1,3]^2+x[1,4]^2

bez = bezoutian(p, [i == j ? 1 : 0 for i=1:n for j=1:i], [u[i,j] for i=1:n for j=1:i])[3,3]
zero_vars = [x[i,j] for i=1:n for j=1:i if i > 1 || j == 1]
bez = subs(bez, zero_vars => zeros(Int64, length(zero_vars)))
#display(bezoutian(p, [1,0,0,0], [u1,u2,u3,u4]))


(b, A, z) = equality(bez, q,[u[i,j] for i=1:n for j=1:i])

cones = [Hypatia.Cones.Hyperbolicity{Float64}(1.0*p, 1.0*[i == j ? 1 : 0 for i=1:n for j=1:i])]

A = vcat([differentiate(eq, [u[i,j] for i=1:n for j=1:i])' for eq in A]...)
A = vcat(A, [differentiate(eq, [u[i,j] for i=1:n for j=1:i])' for eq in z]...)
println(A)
b = vcat(b, zeros(Float64, length(z)))
println(b)


m = length([u[i,j] for i=1:n for j=1:i])
c = zeros(Float64, m)
G = -Matrix{Int}(I, m, m)
h = zeros(Float64, m)
model = Hypatia.Models.Model{Float64}(c, A, b, G, h, cones)
solver = Hypatia.Solvers.Solver{Float64}(verbose = true);
Hypatia.Solvers.load(solver, model)
Hypatia.Solvers.solve(solver)
print(Hypatia.Solvers.get_x(solver))

println(subs(bez, [u[i,j] for i=1:n for j=1:i]=>Hypatia.Solvers.get_x(solver)))
