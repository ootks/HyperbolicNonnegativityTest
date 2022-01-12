using Hypatia
using LinearAlgebra
using DynamicPolynomials

include("hyperbolicity.jl")
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

# Returns the conditions on the extra variables that make general specialize to
# specific.
# 'specific is some fixed polynomial
# 'general' is a polynomial with more variables than specific
# 'extra_variables' are the variables that appear in general but not specific
function equality(general::Polynomial{true, T},
                     specific::Polynomial{true, T}, 
                     extra_variables::Vector{PolyVar{true}}) where T <: Real
    q_terms = terms(specific)
    monomials = [monomial(term) for term in q_terms]
    q_coefficients = [coefficient(term) for term in q_terms]

    p_terms = terms(general)
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

# Returns matrices that represeent linear constraints that general specializes 
# to specific.
# 'specific is some fixed polynomial
# 'general' is a polynomial with more variables than specific
# 'extra_variables' are the variables that appear in general but not specific
function linearized_equality(general::Polynomial{true, T},
                     specific::Polynomial{true, T}, 
                     extra_variables::Vector{PolyVar{true}}) where T <: Real
    (b, A, z) = equality(general, specific, extra_variables)

    A = vcat([differentiate(eq, extra_variables)' for eq in A]...)
    A = vcat(A,
             [differentiate(eq, extra_variables)' for eq in z]...)
    b = vcat(b, zeros(T, length(z)))
    return (A, b)
end

# Uses the Bezoutian formalism found in https://arxiv.org/abs/1904.00491
# to try to certify that `candidate' is globally nonnegative using the
# hyperbolic polynomial `hyperbolic' and the auxilliary data F and g.
function nonnegativity_test(candidate::Polynomial{true, T},
                            hyperbolic::Polynomial{true, T},
                            direction::Vector{T},
                            F::Vector{Polynomial{true, T}},
                            g::Vector{Polynomial{true, T}}) where T <: Real
    cones = [Hypatia.Cones.Hyperbolicity{T}(hyperbolic, direction)]
    m = length(direction)

    @polyvar u[1:length(direction)]
    bez = bezoutian(hyperbolic, direction, u)
    bez = subs(bez, variables(hyperbolic) => F)
    bez = g'*bez*g
    println(bez)

    A, b = linearized_equality(bez, candidate, [u[i] for i in 1:length(direction)])
    c = zeros(T, m)
    G = -Matrix{T}(I, m, m)
    h = zeros(T, m)
    model = Hypatia.Models.Model{T}(c, A, b, G, h, cones)
    solver = Hypatia.Solvers.Solver{T}(verbose = true);
    Hypatia.Solvers.load(solver, model)
    Hypatia.Solvers.solve(solver)
    return solver.status == Hypatia.Solvers.Optimal
end
