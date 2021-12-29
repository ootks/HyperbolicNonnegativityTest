using DynamicPolynomials
using Hypatia

"""
$(TYPEDEF)
Hyperbolicity cone of dimension `dim` for the polynomial `p` in the direction `v`.
    $(FUNCTIONNAME){T}(dim::Int, p::MultivariatePolynomial, v::)
"""
mutable struct Hyperbolicity{T <: Real} <: Cone{T}
    polynomial::Polynomial{True, T}
    direction::Vector{T}

    point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}\
    inv_hess::Symmetric{T, Matrix{T}}\

    poly_derivatives::Vector{Polynomial{True, T}}

    function Hyperbolicity{T}(
          polynomial::Polynomial{True, T},
          direction::Vector) where T <: Real
        @assert nvariables(polynomial) == length(direction)
        @assert p(variables(p) => direction) > 0
        
        cone = new{T}()
        cone.direction = direction
        cone.polynomial = polynomial
        return cone
    end
end

function setup_extra_data!(cone::Hyperbolicity) where T <: Real
    deg = maxdegree(cone.polynomial)
    cone.poly_derivatives = Array(undef, deg)
    q = polynomial
    for i = 1:deg
        poly_derivatives[i] = q
        q = direction'differentiate(q, variables(q))
    end
end

set_initial_point!(arr::AbstractVector, cone::Hyperbolicity) =
    (arr = cone.direction)

function update_feas(cone::Hyperbolicity{T})::Bool where T <: Real
    @assert !cone.feas_updated
    cone.is_feas = all(map((x) -> x(cone.point) > 0, cone.poly_derivatives))
    cone.feas_updated = true
    return cone.is_feas
end


function update_grad(cone::Hyperbolicity{T})::Bool where T <: Real
    @assert cone.is_feasible

    cone.grad = map((x)->x(variables(p) => cone.point),
                    differentiate(p, variables(p)))
    cone.grad /= p(variables(p)=>cone.point)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Nonnegative{T}) where T <: Real
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    p = cone.polynomial
    pval = p(variables(p) => cone.point)
    @inbounds for i in 1:nvariables(p), j in i:nvariables(p)
        ider = differentiate(p, variables(p)[i])
        jder = differentiate(p, variables(p)[j])
        ijder = differentiate(ider, variables(p)[j])
        H[i,j] =
            ider(variables(p) => cone.point) * der(variables(p) => cone.point) -
            - pval => cone.point) * ijder(variables(p) => cone.point)
    end
    H /= pval^2
    cone.hess_updated = true
    return cone.hess
end

