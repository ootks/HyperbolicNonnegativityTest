using LinearAlgebra
using Hypatia
using MultivariatePolynomials
mutable struct Hyperbolicity{T <: Real} <: Hypatia.Cones.Cone{T}
    polynomial::AbstractPolynomial{T}
    direction::Vector{T}

    dim::Int64
    nu::Real
    point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact_updated::Bool
    hess_fact::Factorization{T}

    symb_grad::Vector{AbstractPolynomial{T}}

    dual_point::Vector{T}
    use_hess_prod_slow::Bool
    use_hess_prod_slow_updated::Bool

    poly_derivatives::Vector{AbstractPolynomial{T}}

    function Hyperbolicity{T}(
          polynomial::AbstractPolynomial{T},
          direction::Vector{T}) where T <: Real
        @assert nvariables(polynomial) == length(direction)
        @assert polynomial(variables(polynomial) => direction) > 0

        
        cone = new{T}()
        cone.direction = direction
        cone.polynomial = polynomial
 
        cone.dim = nvariables(polynomial)
        return cone
    end
end

use_dual_barrier(::Hyperbolicity) = false

use_sqrt_hess_oracles(::Int, ::Hyperbolicity) = false

get_nu(cone::Hyperbolicity) = maxdegree(cone.polynomial)

function setup_extra_data!(cone::Hyperbolicity{T}) where T <: Real
    deg = maxdegree(cone.polynomial)
    cone.poly_derivatives = Array{AbstractPolynomial{T},1}(undef, deg)
    q = cone.polynomial
    for i = 1:deg
        cone.poly_derivatives[i] = q
        q = cone.direction'differentiate(q, variables(q))
    end
end

function set_initial_point!(arr::AbstractVector, cone::Hyperbolicity) 
    for i = 1:length(arr)
        arr[i] = cone.direction[i]
    end
end

function update_feas(cone::Hyperbolicity{T})::Bool where T <: Real
    @assert !cone.feas_updated
    cone.is_feas = all(map((x) -> x(cone.point) > eps(T),
                           cone.poly_derivatives))
    cone.feas_updated = true
    return cone.is_feas
end


function update_grad(cone::Hyperbolicity{T}) where T <: Real
    @assert cone.is_feas

    p = cone.polynomial
    cone.symb_grad = differentiate(p, variables(p))
    cone.grad = -map((x)->x(variables(p) => cone.point), cone.symb_grad)
    cone.grad /= p(variables(p)=>cone.point)

    cone.grad_updated = true

    return cone.grad
end

function update_hess(cone::Hyperbolicity{T}) where T <: Real
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)

    H = cone.hess.data
    mul!(H, cone.grad, cone.grad')

    p = cone.polynomial
    pval = p(variables(p) => cone.point)

    @inbounds for i in 1:nvariables(p), j in i:nvariables(p)
        ijder = differentiate(cone.symb_grad[i], variables(p)[j])
        H[i,j] -= ijder(variables(p) => cone.point) / pval
    end
    cone.hess_updated = true
    return cone.hess
end

use_dder3(cone::Hyperbolicity)::Bool = false
