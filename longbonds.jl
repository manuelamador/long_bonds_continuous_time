# This code solves the differential equations that describe a long-term bond 
# equilibrium as described in Aguiar-Amador (2020)
#
# The code is in Julia. There is an additional Jupyter notebook that generates
# the figures used. 

module LongBondsCT 

using Parameters
using Roots
using DifferentialEquations
using Plots 

export AbstractUtility, LogUtility, CRRAUtility, LongBondModel

export solve_efficient, solve_equilibrium, plotc, plotq, plotv


# ----------------  Utility functions 

# types and methods

abstract type AbstractUtility end 

# A concrete type of AbstractUtility must define four methods: 
#   utility, inverse utiliy, the first derivative of u and its inverse. 

# Log utility 
struct LogUtility <: AbstractUtility end 

(u::LogUtility)(c) = log(c)

inverse_utility(::LogUtility, x) = exp(x)

marginal_utility(::LogUtility, c) = 1 / c 

inverse_marginal_utility(::LogUtility, x) = 1 / x


# CRRA utility 
struct CRRAUtility{T<:Real} <: AbstractUtility 
    σ::T 
end 

(u::CRRAUtility)(c) = c^(1 - u.σ) / (1 - u.σ)

inverse_utility(u::CRRAUtility, x) = ((1 - u.σ) * x)^(1 / (1 - u.σ))

marginal_utility(u::CRRAUtility, c) = c^(- u.σ)

inverse_marginal_utility(u::CRRAUtility, x) = x^(- 1 / u.σ) 


# ----------------  Model struct

# Definining the main model struct that contains the parameters 
# and performs some basic computations.
# It takes as an input a concrete instance of AbstractUtility.
# The following sets up some defaults. 
@with_kw struct LongBondModel{T<:AbstractUtility, S<:Real}  @deftype S
    u::T = LogUtility()
    r = 0.05
    ρ = r
    δ = 0.2
    λ = 0.2
    τₗ = 0.05
    τₕ = 0.3
    y = 1.0    
    q̲ = (r + δ) / (r + δ + λ)
    v̲ = u((1 - τₕ) * y) / r
    v̅ = u((1 - τₗ) * y) / r
    b̲ = (y - inverse_utility(u, ρ * v̅)) / r
    b̅ = (y - inverse_utility(u, (ρ + λ)* v̲ - λ * v̅)) / (r + δ * (1 - q̲))
end

# The following allows to pass any parameter as say a BigFloat
# and all of the  other parameters will be promoted to BigFloats, 
# allowing for arbitrary precision computations.
#
# WARNING: This will stack-overflow if type promotion fails 
LongBondModel(uf, vargs...) = LongBondModel(uf, promote(vargs...)...) 


# ------------------ Model Methods 

cfoc(m::LongBondModel, p, q) = inverse_marginal_utility(m.u, - p / q) 

cₛₛ(m::LongBondModel, q, b) = m.y - (m.r + m.δ * (1 - q)) * b

pₛₛ(m::LongBondModel, q, b) = - marginal_utility(m.u, cₛₛ(m, q, b)) * q

bdot(m::LongBondModel, c, q, b) = (c + (m.r + m.δ) * b - m.y) / q - m.δ * b


function stationary_value(m::LongBondModel, q, b) 
    if b < m.b̲
        return m.u(cₛₛ(m, q, b)) / m.r
    else
        return (m.u(cₛₛ(m, q, b)) + m.λ * m.v̅) / (m.r + m.λ)
    end
end


function hjb(m::LongBondModel, v, c, p, q, b)
    return m.u(c) + p * bdot(m, c, q, b) + m.λ * m.v̅ - (m.ρ + m.λ) * v
end


function vprime(m::LongBondModel, v, q, b)
    pss = pₛₛ(m, q, b)
    cero = zero(v)
    if hjb(m, v, cfoc(m, pss, q), pss, q, b) >= cero 
        @info "OH NO! No solution to HJB. Should be stopping at" b
        return cero
    else
        return find_zero(
            p -> hjb(m, v, cfoc(m, p, q), p, q, b), (pss - 1000.0, pss)
        )
    end
end


function qprime(m::LongBondModel, p, q, b) 
    return (
        ((m.r + m.δ + m.λ) * q - (m.r + m.δ)) / bdot(m, cfoc(m, p, q), q, b)
    )
end


function ode_system!(duu, uu, m::LongBondModel, t)
    duu[1] = vprime(m, uu[1], uu[2], t)
    duu[2] = qprime(m, duu[1], uu[2], t)
end


function solve_equilibrium(
    m::LongBondModel; 
    extra_grid_pts=20,
    bv_tol=10.0^(-6),
    stop_tol=10.0^(-9),
    ode_tol=()
)
    u0 = [m.v̅ + bv_tol, 1.0]   # initial condition
    bspan = (m.b̲, Inf) # range for b

    condition = function (uu, t, integrator) 
        pss = pₛₛ(m, uu[2], t)
        hjb(m, uu[1], cfoc(m, pss, uu[2]), pss, uu[2], t) >= -stop_tol 
        # stop if you hit the stationary boundary
    end

    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
    prob = ODEProblem(ode_system!, u0, bspan, m)

    # Solving the crisis zone
    out = solve(
        prob, 
        Rosenbrock32(autodiff=false),
        callback=cb;
        ode_tol...
    )
    return collect_solution(m, out, extra_grid_pts=extra_grid_pts)
end


function collect_solution(m::LongBondModel, out; extra_grid_pts=20)
    # Creating the solution 

    bbar_i = findlast(
        (x) -> x >= m.v̲,
        out[1, :]
    )
    bI = out.t[bbar_i]
    bI_Q = bbar_i < size(out.t)[1] ? false : true

    bgrid = vcat(
        range(0.0, m.b̲, length=extra_grid_pts),
        out.t[1:bbar_i],
        bI_Q ? range(out.t[end], m.b̅, length=extra_grid_pts) : Float64[]
    )

    v = similar(bgrid)
    q = similar(bgrid)
    c = similar(bgrid)
    css = similar(bgrid)
    vss = similar(bgrid)

    for (i, b) in enumerate(bgrid)
        if b <= m.b̲
            v[i] = m.u(m.y - m.r * b) / m.ρ 
            c[i] = m.y - m.r * b
            q[i] = 1.0
            css[i] = c[i]
            vss[i] = v[i]
        elseif b <= bI
            v_and_q = out(b)
            v[i] = v_and_q[1]
            q[i] = v_and_q[2]
            c[i] = cfoc(m, vprime(m, v[i], q[i], b), q[i])
            css[i] = cₛₛ(m, q[i], b)
            vss[i] = stationary_value(m, q[i], b)
        else
            v[i] = stationary_value(m, m.q̲, b)
            q[i] = m.q̲
            c[i] = cₛₛ(m, m.q̲, b)
            css[i] = c[i]
            vss[i] = v[i]
        end
    end

    return (
        m=m,
        b=bgrid,
        v=v,
        q=q,
        c=c,
        vss=vss,
        css=css,
        b̲=m.b̲,
        b̅=bgrid[end],
        bI=bI,
        ode_sol=out
    )    
end


function solve_efficient(m::LongBondModel; 
    extra_grid_pts=20, 
    bI_tol=10^(-10),
    ode_tol=()    
)
    @unpack y, r, ρ, b̲, b̅, λ, v̅, v̲, q̲, δ = m 

    # Computing the exit level of consumption
    # first: solve HJB
    uno = oneunit(y)

    p_exit = find_zero(
        p -> (
            (r + λ) * v̅ - m.u(cfoc(m, p, uno)) - p * 
                (cfoc(m, p, uno) + (r + λ) * b̲ - y) - λ * v̅
        ),
        (
            - uno / (y - r * b̲) - 1000 * uno, 
            - uno / (y - r * b̲)
        )
    )

    # Then get consumption and bI:
    c_exit = cfoc(m, p_exit, uno) 
    bI = (y - c_exit) / ((r + λ) * q̲) - bI_tol

    # Solving Crisis Zone efficient ODE
    eff_ode_system! = function(duu, uu, m, t)
        bdot_ = bdot(m, c_exit, uu[2], t)
        duu[1] = (((r + λ) * uu[1] - m.u(c_exit) - λ * v̅) / 
            bdot_)
        duu[2] = ((r + δ + λ) * uu[2] - (r + δ)) / bdot_
    end
    
    bspan = (b̲, bI)
    u0 = [v̅, uno] 
    prob = ODEProblem(eff_ode_system!, u0, bspan, m)
    out = solve(prob, Rosenbrock32(autodiff=false); ode_tol...)

    # Adjust grid for bgrid to be consistent with maximum outside option
    bbar_i = findlast(
        (x) -> x >= v̲,
        out[1, :]
    )
    bgrid = vcat(
        range(0.0, b̲, length=extra_grid_pts),
        out.t[1:bbar_i],
        bI < b̅ ? range(bI, b̅, length=extra_grid_pts) : Float64[]
    )

    # Collect the efficient solution 
    v = similar(bgrid)
    q = similar(bgrid)
    c = similar(bgrid)
    css = similar(bgrid)
    vss = similar(bgrid)

    for (i, b) in enumerate(bgrid)
        if b <= b̲ 
            q[i] = 1.0
            c[i] = y - r * b 
            v[i] = m.u(y - r * b) / ρ
            vss[i] = v[i]
            css[i] = c[i]
        elseif b <= bI
            v_and_q = out(b)
            v[i] = v_and_q[1]
            q[i] = v_and_q[2]
            c[i] = c_exit 
            css[i] = cₛₛ(m, q[i], b)
            vss[i] = stationary_value(m, q[i], b)
        else
            q[i] = q̲
            c[i] = y - (r + λ) * q̲ * b
            v[i] = (m.u(y - (r + λ) * q̲ * b) + λ * v̅) / (r + λ)
            css[i] = c[i]
            vss[i] = v[i]
        end
    end

    return (
        m=m,
        b=bgrid,
        v=v,
        q=q,
        c=c,
        vss=vss,
        css=css,
        b̲=m.b̲,
        b̅=bgrid[end],
        bI=bI,
        ode_sol=out
    )    
end


# -------------------  Plotting methods 

function plotc(sol)
    f = plot(sol.b, sol.css, line=(1, :dash,), color=2,
        xlabel="b", ylabel="c"); 
    plot!(f, sol.b, sol.c, line=(2), color=1, legend=false)
    plot!(f, size=(300,200))
    vline!(f, [sol.b̲, sol.b̅, sol.bI], line=(1,  :gray))
    return f 
end


function plotq(sol)
    f = plot(sol.b, sol.q, line=(2), color=1, legend=false, 
        xlabel="b", ylabel="q")
    hline!(f, [sol.m.q̲], line=(1, :dash, :gray))
    plot!(f, size=(300,200))
    vline!(f, [sol.b̲, sol.b̅, sol.bI], line=(1, :gray))
    return f
end


function plotv(sol)
    f = plot(sol.b, sol.vss, line=(2, :dash), color=2,
        xlabel="b", ylabel="v")
    plot!(f, size=(300,200))
    vline!(f, [sol.b̲, sol.b̅, sol.bI], line=(1,  :gray))
    for v in [sol.m.v̅, sol.m.v̲]
        plot!(f, sol.b, [v for b in sol.b], 
            line=(1, :dash, :gray))
    end
    plot!(f, sol.b, sol.v, line=(2), color=1, legend=false)
    return f 
end

end 