# This code solves the differential equations that describe a long-term bond 
# equilibrium as described in Aguiar-Amador (2020)
#
# The code is in Julia. There is an additional Jupyter notebook that generates
# the figures used. 

using Parameters
using Roots
using DifferentialEquations
using Plots 


@with_kw struct ModelLB{F1, F2} @deftype Float64
    r = 0.05
    δ = 0.2
    λ = 0.2
    τₗ = 0.05
    τₕ = 0.3
    y = 1.0
    u::F1 = log
    inv_u::F2 = exp
    
    ρ = r
    v̲ = u((1 - τₕ) * y) / r
    v̅ = u((1 - τₗ) * y) / r
    b̲ = (y - inv_u(ρ * v̅)) / r
    q̲ = (r + δ) / (r + δ + λ)
    b̅ = (y - inv_u((ρ + λ)* v̲ - λ * v̅)) / (r + δ * (1 - q̲))
end


function cFun(p, q, m)
    @assert typeof(m.u) == typeof(log)
    return (- q / p)
end 


function pₛ(q, b, m) 
    return - (q / (m.y - b * (m.r +m. δ - q * m.δ)))
end


function c_stationary(q, b, m)
    return m.y - (m.r + m.δ * (1 - q)) * b
end


function u_stationary(q, b, m) 
    if b < m.b̲ 
        return m.u(m.y - (m.r + m.δ * (1 - q)) * b) / m.r
    else
        return ((m.u(m.y - (m.r + m.δ * (1 - q)) * b) + m.λ * m.v̅) /
             (m.r + m.λ))
    end
end


function bDot(c, q, b, m)
    return (c + (m.r + m.δ) * b - m.y) / q - m.δ * b
end


function hjb(v, c, p, q, b, m)
    return (
        m.u(c) + p * bDot(c, q, b, m) + m.λ * m.v̅ - 
        (m.ρ + m.λ) * v
    )
end


function v_prime(v, q, b, m)
    pss = pₛ(q, b, m)
    if hjb(v, cFun(pss, q, m), pss, q, b, m) >= 0.0 
        @info "OOOHHHH no! No solution to HJB. Should be stopping at" b
        # you don't want to be here
        return 0.0
    else
        return find_zero(
            p -> hjb(v, cFun(p, q, m), p, q, b, m), 
            (
                pss - 1000.0, 
                pss
            )
        )
    end
end


function q_prime(p, q, b, m) 
    return (
        ((m.r + m.δ + m.λ) * q - (m.r + m.δ)) / bDot(cFun(p, q, m), q, b, m)
    )
end


function ode_system!(du, u, m, t)
    # u = [ v(b)  q(b) ]
    # println(
    #     "b=", t, " [v, q]=", u, ", v - vss=", 
    #     u[1] - u_stationary(u[2], t)
    # )
    du[1] = v_prime(u[1], u[2], t, m)
    du[2] = q_prime(du[1], u[2], t, m)
end


function solve_equilibrium(
    m::ModelLB; 
    extra_grid_pts=20,
    tol_initial_point=10.0^(-6),
    tol_stopping_ODE=10.0^(-9)
)
    u0 = [m.v̅ + tol_initial_point, 1.0]   # initial condition
    bspan = (m.b̲, Inf) # range for b

    condition = function (u, t, integrator) 
        pss = pₛ(u[2], t, m)
        hjb(u[1], cFun(pss, u[2], m), pss, u[2], t, m) >= -tol_stopping_ODE 
        # stop if you hit the stationary boundary
    end

    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
    prob = ODEProblem(ode_system!, u0, bspan, m)

    # Solving the crisis zone
    out = solve(
        prob, 
        Rosenbrock32(autodiff=false), 
        reltol=1e-6,
        callback=cb
    )
    return collect_solution(m, out, extra_grid_pts=extra_grid_pts)
end


function collect_solution(m, out; extra_grid_pts=20)
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
            c[i] = cFun(v_prime(v[i], q[i], b, m), q[i], m)
            css[i] = c_stationary(q[i], b, m)
            vss[i] = u_stationary(q[i], b, m)
        else
            v[i] = u_stationary(m.q̲, b, m)
            q[i] = m.q̲
            c[i] = c_stationary(m.q̲, b, m)
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


function solve_efficient(m; extra_grid_pts=20)
    @unpack y, r, ρ, b̲, b̅, λ, v̅, v̲, q̲, δ = m 

    # Computing the exit level of consumption
    # first: solve HJB
    p_exit = find_zero(
        p -> (
            (r + λ) * v̅ - m.u(cFun(p, 1.0, m)) - p * 
                (cFun(p, 1.0, m) + (r + λ) * b̲ - y) - λ * v̅
        ),
        (
            -1000.0 - 1. / (y - r * b̲), 
            - 1. / (y - r * b̲)
        )
    )
    # Then get consumption and bI:
    c_exit = cFun(p_exit, 1.0, m) 
    bI = (y - c_exit) / ((r + λ) * q̲)

    # Solving Crisis Zone efficient ODE
    eff_ode_system! = function(du, u, m, t)
        du[1] = (((r + λ) * u[1] - m.u(c_exit) - λ * v̅) / 
            bDot(c_exit, u[2], t, m))
        du[2] = ((r + δ + λ) * u[2] - (r + δ)) / bDot(c_exit, u[2], t, m)
    end
    bspan = (b̲, bI)
    u0 = [v̅, 1.0] 
    prob = ODEProblem(eff_ode_system!, u0, bspan, m)
    out = solve(
        prob, 
        Rosenbrock32(autodiff=false), 
        reltol=1e-6
    )

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
            css[i] = c_stationary(q[i], b, m)
            vss[i] = u_stationary(q[i], b, m)
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


function do_c_plot(sol)
    f = plot(sol.b, sol.css, line=(1, :dash,), color=2,
        xlabel="b", ylabel="c"); 
    plot!(f, sol.b, sol.c, line=(2), color=1, legend=false)
    plot!(f, size=(300,200))
    vline!(f, [sol.b̲, sol.b̅, sol.bI], line=(1,  :gray))

    return f 
end


function do_q_plot(sol)
    f = plot(sol.b, sol.q, line=(2), color=1, legend=false, 
        xlabel="b", ylabel="q")
    hline!(f, [sol.m.q̲], line=(1, :dash, :gray))
    plot!(f, size=(300,200))
    vline!(f, [sol.b̲, sol.b̅, sol.bI], line=(1, :gray))
    return f
end


function do_v_plot(sol)
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