#=
Базовые функции, необходимые для интегрирования модели 
адаптивного осциллятора Хопфа (Стюарта-Ландау) и адаптивного 
Central Pattern Generator (CPG) из статьи
"From Dynamic Hebbian Learning of Oscillators to Adaptive Central Pattern Generators"
=#

########################################################################

#using Statistics: mean
using OrdinaryDiffEq, StaticArrays

########################################################################

const RELTOL::Float64, ABSTOL::Float64 = 1e-5, 1e-5
const MAXITERS::Int64 = Int(1e7)

# Different alg: 
# Rodas5P - for stiff systems; faild on automatic differentiation
# Tsit5 - 
const ALG = Tsit5

########################################################################

"""Model of an adaptive Hopf oscillator 

# Parameters

`u` : vector of state varisbles, `u=(x,y,ω)`

`p` : vector of parameters, `p=(γ, μ, ε, ω_teach)`

`t` : current time"""
function Hopf_adaptive(u, p, t)
    x, y, ω = u
    γ, μ, ε, ω_teach = p
    r = sqrt(x^2 + y^2); r_2 = r^2
    Fₜ = cos(ω_teach*t)
    return SA[
        γ*(μ-r_2)*x - ω*y + ε*Fₜ, 
        γ*(μ-r_2)*y + ω*x,
        -ε*Fₜ*y/r]
end


"""Model of an adaptive Central Pattern Generator, consisting of `N` oscillators

# Parameters

`du` : a matrix of difference of state veriables, size: `(N, 4)`

`u` : a matrix of state variables at `t`, size: `(N, 4)`

`p` : vector of parameters, `p = (γ, μ, ε, η, N, ω_teach..., A_teach...)`, size: `2N+5`
    
`t` : current time"""
function CPG_model(u, p, t)
    γ, μ, ε, η, N = p[1:5]
    ω_teach = p[6:N+5]
    A_teach = p[N+6:end]
    N = Int(N)

    xᵢ, yᵢ, ωᵢ, αᵢ = u[1:N], u[N+1:2*N], u[2*N+1:3*N], u[3*N+1:end]

    P_teach = sum(A_teach.*cos.(ω_teach.*t))
    Q_learned = sum(xᵢ.*αᵢ)
    Fₜ = P_teach - Q_learned

    du = zeros(4*N)

    for i in 1:N
        x, y, ω = xᵢ[i], yᵢ[i], ωᵢ[i]
        r² = x^2 + y^2; r = sqrt(r²)
        du[i] = γ*(μ-r²)*x - ω*y + ε*Fₜ
        du[N+i] = γ*(μ-r²)*y + ω*x
        du[2N+i] = -ε*Fₜ*y/r
        du[3N+i] = η*x*Fₜ
    end
    return SVector{4*N}(du)
end

########################################################################

function Hopf_adaptive_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(Hopf_adaptive, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)

    check_success && sol.retcode!=:Success && return NaN

    return sol
end

function CPG_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(CPG_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)

    check_success && sol.retcode!=:Success && return NaN

    return sol
end