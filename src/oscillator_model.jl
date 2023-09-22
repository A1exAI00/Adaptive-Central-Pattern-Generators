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
# Rodas5P - for stiff systems
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
function Hopf_adaptive_network(du, u, p, t)
    xᵢ, yᵢ, ωᵢ, αᵢ = u[:,1], u[:,2], u[:,3], u[:,4]

    γ, μ, ε, η, N = p[1:3]
    ω_teach = p[6:N+4-1]
    A_teach = p[N+4:end]

    P_teach = sum([A_teach[i]*cos(ω_teach[i]*t) for i in 1:N])
    Q_learned = sum([αᵢ*xᵢ[i] for i in 1:N])
    Fₜ = P_teach - Q_learned

    for i in 0:N-1
        x, y, ω, α = xᵢ[i], yᵢ[i], ωᵢ[i], αᵢ[i]
        r = sqrt(x^2 + y^2); r_2 = r^2
        du[i+1] = γ*(μ-r_2)*x - ω*y + ε*Fₜ
        du[i+2] = γ*(μ-r_2)*y + ω*x
        du[i+3] = -ε*Fₜ*y/r
        du[i+4] = η*x*Fₜ
    end
end

########################################################################

function Hopf_adaptive_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(Hopf_adaptive, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)

    check_success && sol.retcode!=:Success && return NaN

    return sol
end