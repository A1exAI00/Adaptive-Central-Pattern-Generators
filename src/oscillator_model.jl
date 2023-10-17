#=
Базовые функции, необходимые для интегрирования модели 
адаптивного осциллятора Хопфа (Стюарта-Ландау) и адаптивного 
Central Pattern Generator (CPG) из статьи
"From Dynamic Hebbian Learning of Oscillators to Adaptive Central Pattern Generators"
=#

########################################################################

#using Statistics: mean
using OrdinaryDiffEq, StaticArrays, Interpolations
include("../src/misc_tools.jl")

########################################################################

const RELTOL::Float64, ABSTOL::Float64 = 1e-5, 1e-5 # 1e-5, 1e-5
const MAXITERS::Int64 = Int(1e7)

# Different alg: 
# Rodas5P - for stiff systems; faild on automatic differentiation
# Tsit5 - 
const ALG = Tsit5

########################################################################

"""
    Hopf_adaptive(u, p, t)

Модель адаптивного осциллятора Хопфа 

# Параметры

`u` : вектор фазовых переменных в момент времени `t` \\
`u=(x, y, ω)`

`p` : вектор параметров системы \\
`p=(γ, μ, ε, Ω_teach)`

`t` : данный момент времени
"""
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

"""
    CPG1_model(u, p, t)

Модель №1 адаптивного CPG, состоящая из `N` осцилляторов \\
Модель обучается на частоту и амплитуду внешнего гармонического сигнала,
заданного массивами `Ω_teach`, `A_teach`

# Параметры

`u` : вектор фазовых переменных в момент времени `t` \\
`u=(x..., y..., ω..., α...)`, size: `(N, 4)`

`p` : вектор параметров системы \\
`p = (γ, μ, ε, η, N, Ω_teach..., A_teach...)`, size: `2N+5`
    
`t` : данный момент времени
"""
function CPG1_model(u, p, t)
    γ, μ, ε, η, N = p[1:5]
    N = Int(N)
    Ω_teach = p[6:N+5]
    A_teach = p[N+6:end]

    xᵢ, yᵢ, ωᵢ, αᵢ = u[1:N], u[N+1:2*N], u[2*N+1:3*N], u[3*N+1:end]

    P_teach = sum(A_teach.*cos.(Ω_teach.*t))
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

"""
    CPG1_itp_model(u, p, t)

Модель №1 адаптивного CPG, состоящая из `N` осцилляторов \\
Модель обучается на частоту и амплитуду произвольного внешнего периодического сигнала, 
заданного функцией `P_teach_itp`

# Параметры

`u` : вектор фазовых переменных в момент времени `t` \\
`u=(x..., y..., ω..., α...)`, size: `(N, 4)`

`p` : вектор параметров системы \\
`p = (γ, μ, ε, η, N, T_itp, P_teach_itp)` \\
    `T_itp` - период интерполяции; `P_teach_itp` - интерполяционная функции
    
`t` : данный момент времени
"""
function CPG1_itp_model(u, p, t)
    γ, μ, ε, η, N, T_itp, P_teach_itp = p[1:end]
    N = Int(N)
    #Ω_teach = p[6:N+5]
    #A_teach = p[N+6:end]

    xᵢ, yᵢ, ωᵢ, αᵢ = u[1:N], u[N+1:2*N], u[2*N+1:3*N], u[3*N+1:end]

    #P_teach = sum(A_teach.*cos.(Ω_teach.*t))
    P_teach = P_teach_itp(mod(t,T_itp))
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

"""
    CPG2_model(u, p, t)

Модель №2 адаптивного CPG, состоящая из `N` осцилляторов

Модель обучается на частоту и амплитуду внешнего гармонического сигнала,
заданного массивами `Ω_teach`, `A_teach`, а так же на фазовые 
соотношения компонент сигнала по отношению к компоненте с наименьшей частотой

# Параметры

`u` : вектор фазовых переменных в момент времени `t` \\
`u=(x..., y..., ω..., α..., ϕ...)` \\
size: `(N, 5)`

`p` : вектор параметров системы \\
`p = (γ, μ, ε, η, τ, N, Ω_teach..., A_teach..., Φ_teach...)` \\
size: `3N+6`
    
`t` : данный момент времени
"""
function CPG2_model(u, p, t)
    γ, μ, ε, η, τ, N = p[1:6]
    N = Int(N)
    Ω_teach = p[7:1N+6]
    A_teach = p[1N+7:2N+6]
    Φ_teach = p[2N+7:end]

    xᵢ, yᵢ, ωᵢ, αᵢ, ϕᵢ = u[0N+1:1N], u[1N+1:2N], u[2N+1:3N], u[3N+1:4N], u[4N+1:end]

    P_teach = sum(A_teach.*cos.(Ω_teach.*t .+ Φ_teach))
    Q_learned = sum(xᵢ.*αᵢ)
    Fₜ = P_teach - Q_learned
    R₁ = sign(xᵢ[1])*acos(-yᵢ[1]/sqrt((xᵢ[1])^2 + (yᵢ[1])^2))

    du = zeros(5N)
    for i in 1:N
        x, y, ω, ϕ = xᵢ[i], yᵢ[i], ωᵢ[i], ϕᵢ[i]
        r² = x^2 + y^2; r = sqrt(r²)
        R = ω/ωᵢ[1]*R₁

        du[0N+i] = γ*(μ-r²)*x - ω*y + ε*Fₜ + τ*sin(R-ϕ)
        du[1N+i] = γ*(μ-r²)*y + ω*x
        du[2N+i] = -ε*Fₜ*y/r 
        du[3N+i] = η*x*Fₜ
        du[4N+i] = (i==1) ? 0 : sin(R-sign(x)*acos(-y/r)-ϕ)
    end
    return SVector{5N}(du)
end

"""
    CPG2_itp_model(u, p, t)

Модель №2 адаптивного CPG, состоящая из `N` осцилляторов

Модель обучается на частоту и амплитуду произвольного внешнего гармонического сигнала, 
зазаданного функцией `P_teach_itp`, а так же на фазовые 
соотношения компонент сигнала по отношению к компоненте с наименьшей частотой

# Параметры

`u` : вектор фазовых переменных в момент времени `t` \\
`u=(x..., y..., ω..., α..., ϕ...)`, size: `(N, 5)`

`p` : вектор параметров системы \\
`p = (γ, μ, ε, η, τ, N, T_itp, P_teach_itp)`, size: `8` \\
    `T_itp` - период интерполяции; `P_teach_itp` - интерполяционная функции
    
`t` : данный момент времени
"""
function CPG2_itp_model(u, p, t)
    γ, μ, ε, η, τ, N, T_itp, P_teach_itp = p
    N = Int(N)

    xᵢ, yᵢ, ωᵢ, αᵢ, ϕᵢ = u[0N+1:1N], u[1N+1:2N], u[2N+1:3N], u[3N+1:4N], u[4N+1:end]

    Q_learned = sum(xᵢ.*αᵢ)
    P_teach = P_teach_itp(mod(t, T_itp))
    Fₜ = P_teach - Q_learned
    R₁ = sign(xᵢ[1])*acos(-yᵢ[1]/sqrt((xᵢ[1])^2 + (yᵢ[1])^2))

    du = zeros(5N)
    for i in 1:N
        x, y, ω, ϕ = xᵢ[i], yᵢ[i], ωᵢ[i], ϕᵢ[i]
        r² = x^2 + y^2; r = sqrt(r²)
        R = ω/ωᵢ[1]*R₁

        du[0N+i] = γ*(μ-r²)*x - ω*y + ε*Fₜ + τ*sin(R-ϕ)
        du[1N+i] = γ*(μ-r²)*y + ω*x
        du[2N+i] = -ε*Fₜ*y/r 
        du[3N+i] = η*x*Fₜ
        du[4N+i] = (i==1) ? 0 : sin(R-sign(x)*acos(-y/r)-ϕ)
    end
    return SVector{5N}(du)
end

########################################################################

function Hopf_adaptive_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(Hopf_adaptive, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function CPG1_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(CPG1_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function CPG1_itp_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(CPG1_itp_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function CPG2_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(CPG2_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function CPG2_itp_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(CPG2_itp_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end