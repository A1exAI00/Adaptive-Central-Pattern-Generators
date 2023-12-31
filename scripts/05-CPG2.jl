#=
Обучение модели CPG №2 для сигнала, состояшего из N синусоидальных компонент
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 1000)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "05-super_CPG_learning"
PLOT_PX_PER_UNIT_PNG = 2

# Настройки вывода прогресса
print_elapsed_time = true

########################################################################

# Постоянные параметры системы
γ, μ, ε, η, τ, N = 1.0, 1.0, 0.9, 0.5, 0.5, 4
Ω_teach = [15.0, 30.0, 45.0, 60.0]
A_teach = [0.8, 1.0,-1.4,-0.5]
Φ_teach = [0.0, 0.0, 0.0, 0.0]
system_param = SA[γ, μ, ε, η, τ, N, Ω_teach..., A_teach..., Φ_teach...]

# Начальные условия системы
x₀, y₀, ϕ₀ = 1.0, 0.0, 0.0
x₀ = x₀*ones(N)
y₀ = y₀*ones(N)
ω₀ = range(6.0, 70.0, N)
α₀ = 1.0*ones(N) #α₀ = range(1.0, 5.0, N)
ϕ₀ = ϕ₀*ones(N)
U₀ = SA[x₀..., y₀..., ω₀..., α₀..., ϕ₀...]

# Время интегрирования
t₀, t₁ = 0.0, 1_500.0
t_SPAN = [t₀, t₁]

# Проверка параметров и начальных условий
println("s_param = $system_param")
println("U₀ = $U₀")
@assert length(Ω_teach) == N "length of `Ω_teach` is not equal `N`"
@assert length(A_teach) == N "length of `A_teach` is not equal `N`"
@assert length(Φ_teach) == N "length of `Φ_teach` is not equal `N`"
@assert length(x₀) == N "length of `x₀` is not equal `N`"
@assert length(y₀) == N "length of `y₀` is not equal `N`"
@assert length(ω₀) == N "length of `ω₀` is not equal `N`"
@assert length(α₀) == N "length of `α₀` is not equal `N`"
@assert length(ϕ₀) == N "length of `ϕ₀` is not equal `N`"

########################################################################

t_calculation_start = time_ns()

# Интегрирование системы
solution = super_CPG_integrate(U₀, t_SPAN, system_param)
t_sol = solution.t

elapsed_time = elapsed_time_string(time_ns()-t_calculation_start)
print_elapsed_time && println(elapsed_time)

########################################################################

# Создание обучающего сигнала для сравнения, как обучилась система
P_teach = [sum(A_teach.*cos.(Ω_teach.*t .+ Φ_teach)) for (i,t) in enumerate(t_sol)]
#P_teach = zeros(length(t_sol))
#for (i,t) in enumerate(t_sol)
#    P_teach[i] = sum(A_teach.*cos.(Ω_teach.*t .+ Φ_teach))
#end

# Восстановление сигнала из обученной системы 
# Q ⃗_learned = ∑_{i=1}^{N+1} x ⃗_i ⋅ α ⃗_i
Q_learned = zeros(length(t_sol))
for i in 1:N
    Q_learned .+= solution[0N+i,:].*solution[3N+i,:]
end

########################################################################

fig = Figure(resolution=PLOT_RES)

ax_error = Axis(fig[1,1], 
    title="error(t)",
    yscale=log10,
    xlabel="t",
    ylabel="error")
ax_ω = Axis(fig[2,1], 
    title="ω(t)",
    xlabel="t",
    ylabel="ω")
ax_α = Axis(fig[3,1], 
    title="α(t)",
    xlabel="t",
    ylabel="α")
ax_ϕ = Axis(fig[4,1], 
    title="ϕ(t)",
    xlabel="t",
    ylabel="ϕ")

hlines!(ax_error, 0.0, color=:black)
hlines!(ax_ω, 0.0, color=:black)
hlines!(ax_α, 0.0, color=:black)
hlines!(ax_ϕ, 0.0, color=:black)

lines!(ax_error, t_sol, abs.(P_teach-Q_learned))

for i in 1:N
    lines!(ax_ω, t_sol, solution[2N+i,:])
    lines!(ax_α, t_sol, solution[3N+i,:])
    lines!(ax_ϕ, t_sol, mod.(2π .* solution[4N+i,:], 2π))
end

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)