#=
Обучение модели CPG №1 для сигнала, состояшего из одной синусоидальной компоненты
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 1000)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "03-first_order_CPG_learning"
PLOT_PX_PER_UNIT_PNG = 2

# Настройки вывода прогресса
print_elapsed_time = true

########################################################################

# Постоянные параметры системы
γ, μ, ε, η, N = 1.0, 1.0, 0.9, 0.5, 1
Ω_teach = [10.0]; A_teach = [3.0]
system_param = SA[γ, μ, ε, η, N, Ω_teach..., A_teach...]

# Начальные условия системы
x₀, y₀ = 1.0, 0.0
ω₀ = [20.0]; α₀ = [1.0]
U₀ = SA[x₀, y₀, ω₀..., α₀...]

# Время интегрирования
t₀, t₁ = 0.0, 500.0
t_SPAN = [t₀, t₁]

# Проверка параметров и начальных условий
println("system_param = $system_param")
println("U₀ = $U₀")
@assert length(Ω_teach) == N "length of `Ω_teach` is not equal `N`"
@assert length(A_teach) == N "length of `A_teach` is not equal `N`"
@assert length(ω₀) == N "length of `ω₀` is not equal `N`"
@assert length(α₀) == N "length of `α₀` is not equal `N`"

########################################################################

t_calculation_start = time_ns()

# Интегрирование системы
solution = CPG_integrate(U₀, t_SPAN, system_param)
x_sol = solution[1,:]
y_sol = solution[2,:]
ω_sol = solution[3,:]
α_sol = solution[4,:]
t_sol = solution.t

elapsed_time = elapsed_time_string(time_ns()-t_calculation_start)
print_elapsed_time && println(elapsed_time)

########################################################################

fig = Figure(resolution=PLOT_RES)

ax_x = Axis(fig[1,1], 
    title="x(t)",
    xlabel="t",
    ylabel="x")
ax_y = Axis(fig[2,1], 
    title="y(t)",
    xlabel="t",
    ylabel="y")
ax_ω = Axis(fig[3,1], 
    title="ω(t)",
    xlabel="t",
    ylabel="ω")
ax_α = Axis(fig[4,1], 
    title="α(t)",
    xlabel="t",
    ylabel="α")

lines!(ax_x, t_sol, x_sol)
lines!(ax_y, t_sol, y_sol)
lines!(ax_ω, t_sol, ω_sol)
lines!(ax_α, t_sol, α_sol)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)