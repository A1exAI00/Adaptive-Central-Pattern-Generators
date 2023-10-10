#=
Обучение Хеббиана адаптивного осциллятора Хопфа.
Модель обучается на частоту внешнего сигнала
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "01-ω_learning"
PLOT_PX_PER_UNIT_PNG = 2

########################################################################

# Постоянные параметры системы
γ, μ, ε, Ω_teach = 1.0, 1.0, 0.9, 30
system_param = SA[γ, μ, ε, Ω_teach]

# Начальные условия системы
x₀, y₀, ω₀ = 1.0, 0.0, 40.0
U₀ = SA[x₀, y₀, ω₀]

# Время интегрирования
t₀, t₁ = 0.0, 1200.0
t_SPAN = [t₀, t₁]

########################################################################

# Интегрирование системы
solution = Hopf_adaptive_integrate(U₀, t_SPAN, system_param)
x_sol = solution[1,:]
y_sol = solution[2,:]
ω_sol = solution[3,:]
t_sol = solution.t

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

lines!(ax_x, t_sol, x_sol)
lines!(ax_y, t_sol, y_sol)
lines!(ax_ω, t_sol, ω_sol)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)
