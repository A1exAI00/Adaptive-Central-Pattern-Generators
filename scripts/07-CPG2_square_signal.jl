#=
Обучение модели CPG №2 для прямоугольного периолического сигнала (меандра)
Результат:
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie, Interpolations

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 1000)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "07-CPG2_square"
PLOT_PX_PER_UNIT_PNG = 2

# Настройки вывода прогресса
print_elapsed_time = true

########################################################################

# Время интегрирования
t₀, t₁ = 0.0, 10_000.0
t_SPAN = [t₀, t₁]

# Постоянные параметры системы
γ, μ, ε, η, τ, N = 1.0, 1.0, 0.9, 0.5, 0.5, 9

# Интерпоряция меандра с шумом
Ω_square_wave, duty_cycle, A_square_wave, Nₜ, noise_aplitude  = 30.0, 0.5, 1.0, 5_000, 0.2
T_itp = 2π/Ω_square_wave
itp_param = (0.0, T_itp, Nₜ, T_itp, duty_cycle, A_square_wave, noise_aplitude)
P_teach_itp = generate_square_itp(itp_param)

system_param = (γ, μ, ε, η, τ, N, T_itp, P_teach_itp)

# Начальные условия системы
x₀, y₀, ϕ₀ = 1.0, 0.0, 0.0
x₀ = x₀*ones(N)
y₀ = y₀*ones(N)
ω₀ = range(Ω_square_wave+5, N*Ω_square_wave+5, N)
α₀ = 1.0*ones(N) #α₀ = range(1.0, 5.0, N)
ϕ₀ = ϕ₀*ones(N)
U₀ = SA[x₀..., y₀..., ω₀..., α₀..., ϕ₀...]

########################################################################

t_calculation_start = time_ns()

# Интегрирование системы
solution = CPG2_itp_integrate(U₀, t_SPAN, system_param; reltol=1e-3, abstol=1e-3)
t_sol = solution.t

elapsed_time = elapsed_time_string(time_ns()-t_calculation_start)
print_elapsed_time && println(elapsed_time)

########################################################################

# Выученный спектр
ω_learned = [mean_of_tail(solution[2N+i,:], 0.9) for i in 1:N]
α_learned = [mean_of_tail(solution[3N+i,:], 0.9) for i in 1:N]

# Аналитический спектр
ω_plot_range = range(0, maximum(ω_learned), 100)
α_plot_func(ω) = 2*A_square_wave*(-1)/(ω*π)*(cos(ω*π)-1)
α_plot = α_plot_func.(ω_plot_range/Ω_square_wave)

# Обучающий сигнал на периоде
t_plot_range = range(t₁-T_itp, t₁, Nₜ)
P_teach_plot = P_teach_itp.(mod.(t_plot_range, T_itp))

# Обученный сигнал на периоде
Q_learned = zeros(length(t_plot_range))
for (i,t) in enumerate(t_plot_range)
    for j in 1:N
        Q_learned[i] += solution(t)[0N+j]*solution(t)[3N+j]
    end
end

########################################################################

fig = Figure(resolution=PLOT_RES)
ax_ω = Axis(fig[1,1], 
    title="ω(t)",
    xlabel="t",
    ylabel="ω")
ax_α = Axis(fig[2,1], 
    title="α(t)",
    xlabel="t",
    ylabel="α")
ax_ϕ = Axis(fig[3,1], 
    title="ϕ(t)",
    xlabel="t",
    ylabel="ϕ")
ax_s_t = Axis(fig[4,1], 
    title="P_teach(t) & Q_learned(t)",
    xlabel="t",
    ylabel="signal")

hlines!(ax_ω, 0.0, color=:black)
hlines!(ax_α, 0.0, color=:black)
hlines!(ax_ϕ, 0.0, color=:black)
hlines!(ax_s_t, 0.0, color=:black)

for i in 1:N
    lines!(ax_ω, t_sol, solution[2N+i,:])
    lines!(ax_α, t_sol, solution[3N+i,:])
    lines!(ax_ϕ, t_sol, mod.(2π .* solution[4N+i,:], 2π))
end
lines!(ax_s_t, t_plot_range, P_teach_plot, color=:blue)
lines!(ax_s_t, t_plot_range, Q_learned, color=:red)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)