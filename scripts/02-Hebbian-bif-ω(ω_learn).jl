#=
Бифуркационная диаграмма, показывающая различные устойчивые обученные 
частоты ω_learned при некотором значении обучающей частоты Ω_teach

Результат: линейная зависимость ω_learn = Ω_teach
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "02-ω(ω_learn)_bif_diagram"
PLOT_PX_PER_UNIT_PNG = 2

# Настройки вывода прогресса 
print_loop_progress = true
print_elapsed_time = true

########################################################################

# Постоянные параметры системы
γ, μ, ε = 1.0, 1.0, 0.9
Ω_teach_1, Ω_teach_2, Ω_teach_n = 1.0, 40.0, 100
Ω_teach_range = range(Ω_teach_1, Ω_teach_2, Ω_teach_n)

# Начальные условия системы
x₀, y₀ = 1.0, 0.0
ω₀_1, ω₀_2, ω₀_n = 1.0, 20.0, 5
ω₀_range = range(ω₀_1, ω₀_2, ω₀_n)

# Время интегрирования
t₀, t₁ = 0.0, 50_000.0
t_SPAN = [t₀, t₁]

########################################################################

ω_learned_array = Array{Float64}(undef,0)
Ω_teach_array = Array{Float64}(undef,0)

t_calculation_start = time_ns()

# Многократное интегрирование 
# Перебор по обучающим частотам Ω_teach
for (i,Ω_teach) in enumerate(Ω_teach_range)
    print_loop_progress && println("Ω_teach=$Ω_teach, $i/$Ω_teach_n")

    # Перебор по начальным условиям (частотам) ω₀
    for (j,ω₀) in enumerate(ω₀_range)
        global ω_learned_array, Ω_teach_array

        system_param = SA[γ, μ, ε, Ω_teach]
        U₀ = SA[x₀, y₀, ω₀]
        solution = Hopf_adaptive_integrate(U₀, t_SPAN, system_param)
        x_sol = solution[1,:]
        y_sol = solution[2,:]
        ω_sol = solution[3,:]
        t_sol = solution.t
    
        mean_learned_ω = mean_of_tail(ω_sol, 0.9)
    
        push!(ω_learned_array, mean_learned_ω)
        push!(Ω_teach_array, Ω_teach)
    end
    
end

elapsed_time = elapsed_time_string(time_ns()-t_calculation_start)
print_elapsed_time && println(elapsed_time)

########################################################################

fig = Figure(resolution=PLOT_RES)

ax = Axis(fig[1,1], 
    title="Bifurcation diagram, ω_learned(ω_teach)",
    xlabel="ω_teach",
    ylabel="ω_learned")

scatter!(ax, Ω_teach_array, ω_learned_array, markersize=2)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)
