#=
Выполняет CPG обучение для многокомпонентного сигнала (N=3)
Результат: 
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Plot settings
PLOT_RES = (1000, 1000)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "higher_order_CPG_learning"
PLOT_PX_PER_UNIT_PNG = 2

print_elapsed_time = true



########################################################################

# Dynamical system parameters
γ, μ, ε, η, N = 1.0, 1.0, 0.9, 0.5, 3
ω_teach = [10.5, 20.5, 30.5]; A_teach = [3.5, 4.5, 5.5]
system_param = SA[γ, μ, ε, η, N, ω_teach..., A_teach...]

# Initial values
x₀, y₀ = 1.0, 0.0
x₀ = x₀*ones(N); y₀ = y₀*ones(N); ω₀ = range(10.0, 50.0, N); α₀ = range(1.0, 5.0, N)
U₀ = SA[x₀..., y₀..., ω₀..., α₀...]

# Time span
t₀, t₁ = 0.0, 500.0
t_SPAN = [t₀, t₁]

# Check parameters and initial values
println("s_param = $system_param")
println("U₀ = $U₀")
@assert length(ω_teach) == N "length of `ω_teach` is not equal `N`"
@assert length(A_teach) == N "length of `A_teach` is not equal `N`"
@assert length(x₀) == N "length of `x₀` is not equal `N`"
@assert length(y₀) == N "length of `y₀` is not equal `N`"
@assert length(ω₀) == N "length of `ω₀` is not equal `N`"
@assert length(α₀) == N "length of `α₀` is not equal `N`"

########################################################################

t_calculation_start = time_ns()

# Integration
solution = CPG_integrate(U₀, t_SPAN, system_param)
t_sol = solution.t

elapsed_time = elapsed_time_string(time_ns()-t_calculation_start)
print_elapsed_time && println(elapsed_time)

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

for i in 1:N
    lines!(ax_ω, t_sol, solution[2N+i,:])
    lines!(ax_α, t_sol, solution[3N+i,:])
end
hlines!(ax_ω, 0.0, color=:black)
hlines!(ax_α, 0.0, color=:black)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)