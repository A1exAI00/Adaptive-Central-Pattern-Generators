#=
Shows the bifurcation diagram of learned freq against incoming freq
Result: A linear function ω_learn = ω_teach
=#

include("../src/oscillator_model.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Plot settings
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "ω(ω_learn)_bif_diagram"
PLOT_PX_PER_UNIT_PNG = 2

print_loop_progress = true
print_elapsed_time = true

########################################################################

# Dynamical system parameters
γ, μ, ε = 1.0, 1.0, 0.9
ω_teach_1, ω_teach_2, ω_teach_n = 1.0, 40.0, 100
ω_teach_range = range(ω_teach_1, ω_teach_2, ω_teach_n)

# Initial values
x₀, y₀ = 1.0, 0.0
ω₀_1, ω₀_2, ω₀_n = 1.0, 20.0, 10
ω₀_range = range(ω₀_1, ω₀_2, ω₀_n)

# Time span
t₀, t₁ = 0.0, 50_000.0
t_SPAN = [t₀, t₁]

########################################################################

ω_learned_array = Array{Float64}(undef,0)
ω_teach_array = Array{Float64}(undef,0)

t_calculation_start = time_ns()

# Repeated integration for every ω_teach in a range
for (i,ω_teach) in enumerate(ω_teach_range)
    print_loop_progress && println("ω_teach=$ω_teach, $i/$ω_teach_n")

    for (j,ω₀) in enumerate(ω₀_range)
        global ω_learned_array, ω_teach_array

        system_param = SA[γ, μ, ε, ω_teach]
        U₀ = SA[x₀, y₀, ω₀]
        solution = Hopf_adaptive_integrate(U₀, t_SPAN, system_param)
        x_sol = solution[1,:]
        y_sol = solution[2,:]
        ω_sol = solution[3,:]
        t_sol = solution.t
    
        mean_learned_ω = mean_of_tail(ω_sol, 0.9)
    
        push!(ω_learned_array, mean_learned_ω)
        push!(ω_teach_array, ω_teach)
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

scatter!(ax, ω_teach_array, ω_learned_array, markersize=2)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)
