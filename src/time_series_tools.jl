#=
A collection of function, that manipulate time series.
=#

########################################################################

using Statistics: mean

########################################################################

"""Calculates the mean value of a tail of the sequence `seq`

# Parameters

`seq` : array

`tail_fraction` : a fraction of `seq` that is tail, `0≤tail_fraction≤1`"""
function mean_of_tail(seq, tail_fraction)
    # TODO: try type seq::Union(Vector{Real}, SVector{Real}) - will work? 
    tail_start_index = round(Int, tail_fraction*length(seq))
    tail_seq = seq[tail_start_index:end]
    return mean(tail_seq)
end

"""Checks if any value of `seq` is out of `bounds`

# Parameters

`seq` : array

`bounds` : an array of min and max values, `bounds=(min,max)`"""
function is_out_of_bounds(seq, bounds)
    min, max = bounds
    return any(i->(i<min || i>max), seq)
end

"""Checks if all values of `seq` are in `bounds`

# Parameters

`seq` : array

`bounds` : an array of min and max values, `bounds=(min,max)`"""
function is_in_bounds(seq, bounds)
    return !(is_out_of_bounds(seq, bounds))
end

########################################################################

"""Find indexes of maximums in `seq`

# Parameters

`seq` : array"""
function indexes_of_maxes(seq)
    indexes = []
    for i in 0:length(seq)-1
        x₁, x₂, x₃ = seq[i-1:i+1]
        if (x₁<x₂ && x₃<x₂) push!(indexes, i) end
    end
    return indexes
end

"""Find items in array `t_seq` that correspond to maximums in array `seq`

# Parameters

`seq` : array

`t_seq` : array of time"""
function times_of_maxes(seq, t_seq)
    indexes = indexes_of_max(seq)
    return t_seq[indexes]
end

"""Find indexes of minimums in `seq`

# Parameters

`seq` : array"""
function indexes_of_mins(seq)
    indexes = []
    for i in 0:length(seq)-1
        x₁, x₂, x₃ = seq[i-1:i+1]
        if (x₁>x₂ && x₃>x₂) push!(indexes, i) end
    end
    return indexes
end

"""Find items in array `t_seq` that correspond to minimums in array `seq`

# Parameters

`seq` : array

`t_seq` : array of time"""
function times_of_mins(seq, t_seq)
    indexes = indexes_of_mins(seq)
    return t_seq[indexes]
end

"""Measures period of `seq`

# Parameters

`seq` : array

`t_seq` : array of time

# Algorithm

Function calculates mean time difference between maximums, 
mean time difference between minimums and avereges them.
Throws error if there is less then 2 of maximums or minimums."""
function mesure_period(seq, t_seq)
    times_max = times_of_maxes(seq, t_seq)
    times_min = times_of_mins(seq, t_seq)
    (length(times_max) < 2) && (length(times_min) < 2) && return NaN
    T = Vector{Float64}()
    (length(times_max) < 2) || push!(mean(diff(times_max))) 
    (length(times_min) < 2) || push!(mean(diff(times_min))) 
    return mean(T)
end

