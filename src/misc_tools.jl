#=
Misc tools
=#

using Dates

########################################################################

function elapsed_time_string(time_ns)
    seconds = time_ns/1e9
    munutes = seconds/60
    hours = munutes/60
    return "Elapsed time: $(seconds)s = $(munutes)m = $(hours)h"
end

function get_easy_time()
    return Dates.format(now(), "HH:MM:SS") 
end