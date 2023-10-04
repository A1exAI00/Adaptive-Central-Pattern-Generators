#=
Misc tools
=#

using Dates

########################################################################

function elapsed_time_string(time_ns)
    seconds = time_ns/1e9
    munutes = round(seconds/60, digits=5)
    hours = round(munutes/60, digits=5)
    return "Elapsed time: $(seconds)s = $(munutes)m = $(hours)h"
end

function get_easy_time()
    return Dates.format(now(), "HH:MM:SS") 
end