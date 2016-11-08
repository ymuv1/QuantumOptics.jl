function f_real{T}(x::T, N)
    l = 0.1
    a = 0im
    for i=1:N
        a = l*x
    end
end

function f_compl{T}(x::T, N)
    l::Complex128 = 0.1im
    for i::Int=1:N
        a = l::Complex128*x
    end
    #return nothing
end

#N = 1e10

# @time f_real(1im, 10)
# #@time f_real(1im, N)
# @time f_real(1., 10)
# #@time f_real(1., N)
# @time f_compl(1., 10)
#@time f_compl(1., N)
@time 1

#const a = Complex128(1,0)
const a = 1.
const N = 10^6

f_compl(a, 1)
@time f_compl(a, N)
#@time f_compl(1im, N)