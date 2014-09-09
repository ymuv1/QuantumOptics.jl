using Base.Cartesian

function mul_suboperator(a, b, index::Int)
    result = Ket(b.basis)
    result_data = reshape(result.data, reverse(result.shape)...)
    b_data
    for m=1:size(a.data,2)
        
    end
end


# println(macroexpand(:(@nloops 3 i d->(1:b.shape[d]) d->(if d==index && i_d!=m continue end) begin
#             for n=1:size(b.data,1)
#                 (@nref 3 result_data d->(4-d==index ? n : i_{4-d})) += a[n,m] * (@nref 3 b_date d->(i_{4-d}))
#             end
#         end)))


# # println(macroexpand(:(@nloops 3 i b begin
# #                 (@nref 3 result_data i) += a[n,m] * (@nref 3 b_date i)
# #         end)))
# println(macroexpand(:(function mul_suboperator(a::Operator, b::Ket, index::Int)
#     result = Ket(b.basis)
#     result_data = reshape(result.data, reverse(result.basis.shape)...)
#     b_data = reshape(b.data, reverse(result.basis.shape)...)
#     for m=1:size(a.data,2)
#         @nloops 3 i d->(1:b.basis.shape[d]) d->(if d==index && i_d!=m continue end) begin
#             for n=1:size(b.data,1)
#                 (@nref 3 result_data d->(4-d==index ? n : i_{4-d})) += a.data[n,m] * (@nref 3 b_data d->(i_{4-d}))
#             end
#         end
#     end
# end)))

# println(macroexpand(:(@ngenerate N Int64 function gemm_r!{N}(alpha, a::Array{Complex128, N}, b::Array{Complex128, 2},
#                                         index::Int, beta, result::Array{Complex128, N})
#     for n=1:size(b,1)
#         @nloops N i d->(1:size(a,N+1-d)) d->(if d==index && i_d!=n continue end) begin
#             for m=1:size(b,2)
#                  (@nref N result d->(N+1-d==index ? m : i_{N+1-d})) = beta*(@nref N result d->(N+1-d==index ? m : i_{N+1-d})) + alpha*(@nref N a d->(i_{N+1-d})) * b[n,m]
#             end
#         end
#     end
#     return 0
# end)))
# println(macroexpand(:(@ngenerate N Int64 function mul_suboperator_r{N}(a::Array{Complex128, N}, b::Array{Complex128, 2}, index::Int, result::Array{Complex128, N})
#     for n=1:size(b,1)
#         @nloops N i d->(1:size(a,N+1-d)) d->(if d==index && i_d!=n continue end) begin
#                 (@nref N result (d->(i_{N+1-d}))) = alpha * (@nref N psi (d->(i_{N+1-d}))) * op[m,n]
#         end
#     end
#     return 0
# end)))

type Index{I}
    data::Type{Array{Int,I}}
end

Index(i::Int) = Index(Array{Int,i})

#println(macroexpand(:(
@ngenerate RANK Nothing function t{RANK}(x::Type{Type{RANK}}, shape::Vector{Int}, strides::Vector{Int}, index_N::Int, stride_index::Int, a::Matrix, b::Vector, result::Vector)
    @nloops RANK i (d->1:shape[d])  begin
        for m =1:index_N
            for n=1:index_N
                #println(I_0, " ", I_1, " ", I_2, " ", m, " ", n)#, " ", N_0, " ", N_m, " ", a[m,n], " ", b[N_m])
                N_n = (n-1)*stride_index + 1
                N_m = (m-1)*stride_index + 1
                @nexprs RANK (d->(s = (i_d-1)*strides[d]; N_n+=s; N_m+=s))
                #println(i_1, " ", i_2, " ", m, " ", n, " ", N_n, " ", N_m)
                result[N_m] += a[m,n]*b[N_n]
               #N_m += stride_index
            end
        end        
    end
    return nothing
end
#)))





function strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[N] = 1
    for i=N-1:-1:1
        S[i] = S[i+1]*shape[i+1]
    end
    return S
end


a = [-1  4 3; -3 2 1; 6 8 5]
v1 = [3,5]
v2 = [2,7,1]
v3 = [1:10000000]

v = kron(kron(v1,v2),v3)
shape = [length(v1), length(v2), length(v3)]

S = strides(shape)
index = 2

shape_reduced = [shape[1:index-1], shape[index+1:end]]
S_reduced = [S[1:index-1], S[index+1:end]]
stride_index = S[index] 
index_N = shape[index]


println("shape", shape)
println("shape_reduced", shape_reduced)
println("stride", S)
println("stride_reduced", S_reduced)
println("index_N", index_N)
println("stride_index", stride_index)

result = zeros(Int,prod(shape))
@time t(Type{length(S)-1}, shape_reduced, S_reduced, index_N, stride_index, a, v, result)
result = zeros(Int,prod(shape))
@time t(Type{length(S)-1}, shape_reduced, S_reduced, index_N, stride_index, a, v, result)

@time kron(kron(v1,a*v2), v3)
@time kron(kron(v1,a*v2), v3)
#Index(3)
#t4(Index{3})

# @ngenerate N Nothing function t{N}(a::Array{Complex128, 2}, b::Array{Complex128, N}, index::Int, result::Array{Complex128, N})
#     for m=1:size(a,2)
#         @nloops N i d->(1:size(b,N+1-d)) d->(if d==index && i_d!=m continue end) begin
#             for n=1:size(a,1)
#                 #1+1
#                 #(@nref N result d->(N+1-d==index ? n : i_{N+1-d})) += 1#a[n,m] * (@nref N b d->(i_{N+1-d}))
#                 @time (@nref 2 a i) = 0
#                 @time (@nref 2 a i) = 0
#                 error()
#             end
#         end
#     end
#     return nothing
# end

# a = rand(Complex128, 2,2)
# b = rand(Complex128, 2,2,2,2,2,2,2,2,2)
# result = rand(Complex128, 2,2,2,2,2,2,2,2,2)

# f{N}(b::Array{Complex128,N}) = (b[2,2,2,2,2]=1; nothing)
# #@time f(b)
# #@time f(b)
# @time t(a,b,8, result)
# @time t(a,b,8, result)
