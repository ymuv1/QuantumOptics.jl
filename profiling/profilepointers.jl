
N = 2^12
M = 2^9

A = rand(Complex128, N)
B = rand(Complex128, N)

function f1(M::Int, A::Vector{Complex128}, B::Vector{Complex128})
    counter1 = 1
    counter2 = 1
    for i=1:M
        counter1 += 2
        counter2 += 2
        B[counter2] = A[counter1]
    end
end

function f2(M::Int, A::Vector{Complex128}, B::Vector{Complex128})
    ptr1 = pointer(A)
    ptr2 = pointer(B)
    for i=1:M
        ptr1 += 2*16
        ptr2 += 2*16
        unsafe_copy!(ptr2, ptr1, 1)
    end
end

@time f1(M, A, B)
@time f1(M, A, B)
@time f2(M, A, B)
@time f2(M, A, B)