#include("quantumoptics.jl")
#include("ode.jl")

using quantumoptics
#using ode


function profile_basis()
	basis0 = GenericBasis([4,5])
	basis1 = GenericBasis([3,7])
	basis = bases.compose(basis0, basis1)
	#basis = basis0

	println(bases.multiplicable(basis, basis))
	function f(N)
		for i=1:N
			#basis==basis
			bases.multiplicable(basis, basis)
		end
	end
	f(1)
	@time f(10000)
end

function profile_mul()
	basis = FockBasis(3)
	op2 = destroy(basis)
	op1 = create(basis)
	tmp = Operator(basis,basis)
	tmp = op1*op2
	@time tmp = op1*op2
	println(tmp)

	inplacearithmetic.mul!(op1, op2, tmp)
	println(tmp)
	@time inplacearithmetic.mul!(op1, op2, tmp)
	inplacearithmetic.mul!(op1.data, op2.data, tmp.data)
	@time inplacearithmetic.mul!(op1.data, op2.data, tmp.data)
end

function profile_add()
	basis = FockBasis(400)
	op1 = destroy(basis)
	op2 = create(basis)
	function f(N)
		for i=1:N
			op1 += op2
		end
	end
	f(1)
	@time f(100)

	op1 = destroy(basis)
	op2 = create(basis)
	result = Operator(basis)
	function f2(N)
		for i=1:N
			inplacearithmetic.add!(op1,op2,result)
		end
	end
	f2(1)
	@time f2(100)
end


function profile_dmaster()
	basis_cavity = FockBasis(2^6)
	a = destroy(basis_cavity)
 	x0 = basis_ket(basis_cavity, 3)
 	rho0 = tensor(x0, dagger(x0))
	H = 1*(a + dagger(a))
	J = [a]
    Jdagger = [dagger(j) for j=J]
    tmp1 = Operator(rho0.basis_l, rho0.basis_r)
    tmp2 = Operator(rho0.basis_l, rho0.basis_r)
    drho = Operator(rho0.basis_l, rho0.basis_l)

    function f(N)
        for i=1:N
            quantumoptics.timeevolution_simple.dmaster(rho0, H, J)
        end
    end
    function f2(N)
        for i=1:N
            quantumoptics.timeevolution.dmaster(rho0, H, J, Jdagger, drho, tmp1, tmp2)
        end
    end
    f(1)
    @time f(10)
    f2(1)
    @time f2(10)
end


function profile_master()
	basis_cavity = FockBasis(3)
	a = destroy(basis_cavity)
	x0 = basis_ket(basis_cavity, 3)
	rho0 = tensor(x0, dagger(x0))
	H = 1*(a + dagger(a))
	J = [a]
	T = [0,1]
	tout, xout = quantumoptics.timeevolution_simple.master(T, rho0, H, J)
	#println("T: ", tout)
	@time quantumoptics.timeevolution_simple.master(T, rho0, H, J)
	tout2, xout2 = quantumoptics.timeevolution.master(T, rho0, H, J)
	#println("T: ", tout)
	@time quantumoptics.timeevolution.master(T, rho0, H, J)
	#println(norm(xout2[end]-xout[end], Inf))
	#println(xout2[end])
end

#profile_basis()
#profile_mul()
#profile_add()
#println(methods(operators.zero!))
#profile_dmaster()
profile_master()

0