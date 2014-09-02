include("quantumoptics.jl")
#include("ode.jl")

using quantumoptics
#using ode


function profile_basis()
	basis0 = GenericBasis([4,5])
	basis1 = GenericBasis([3,7])
	basis = compose(basis0, basis1)
	#basis = basis0

	println(multiplicable(basis, basis))
	function f(N)
		for i=1:N
			#basis==basis
			multiplicable(basis, basis)
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

	mul!(op1, op2, tmp)
	println(tmp)
	@time mul!(op1, op2, tmp)
	mul!(op1.data, op2.data, tmp.data)
	@time mul!(op1.data, op2.data, tmp.data)
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
	function f2(N)
		for i=1:N
			add!(op1,op2)
		end
	end
	f2(1)
	@time f2(100)
end


function profile_dmaster()
	basis_cavity = FockBasis(2^11)
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
            quantumoptics.dmaster(rho0, H, J)
        end
    end
    function f2(N)
        for i=1:N
            quantumoptics.dmaster2(rho0, H, J, Jdagger, drho, tmp1, tmp2)
        end
    end
    f(1)
    @time f(10)
    f2(1)
    @time f2(10)
end

function profile_ode()
	f(t::Number, y::Vector{Complex{Float64}}) = y
	y0 = zeros(Complex128, 10)
	T = [0,10]
	f(0, y0)
	@time f(0, y0)
	ode.ode45(f, y0, T)
	@time ode.ode45(f, y0, T)
end

function profile_master()
	basis_cavity = FockBasis(2^8)
	a = destroy(basis_cavity)
	x0 = basis_ket(basis_cavity, 3)
	rho0 = tensor(x0, dagger(x0))
	H = 1*(a + dagger(a))
	J = [a]
	T = [0,1]
	tout, xout = quantumoptics.master(T, rho0, H, J)
	#println("T: ", tout)
	@time quantumoptics.master(T, rho0, H, J)
	tout2, xout2 = quantumoptics.master2(T, rho0, H, J)
	#println("T: ", tout)
	@time quantumoptics.master2(T, rho0, H, J)
end

#profile_basis()
#profile_mul()
#profile_add()
#profile_dmaster()
#profile_ode()
profile_master()
0