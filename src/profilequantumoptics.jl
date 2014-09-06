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
	basis_cavity = FockBasis(2^11)
	a = destroy(basis_cavity)
 	x0 = basis_ket(basis_cavity, 3) + basis_ket(basis_cavity, 5)
 	rho0 = tensor(x0, dagger(x0))
	H = 1*(a + dagger(a))
	J = [a]
    Jdagger = [dagger(j) for j=J]
    Heff = -1im*H -0.5*dagger(a)*a
    Heff_dagger = dagger(Heff)
    tmp1 = Operator(rho0.basis_l, rho0.basis_r)
    tmp2 = Operator(rho0.basis_l, rho0.basis_r)
    drho = Operator(rho0.basis_l, rho0.basis_l)

    rho0_data = rho0.data
    Heff_data = Heff.data
    J_data = [j.data for j=J]
    Jdagger_data = [j.data for j=Jdagger]
    drho_data = drho.data
    tmp_data = tmp1.data 
    function f(N)
        for i=1:N
            quantumoptics.timeevolution_simple.dmaster(rho0, H, J, Jdagger)
        end
    end
    function f2(N)
        for i=1:N
            quantumoptics.timeevolution.dmaster(rho0, H, J, Jdagger, [drho, tmp1, tmp2])
        end
    end
    function f3(N)
        for i=1:N
            quantumoptics.timeevolution.dmaster_Heff(rho0_data, Heff_data, Heff_dagger.data, J_data, Jdagger_data, drho_data, tmp_data)
        end
    end
    function f4(N)
        for i=1:N
            quantumoptics.timeevolution.dmaster_H(rho0_data, H.data, J_data, Jdagger_data, drho_data, tmp_data)
        end
    end
    function f5(N)
        for i=1:N
            quantumoptics.timeevolution.dmaster_Heff(rho0_data, Heff, Heff_dagger, J, Jdagger, drho_data, tmp_data)
        end
    end
    drho = quantumoptics.timeevolution_simple.dmaster(rho0, H, J, Jdagger)
    quantumoptics.timeevolution.dmaster_H(rho0_data, H.data, J_data, Jdagger_data, drho_data, tmp_data)
    drho1 = drho_data*1
    quantumoptics.timeevolution.dmaster_Heff(rho0_data, Heff, Heff_dagger, J, Jdagger, drho_data, tmp_data)
    drho2 = drho_data*1
    println(norm(drho2-drho.data))
    println(norm(drho1-drho.data))
    println(norm(drho1-drho2))
    #error()
    @time 1

    f(1)
    @time f(100)
    f2(1)
    @time f2(100)
    f3(1)
    @time f3(100)
    f4(1)
    @time f4(100)
    f5(1)
    @time f5(100)
    #@profile f4(10000)
end


function profile_master()
	basis_cavity = FockBasis(2^5)
	a = destroy(basis_cavity)
	x0 = basis_ket(basis_cavity, 3)
	rho0 = tensor(x0, dagger(x0))
	H = 1*(a + dagger(a))
	J = [a]
	Jdagger = [dagger(a)]
	T = [0.,1.]

    H_eff = -1im*H -0.5*dagger(a)*a
    H_eff_dagger = dagger(H_eff)
    tmpop = Operator(rho0.basis_l, rho0.basis_r)
    tmp = tmpop.data
    J_data = [j.data for j=J]
    Jdagger_data = [j.data for j=Jdagger]
    tmps = Matrix{Complex128}[]
    for i=length(tmps):11
        push!(tmps, Operator(rho0.basis_l, rho0.basis_r).data)
    end
    tmps_op = Operator[]
    for i=1:14
        push!(tmps_op, Operator(rho0.basis_l, rho0.basis_r))
    end
    
    @time 1

	tout, xout = quantumoptics.timeevolution_simple.master(T, rho0, H, J)
	@time quantumoptics.timeevolution_simple.master(T, rho0, H, J)

	#tout2, xout2 = quantumoptics.timeevolution.master5(T, rho0, H, J, Jdagger, tmps_op)
	#@time quantumoptics.timeevolution.master5(T, rho0, H, J, Jdagger, tmps_op)

    t, x = quantumoptics.timeevolution.master_nh(T, rho0, H_eff, J, Hdagger=H_eff_dagger, Jdagger=Jdagger, tmp=tmpop, tmps_ode=tmps)
    @time quantumoptics.timeevolution.master_nh(T, rho0, H_eff, J, Hdagger=H_eff_dagger, Jdagger=Jdagger, tmp=tmpop, tmps_ode=tmps)
    #@profile quantumoptics.timeevolution.master2(T, rho0, H_eff, H_eff_dagger, J, Jdagger, tmp, tmps)
	println(norm(x[end]-xout[end], Inf))
	#println(xout2[end])
end

Profile.clear()
#profile_basis()
#profile_mul()
#profile_add()
#println(methods(operators.zero!))
#profile_dmaster()
profile_master()
# using ProfileView
# ProfileView.view()
0