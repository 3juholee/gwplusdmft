May23/2016

EXECUTION file:
	"gwd*.py" in each folder of various versions of GW+DMFT
	-Options


This folder is collection of common functions used in each version of GW+DMFT
	
	cix.py
		-an input file used in DMFT(./ctqmc) calculation
		-number of sites, spin configuration, basis assignment (Fock space)...
	
	class_gto.py
		-All the molecular informations from PyQuante.py
		-You need to install PyQuante package for quantum calculation 
		and export it to PYTHONPATH
		-http://pyquante.sourceforge.net/
		
	ftns_Fourier.py
		-Creates Matsubara frequency and imaginary time mesh
			->Create_om_mesh, Create_tau_mesh2
		-All types of Fourier transformation
		-Spline(interpolation) log-mesh frequency to regular mesh one
		
	ftns_Matsubara.py
		-Performs Matsubara summation 
		-The high frequency tail is replaced by asymptotic analytic sum
			-> Cmp_eks, Cmp_eVks,
	
	include_DMFT.py
		-To use it install ctqmc code (Copy Right: Kristjan Haule)
		-Download "dmft_w2k.tgz" from : http://hauleweb.rutgers.edu/downloads/
		-Install it and export the path of executable folder to "WIEN_DMFT_ROOT"
		-IMP class:
			-nom: number of matsubara frequencies at which DMFT self-energy is
			evaluated (100-200 is enough, depending on the temperature)
			-beta: 1/T (30.-100.)
			-M: # of Monte Carlo steps: typically larger than at least 10e6
			-function Run
				-inputs:
					-U
					-mu_QMC=(-Eimp)
					-omega_large: equal mesh of Matsubara frequency (Nomega=2000-5000)
					-ind_om (mapping index of omega_large that corresponds to log
					mesh
					-Delta: (3,nom) array
						-1st column: oms
						-2nd column: Re(Delta)
						-3rd column: Im(Delta)
	
	read.py
		-from "data" where all the calculation results are stored
		read the energy and write down in "ER.dat"
		-ER.dat: Energy vs R
		-Execute in the folder where "data" is stored
	
	remove_imp.py
		-remove unnecessary data files after calculations.
	
	statistics.py
		-from "data" see the conversions of energy over iterations
	
	sub.src
		-submission code used in server
	
	
	
