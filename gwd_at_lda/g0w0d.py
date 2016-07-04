"""
1/9/2016 (new ver6_gw+dmft.py)
updated: --mu / --nom options
"""

import time
import sys
import os
import argparse
from scipy import *
from scipy import linalg,integrate,interpolate
from PyQuante import *
from PyQuante.LA2 import geigh,trace2
from PyQuante.Ints import get2JmK,getbasis,getints, fetch_jints, fetch_kints

from class_gto import index_double, trans_2matrix, trans_4matrix,\
		basis_transform, gtos, HartreeF, savedata,LDA_gto
from ftns_Matsubara import Cmp_DensityMatrix, Cmp_mu
from ftns_Fourier import Spline_Real,Spline_Complex,Gw_to_Gtau,Atau_to_Aw,\
		Create_om_mesh, Create_tau_mesh2,Ww_to_Wtau, Ww_to_Wtau_loc
from include_DMFT import IMP


def extract_sig(Sigma,oms_f_log,oms_f,outputname="SigGW.out"):
	data = zeros((len(oms_f),3),float)
	data[:,0] = oms_f
	Sig = Spline_Complex(Sigma, oms_f_log,oms_f)
	data[:,1] = real(Sig)
	data[:,2] = imag(Sig)
	savetxt(outputname,data)

def Create_Pinit_loc(E0,TProj,oms):
	"""
	P(t) = -G(t)*G(-t)
	--> P(iw) = (fa - fb)/(iw-(Ea-Eb))
	"""
	M = len(E0)
	ist, dict_ist = index_double(M)
	MM = M*M
	def matsubara_Pw(a,b):
		if a == 0 and b != 0: 
			r =  -2./(1j*w-(E0[a]-E0[b]))
			return r
		if b == 0 and a != 0: 
			r = +2./(1j*w-(E0[a]-E0[b]))
			return r
		else: return 0.0
	
	Pw = zeros((len(oms)),dtype = float)
	for iw,w in enumerate(oms):
		PHF_w = zeros((MM,MM),dtype = float)
		for p,(a,d) in enumerate(ist):
			for q,(c,b) in enumerate(ist):
				if a==c and b==d:
					PHF_w[p,q] = real(matsubara_Pw(a,b))
		Pw[iw] = trans_4matrix(PHF_w,TProj.T)[0,0]
	return Pw


def Create_Ginit(Ener,mu, Evec, oms_f_log,HFbasis=True):
	G_init = zeros((len(oms_f_log),M,M),complex)
	for iw,w in enumerate(oms_f_log):
		Gdiag = zeros((M,M),complex)
		for i in range(M):
			Ginv = 1j*w - (Ener[i]-mu)
			Gdiag[i,i] = 1/Ginv
			if not HFbasis:
				G_init[iw] = trans_2matrix(Gdiag,Evec.T)
			else: G_init[iw] = Gdiag
	return G_init


def R_svd_init(E0,UH, UdotS_init, UdotSi_init , oms): 
	#density-density correlation R=P/(1-UP)
	#Singular value decomposition(svd) for memory and time saving
	
	def matsubara_Pw(a,b):
		if a == 0 and b != 0: 
			r =  -2./(1j*w-(E0[a]-E0[b]))
			return real(r)
		if b == 0 and a != 0: 
			r = +2./(1j*w-(E0[a]-E0[b]))
			return real(r)
		else: return 0.0

	M = len(E0)
	MM = M*M
	ist, dict_ist = index_double(M)
	Msvd = len(UdotS_init.T)

	Rsvd = zeros((len(oms),Msvd,Msvd),float)
	UH_svd = dot( UdotSi_init.T, dot(UH, UdotSi_init))
	
	for iw,w in enumerate(oms):
		PHF_w = zeros((MM,MM),dtype = 'float')
		for p,(a,d) in enumerate(ist):
			for q,(c,b) in enumerate(ist):
				if a==c and b==d:
					PHF_w[p,q] = matsubara_Pw(a,b)
		Pw_svd = dot( UdotS_init.T, dot(PHF_w, UdotS_init))

		UPsvd = dot(UH_svd, Pw_svd)
		UPsvd_inv = linalg.inv(identity(Msvd) - UPsvd)
		Rsvd[iw] = dot(Pw_svd,UPsvd_inv)

	return Rsvd


def R_svd_Gtau(Gtau,UH, UdotS_init, UdotSi_init , oms): 
	
	MM = len(Gtau[0])
	M = int(sqrt(MM+1e-5))
	Msvd = len(UdotS_init.T)
	ist, dict_ist = index_double(Msvd)

	Psvd_tau = zeros((len(tau_mesh),Msvd,Msvd),float)
	R_svd = zeros((len(oms),Msvd,Msvd),float)
	UH_svd = dot( UdotSi_init.T, dot(UH, UdotSi_init))
	
	for itau,tau in enumerate(tau_mesh):
		Gt = Gtau[itau][newaxis]
		Gt_inv = -(Gtau[-itau-1])[newaxis] #G(-tau)=-G(beta-tau)
		Ptau = 2*dot(Gt.T,Gt_inv)
		Ptau2 = reshape(Ptau,(M,M*M,M))
		USigmadotP = dot(UdotS_init.T,Ptau2)
		USigmadotP = reshape(USigmadotP,(Msvd,M*M))
		Psvd_tau[itau] = dot(USigmadotP, UdotS_init)

	P_tau_large = zeros((len(tau_large),Msvd,Msvd),float)
	for i,(i1,i2) in enumerate(ist):
		P_tau_large[:,i1,i2] = Spline_Real(Psvd_tau[:,i1,i2],tau_mesh,tau_large)
	Psvd_w = real(Atau_to_Aw(P_tau_large, tau_large, oms))
	
	for iw,w in enumerate(oms):
		UPsvd = dot(UH_svd, Psvd_w[iw])
		UPsvd_inv = linalg.inv(identity(Msvd) - UPsvd)
		R_svd[iw] = dot(Psvd_w[iw],UPsvd_inv)

	return R_svd

def Gtau_to_Pwloc(Gtau, tau_mesh,tau_large,oms_b_log):
	"""
	P(t) = G(t)*G(-t) = -G(t)*G(beta-t)
			 = -G(t)*G(t)[::-1]
	"""
	Gtau_large = Spline_Real(Gtau,tau_mesh, tau_large)
	Gtau_large_inv = -Gtau_large[::-1] #G(-tau)
	Ptau_large = 2.*Gtau_large*Gtau_large_inv
	Pw = real(Atau_to_Aw(Ptau_large,tau_large,oms_b_log))
	return Pw

def SVD(S,bfs,U_gi):
	from PyQuante.CGBF import three_center
	"""
	Single variable decomposition (SVD) within HF product basis
	"""
	Sinv = linalg.inv(S)
	M = len(bfs)
	M2 = M**2
	ist,dict_ist = index_double(M)
	product_basis = zeros((M,M2),float)
	
	for a in range(M):
		for il, (i1,i2) in enumerate(ist):
			product_basis[a,il] = three_center(bfs[a],bfs[i1],bfs[i2])

	S_ijkl = dot(product_basis.T,dot(Sinv,product_basis))
	SHF_abcd = trans_4matrix(S_ijkl,U_gi)
	eivalSHF, eivecSHF = linalg.eigh(SHF_abcd)
	
	if False:
		test = matrix(eivecSHF.T)*matrix(eivecSHF)
		for ii in range(M*M):
			for jj in range(M*M):
				if ii==jj:
					if abs(1-test[ii,jj])>1e-6:
						print ii
				else: 
					if abs(test[ii,jj])>1e-6:
						print ii,jj
	for i,ev in enumerate(eivalSHF[::-1]):
		#print i,ev
		if ev < 1e-6:
			iLimit = i
			print 'iLimit of Tsvd = ',iLimit
			break
	
	Usvd = (eivecSHF.T[::-1][:iLimit]).T #Usvd_IP
	SQRTsigma = sqrt(eivalSHF[::-1][:iLimit])
	UdotS = dot(Usvd,diag(SQRTsigma)) #U_IP * sqrt(sigma_P)
	UdotSi = dot(Usvd,diag(1/SQRTsigma)) #U_IP * sqrt(1/sigma_P)
	#test = dot(U_inv_sqsigma.T,U_sqsigma)
	#print test
	return UdotS, UdotSi


def Create_Sigma_tau(R_tau, Gtau, UH, UdotSi_HF, tau_mesh):
	M = int(sqrt(len(UH)+0.1))
	if M*M != len(UH):
		print "ERROR:MM != M**2"
		sys.exit()
	ist, dict_ist = index_double(M)

	Sigma_tau = zeros((len(tau_mesh),M*M),float)
	
	vUS_i = dot(UH, UdotSi_HF) # v * (Usvd* 1/sqrt(S))
	for itau, tau in enumerate(tau_mesh):
		#print itau
		G = Gtau[itau]
		WHtau = dot(vUS_i, dot( R_tau[itau], vUS_i.T)) #W = v*R*v
		W = reshape(WHtau, (M,M*M,M))
		Sig2 = -dot(G,W)
		Sigma_tau[itau] = reshape(Sig2,(M*M))
	return Sigma_tau


def Cmp_Delta(Eimp, Gloc, Sigs):
	Delta=zeros(shape(Sigs),dtype=float)
	Delta[:,0] = Sigs[:,0]
	for iw,w in enumerate(Sigs[:,0]):
		Sigm = (Sigs[iw,1]+Sigs[iw,2]*1j)
		Delt = w*1j-Eimp-Sigm-1/Gloc[iw]
		Delta [iw,1:] = array([Delt.real,Delt.imag])
	return Delta


def Create_Gw_Gloc(oms_f_log,mu,Ham,Um_RL):
	dim = shape(Ham)
	M = dim[-1]
	Gw = zeros((len(oms_f_log),M,M),complex)
	Gloc = zeros((len(oms_f_log)),complex)
	for iw,w in enumerate(oms_f_log):
		if len(dim)<3:
			Gw_inv = (1j*w+mu)*identity(M)- Ham
		else:
			Gw_inv = (1j*w+mu)*identity(M)- Ham[iw]
		Gw[iw] = linalg.inv(Gw_inv)
		Gloc[iw] = trans_2matrix(Gw[iw],Um_RL)[0,0]
	return Gw,Gloc
		

if __name__ == '__main__':
################## options ####################
		
	parser = argparse.ArgumentParser()

	parser.add_argument('--natom', action='store', dest='natom',type=int)
	parser.set_defaults(natom=1)
	
	parser.add_argument('--R', action='store', dest='R',type=float)
	parser.set_defaults(R=1.4)

	parser.add_argument('--basis', action='store',dest='basis',type=str)
	parser.set_defaults(basis='ccpvdz')

	parser.add_argument('--beta', action='store', dest='beta',type=float)
	parser.set_defaults(beta=30.)

	parser.add_argument('--Nomega', action='store', dest='Nomega',type=int)
	parser.set_defaults(Nomega=2000)
	
	parser.add_argument('--nom', action='store', dest='nom',type=int)
	parser.set_defaults(nom=100)
	
	parser.add_argument('--nomD', action='store', dest='nomD',type=int)
	parser.set_defaults(nomD=100)
	
	parser.add_argument('--dmft', action='store_true',dest='DMFT')
	parser.set_defaults(DMFT=False)
	
	parser.add_argument('--Nfix', action='store_true',dest='Nfix')
	parser.set_defaults(Nfix=False)
	
	parser.add_argument('--mu', action='store', dest='mu',type=float)
	
	parser.add_argument('--Mstep', action='store', dest='Mstep',type=float)
	parser.set_defaults(Mstep=2e6)

	parser.add_argument('--Nitt', action='store', dest='Nitt',type=int)
	parser.set_defaults(Nitt=20)
	
	parser.add_argument('--mixr', action='store', dest='mixr',type=float)
	parser.set_defaults(mixr=0.5)

	parser.add_argument('--xtol', action='store', dest='xtol',type=float)
	parser.set_defaults(xtol=5e-6)


	results = parser.parse_args()
	natom = results.natom
	R = results.R
	basis = results.basis
	beta = results.beta
	Nomega = results.Nomega
	nom = results.nom
	nomD = results.nomD
	DMFT = results.DMFT
	mu = results.mu
	Nfix = results.Nfix
	Mstep=results.Mstep
	Nitt = results.Nitt
	mixr = results.mixr
	xtol = results.xtol
	if mu:
		fixmu=True
		print "mu=",mu
		mu0=mu
	else:
		fixmu=False
		print "No initial mu, updated at each iteration"
	
	Noccup= natom*2.0  # occupancy for H2

	print "natom=",natom
	print "R=%2.3f"%R
	print "basis=",basis
	print "beta=",beta
	print "nomD=",nomD
	print "Nomega=",Nomega
	print "DMFT = ", DMFT
	print "Mstep=",Mstep
	print "Nitt=", Nitt
	print "mixr=%2.2f"%mixr
	print "convergence=",xtol
	print 

	###############Global vaules#####################

	nom_tail = 50 #number of log tail
	oms_f, oms_f_log, oms_b, oms_b_log, oms_equal_f,oms_ind = Create_om_mesh(beta, Nomega, nom, nom_tail)
	
	Ntau = 2*Nomega
	tau_safe = 8.
	ntau = 100
	tau_mesh, tau_large = Create_tau_mesh2(beta,Ntau,ntau,tau_safe)
	if False:
		print "tau_mesh and tau_large are the same"
		tau_mesh = tau_large
	#savetxt('tau_large.txt',array([tau_large,zeros(len(tau_large),float)]).transpose())

	mu_a, mu_b = -5., 5.
	
	print '# log_omega = ', len(oms_b_log)
	print '# mesh_omega = ', len(oms_b)
	print '# logmesh_tau = ', len(tau_mesh)
	print '# mesh_tau = ', len(tau_large)


################# basis and non-interacting H ####################
	print 'basis = ',basis
	print "R = ", R

	gto = gtos(natom,R, basis,LDA=True)
	h0, UH, UF = gto.inputs
	eivec_gto = gto.eivec #U such that |H2> = U_(g,H2)|g>
	bfs = gto.bfs
	S = gto.S
	gr = gto.gr
	
	M = len(h0)
	print "Dimensions = ", M
	ist , dict_ist = index_double(M)

	#Ener0, Evec0, Nt = HartreeF(natom, R, h0, UH, UF)
	Ener0, Evec0, Nt = LDA_gto(natom, R, h0, UH, UF, gr, eivec_gto)
	from scipy import *
	
	#HF solution is now the basis
	h0, UH, UF = basis_transform(h0,UH, UF,Evec0)
	
########### LOCAL basis and variables ################
	
	#{R,L} basis <i|RL>
	RL = identity(M)
	sq2 = 1/sqrt(2.)
	RL[0,0]= sq2
	RL[0,1]= sq2
	RL[1,0]= sq2
	RL[1,1]= -sq2
	
	#Unitary matrix between given basis and RL basis. Um_RL = <m|i><i|RL>
	Um_RL = dot(Evec0.T,RL)
	#<RL|m> for first two space: TProj for using in DMFT
	TProj = Um_RL.T[:2]

	#define U_local
	U_local = trans_4matrix(UH,TProj.T)[0,0] #<m|RL>
	print "U_local = ", U_local
	
	#SVD
	U_gHF = dot(eivec_gto, Evec0) #U such that |HF> = U_(g,H2)U_(H2,HF)|g>
	UdotS_HF, UdotSi_HF = SVD(S,bfs,U_gHF)
	
########### Initial variables (mu,G,Gloc,Nt,Eimp,Delta) ###########

	##########mu0########
	if not fixmu:
		mu=0.5*(Ener0[0]+Ener0[1])
		print "mu is calculated by N(mu)=2.0" 
		print "mu0= ", mu
	
	##########G0/Gloc0 based on HF###########
	Gw = Create_Ginit(Ener0,mu,identity(M),oms_f_log)
	Gtau = Gw_to_Gtau(Gw, oms_f_log ,oms_f,tau_mesh, beta)
	Gloc = zeros(len(oms_f_log),complex)
	for iw,w in enumerate(oms_f_log):
		Gloc[iw] = trans_2matrix(Gw[iw],Um_RL)[0,0]
	Gloctau = Gw_to_Gtau(Gloc, oms_f_log ,oms_f,tau_mesh, beta)

	##########Nt0 based on HF##############
	Nt = zeros((M,M),float)
	for i in range(natom):
		Nt[i,i] = 2.0
	Nc = reshape(Nt,(M*M,))
	
	######### W0(=V+VRV) and Wloc0  based on HF ##############
	R_w = R_svd_init(Ener0, UH, UdotS_HF, UdotSi_HF , oms_b_log)
	R_tau = Ww_to_Wtau(R_w, tau_mesh, oms_b_log, oms_b,beta)
	
	############ local polarization and local W for DMFT ##########
	Pwloc = Create_Pinit_loc(Ener0,TProj,oms_b_log)
	Wloc = U_local/(1-U_local*Pwloc)-U_local
	Wloc_tau = Ww_to_Wtau_loc(Wloc, tau_mesh, oms_b_log, oms_b,beta)

	#############DMFT: Eimp0 / Delta0 ###############################
	if DMFT:
		ctqmc = IMP(nom,beta,Mstep)
		Eimpf = open('imp/Eimp.dat','w')
		Eimp= trans_2matrix(diag(Ener0),TProj.T)[0,0]-mu
	
		if os.path.exists("./Delta.dat"):
			print "old Delta file exists!!!"
			Delta_l=loadtxt("Delta.dat")
			Delta=zeros((len(oms_f_log),3),float)
			for iw,w in enumerate(oms_f_log):
				Delta[iw]=Delta_l[oms_ind[iw]]
		else:
			print "no old Delta, compute a new one"
			Sigs = zeros((len(oms_f_log),3),dtype=float)
			Sigs[:,0] = oms_f_log
			Delta = Cmp_Delta(Eimp, Gloc, Sigs)


	##################### GW self-energy #####################
	Sigma_gw_tau = Create_Sigma_tau(R_tau, Gtau, UH, UdotSi_HF, tau_mesh)
	Sigma_tau_large = zeros((len(tau_large),len(ist)),float)
	for i,(i1,i2) in enumerate(ist):
		Sigma_tau_large[:,i] = Spline_Real(Sigma_gw_tau[:,i],tau_mesh,tau_large)
	Sigma_gw1 = Atau_to_Aw(Sigma_tau_large, tau_large, oms_f_log)
	Sigma_gw = reshape(Sigma_gw1,(len(Sigma_gw1),M,M))
	extract_sig(Sigma_gw[:,0,0],oms_f_log,oms_f,"Sigma_GW00.out")
	extract_sig(Sigma_gw[:,1,1],oms_f_log,oms_f,"Sigma_GW11.out")
	
	###############local GW self-energy for DMFT #############
	Sigma_tau_loc = -Gloctau*Wloc_tau
	Sigma_tau_loc_large= Spline_Real(Sigma_tau_loc,tau_mesh,tau_large)
	Sigma_gw_loc = Atau_to_Aw(Sigma_tau_loc_large, tau_large, oms_f_log)
	extract_sig(Sigma_gw_loc,oms_f_log,oms_f,"Sigma_GW_DC.out")
		

###############self-consistent loop################
	p_ener = 0.0
	for itt in range(Nitt):

	
		##################Hartree-Fock self-energy ###############
		VHa = dot(UH, Nc)     # Hartree
		VFx = -0.5*dot(UF,Nc) # Exact exchange
		JK = reshape(VHa+VFx,(M,M))
		Ham = h0+JK
		Ener,Evec = linalg.eigh(Ham)
		print Ener[:5]

		################ DMFT self-energy #########################
		if DMFT:
			(Sigs, n_imp, TrSigmaG) =	ctqmc.Run(U_local,-Eimp,oms_f,oms_ind,Delta)
			ctqmc.copyfiles(itt)
			Edc = 0.5*U_local*n_imp
		
	
		################# update (mu,G,Gloc,Nt,Eimp,Delta) ###################
		eks_GW = zeros((len(oms_f_log),M),dtype = 'complex')
		Ars_GW = zeros((len(oms_f_log),M,M),dtype = 'complex')
		Ham_GW = zeros((len(oms_f_log),M,M),'complex')
		
		for iw,w in enumerate(oms_f_log):
			if DMFT:
				Sig_dmft = (Sigs[iw,1]+Sigs[iw,2]*1j)
				Sig_loc = dot(TProj.T,dot((identity(2)*\
						(Sig_dmft-Sigma_gw_loc[iw]-Edc)),TProj))
				Sig_tot=Sigma_gw[iw]+Sig_loc

			Ham_GW[iw] = h0+JK + Sig_tot
			(Ener_GW,Evec_GW) = linalg.eig(Ham_GW[iw])
			eks_GW[iw] = Ener_GW
			Ars_GW[iw] = Evec_GW

		#mu1
		if not fixmu:
			if Nfix:
				"total N is fixed by 2.0"
				mu0 = Cmp_mu(Noccup/2.,mu_a,mu_b,eks_GW,oms_f_log,oms_equal_f,beta)
			else:
				mu0 = 0.5*(Ener[natom-1]+Ener[natom])
		mu = mu0*(1.-mixr)+mu*mixr #mix

		#G1 and Gloc1  
		Gw0, Gloc0 = Create_Gw_Gloc(oms_f_log,mu,Ham_GW,Um_RL)
		Gtau0 = Gw_to_Gtau(Gw0, oms_f_log ,oms_f,tau_mesh, beta)
		Gloctau0 = Gw_to_Gtau(Gloc0, oms_f_log ,oms_f,tau_mesh, beta)
		# New Eimp and Delta
		if DMFT:
			Eimp0 = dot(TProj,dot(Ham,TProj.T))[0,0]-mu-Edc
			Delta0 = Cmp_Delta(Eimp0, Gloc0, Sigs)
		#Nt1
		Nt0 = 2.*Cmp_DensityMatrix(eks_GW,Ars_GW,mu,oms_f_log,oms_equal_f,beta)

		#mixr
		Gtau 	 = Gtau0*		(1.-mixr)+Gtau		*mixr
		Gloctau= Gloctau0*(1.-mixr)+Gloctau*mixr
		Nt		 = Nt0*			(1.-mixr)+Nt			*mixr
		if DMFT:
			Eimp	 = Eimp0*		(1.-mixr)+Eimp		*mixr
			Delta	 = Delta0*  (1.-mixr)+Delta 	*mixr

		##########print out data for spectral calculation###########
		Nc = reshape(Nt,(M*M,))
		if DMFT:
			print >> Eimpf, itt, Eimp

		#Ntotal, G_total, Nc
		n_ = 0.0
		for i in range(M):
			n_ += real(Nt[i,i])
		
		spectral = zeros(len(Gw),complex)
		for iw in range(len(Gw)):
			sp = trace(Gw[iw])
			spectral[iw] = sp
		spectral_large = Spline_Complex(spectral, oms_f_log,oms_f)
		Gf = zeros((len(oms_f),3),float)
		Gf[:,0] = oms_f
		Gf[:,1] = real(spectral_large)
		Gf[:,2] = imag(spectral_large)
		savetxt("Gf_GW.out",Gf)
		savetxt('Nc.out',Nc)
		savetxt('H_HF.out',Ham)

		######################### Energy ########################
		#EHF
		Eone = trace2(h0,Nt)
		EHa = 0.5*sum(Nc*VHa)
		EFx = 0.5*sum(Nc*VFx)
		EHF = Eone + EHa + EFx + 2./R*natom**2 #HF total energy
		
		#GW
		TrSigmaG_tau = zeros((len(tau_mesh)),float)
		for itau,tau in enumerate(tau_mesh):
			TrSigmaG_tau[itau] = - dot(Gtau[itau],Sigma_gw_tau[::-1][itau])
		
		delE_GW = 0.5*2*integrate.simps(TrSigmaG_tau,tau_mesh) # 2 for SPIN summation!!
		Etot = EHF+delE_GW  #GW total energy 

		
		print "mu = ",mu
		print 'Ntotal = ', n_
		print 'Eone = ', Eone
		print 'Ej = ', EHa
		print 'Exch = ', EFx
		print 'EHF = ', EHF
		print 'delE_GW = ',delE_GW

		#DMFT (local energy per each site)
		if DMFT:
			PhiDMFT = TrSigmaG
			delE_GW_dc = 0.5*2*integrate.simps(-Gloctau[::-1]\
					*Sigma_tau_loc,tau_mesh) # 2 for SPIN summation!!
			PhiDC_HF = 1./4*U_local*n_imp**2
			PhiDC_GW = delE_GW_dc
			Nloc = dot(TProj,dot(Nt,TProj.T))[0,0]
		
			Etot += (PhiDMFT-PhiDC_HF-PhiDC_GW)*2
			print 'Nimp = ',	n_imp
			print 'Nloc = ',	Nloc
			print 'Edc = ',	Edc
			print 'PhiDMFT = ', PhiDMFT*2
			print 'EHF_DC = ', PhiDC_HF*2
			print 'DMFT-HF_DC = ',PhiDMFT*2-PhiDC_HF*2
			print 'delE_GW_DC = ', PhiDC_GW*2
			print 'GW_nonloc = ',delE_GW-PhiDC_GW*2

		convergence = (Etot-p_ener)
		print R, Etot, convergence , "itt = ",itt


		if not DMFT and abs(convergence) < xtol: 
			print "self-consistent! ", "# of iteration = ",itt
			print 'EHF     = ',EHF
			print 'delE_GW = ',delE_GW
			#print Nt
			print R, Etot
			break
		
		p_ener = Etot


