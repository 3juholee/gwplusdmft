from class_gto import *
from scipy import *
from scipy import linalg
from scipy import interpolate
import sys
import time


def extract_sig(Sigma,oms_f_log,oms_f,outputname="SigGW.out"):
	data = zeros((len(oms_f),3),float)
	data[:,0] = oms_f
	Sig = Spline_Complex(Sigma, oms_f_log,oms_f)
	data[:,1] = real(Sig)
	data[:,2] = imag(Sig)
	savetxt(outputname,data)
			

def log_mesh(om,nom, ntail_,del_nom = 1):
    """Creates logarithmic mesh on Matsubara axis
       Takes first istart points from mesh om and
       the rest of om mesh is replaced by ntail poinst
       redistribued logarithmically.
       Input:
           om      -- original long mesh
           istart  -- first istart points unchanged
           ntail   -- tail replaced by ntail points only
       Output:
           som             -- smaller mesh created from big om mesh
           sSig[nom,nc]    -- Sig on small mesh
       Also computed but not returned:
           ind_om  -- index array which conatins index to
                      kept Matsubara points
    """
    
    istart = min(nom, len(om))
    ntail = min(ntail_, len(om)-istart)

    ind_om=[]
    alpha = log((len(om)-1.)/istart)/(ntail-1.)
    for i in range(istart)[::del_nom]:
        ind_om.append(i)
    for i in range(ntail):
        t = int(istart*exp(alpha*i)+0.5)
        if (t != ind_om[-1]):
            ind_om.append(t)

    ind_oms_equal = [[0]]
    for it in range(1,len(ind_om)-1):
        istart = int(0.5*(ind_om[it-1]+ind_om[it])+0.51)
        iend = int(0.5*(ind_om[it]+ind_om[it+1])-0.01)
        equal = [i for i in range(istart,iend+1)]
        ind_oms_equal.append(equal)
    istart = int(0.5*(ind_om[-2]+ind_om[-1])+0.51)
    equal = [i for i in range(istart,ind_om[-1]+1)]
    ind_oms_equal.append(equal)

    oms_equal=[]
    for ind in ind_oms_equal:
        oms_equal.append( array([om[i] for i in ind]) )
    
    return (ind_om,oms_equal)




def Spline_Real(Arr,mesh_log,mesh_reg):
	#Arr = 1D real array
	#Arr = Arr[:10]
	#mesh_log = mesh_log[:10]
	#print Arr, mesh_log

	Re = interpolate.UnivariateSpline(mesh_log,Arr,s=0)
	Arr_large = Re(mesh_reg)
	#Nmesh = len(mesh_log)
	#print mesh_log[Nmesh/2]
	#print Re(0.05)
	#sys.exit()
	return Arr_large


def Spline_Complex(Arr,mesh_log,mesh_reg):
	#Arr = 1D complex array
	Re = interpolate.UnivariateSpline(mesh_log,real(Arr),s=0)
	Im = interpolate.UnivariateSpline(mesh_log,imag(Arr),s=0)
	Arr_large = Re(mesh_reg)+1j*Im(mesh_reg)
	return Arr_large


def	Gw_to_Gtau(Gw ,oms_f_log,oms_f,tau_mesh,beta):
	if len(shape(Gw)) <= 1:
		Gw_large = Spline_Complex( Gw,oms_f_log,oms_f)
		Gtau = zeros((len(tau_mesh)),float)
		corr2 = -0.5*beta
		for itau,tau in enumerate(tau_mesh):
			phase = exp(-1j*oms_f*tau) 
			om_inv = 1/(1j*oms_f)
			G_error = dot(phase, Gw_large)
			corr1 = dot(phase,om_inv)
			Gtau[itau] = 2*real(G_error - corr1) + corr2
		Gtau *= 1./beta
		return Gtau
	
	else:
		t1 = time.clock()
		M = len(Gw[0,0])
		ist, dict_ist = index_double(M)
		N_ist = len(ist)
		Gw_large = zeros((len(oms_f),M,M),complex)
		
		for i,(i1,i2) in enumerate(ist):
			Gw_large[:,i1,i2] = Spline_Complex( Gw[:,i1,i2],oms_f_log,oms_f)

		# sum_{iw_n 0 to infty} 1/iw exp(-1j*iw*tau)
		phase = exp(-1j*dot(tau_mesh[newaxis].T,oms_f[newaxis]))
		G_error = tensordot(phase,Gw_large,([1],[0]))
		om_inv = tensordot(1/(1j*oms_f)[newaxis].T,\
				identity(M)[newaxis],axes=([1],[0]))
		corr1 = tensordot(phase,om_inv,([1],[0]))
		corr2 = -0.5*beta*identity(M)[newaxis]
		G2 = 2*real(G_error-corr1)+corr2
		Gtau = reshape(G2,(len(G2),M*M))
		Gtau *= 1./beta
		t2 = time.clock()
		#print "time for Gw_to_Gtau = %2.3f" %(t2-t1)

		return Gtau

def cot(x): return 1./tan(x)

def corr(x):
	"""
	correction function (imaginary part) 
	when transforming from tau to freq
	"""
	#Corr_im = 0.0 											#no correction
	#Corr_im = - 1j*x/12. 							#first order
	#Corr_im = 1j*(- x/12. -x**3/720*)  #third order
	if x == 0.: Corr_im = 0.0
	else:
		Corr_im = 1j*((0.5*cot(0.5*x))-1/x) #almost exact
	
	return Corr_im

def Gtau_to_Gw(Gtau, tau_mesh, tau_large, oms_f_log):
	dtau = tau_large[1]-tau_large[0]
	if len(shape(Gtau)) <= 1:
		Gtau_large = Spline_Real(Gtau,tau_mesh, tau_large)
		Gw = zeros((len(oms_f_log)),complex)
		for iw, w in enumerate(oms_f_log):
			x = w*dtau
			phase = exp(1j*w*tau_large)
			Gw_error = dtau*dot(phase, Gtau_large)
			Corr_im = dtau*corr(x)
			Gw[iw] = Gw_error + (Corr_im)
		#Corr_re = .0
		Corr_re = -real(Gw[-1])
		Gw = Gw + Corr_re
		return Gw

	else:
		M = int(sqrt(len(Gtau[0]))+0.1)
		ist, dict_ist = index_double(M)
		Gtau_large = zeros((len(tau_large),M,M),float)
		for i, (i1,i2) in enumerate(ist):
			Gtau_large[:,i1,i2] = Spline_Real(Gtau[:,i],tau_mesh, tau_large)
		Gw = zeros((len(oms_f_log),M,M),complex)
		for iw, w in enumerate(oms_f_log):
			x = w*dtau
			phase = exp(1j*w*tau_large)
			Gw_error = dtau*tensordot(phase, Gtau_large, axes= ([0],[0]))
			Corr_im = dtau*corr(x)
			Gw[iw] = Gw_error + (Corr_im)*identity(M)
		#Corr_re = 0.0
		Corr_re = -real(Gw[-1])
		Gw = Gw + Corr_re
		return Gw

def Gtau_to_Gw2(Gtau, tau_mesh,tau_large, oms_f_log):
	M = int(sqrt(len(Gtau[0]))+0.1)
	ist, dict_ist = index_double(M)
	Gtau_large = zeros((len(tau_large),M,M),float)
	for i, (i1,i2) in enumerate(ist):
		Gtau_large[:,i1,i2] = Spline_Real(Gtau[:,i],tau_mesh, tau_large)
	return Atau_to_Aw(Gtau_large,tau_large,oms_f_log)



def Atau_to_Aw(Atau, tau_mesh, oms_f_log,option=0):
	exp_wt = exp(+1j*dot(oms_f_log[newaxis].T,tau_mesh[newaxis]))
	if not option:
		Aw = FT_matrix(tau_mesh,exp_wt,Atau)
	else:
		dtau = tau_mesh[1]-tau_mesh[0]
		Aw = tensordot(exp_wt,Atau,axes=([1],[0]))*dtau
	return Aw


def Gtau_to_Pw(Gtau, tau_mesh,tau_large,oms_b_log):
	"""
	P(t) = G(t)*G(-t) = -G(t)*G(beta-t)
			 = -G(t)*G(t)[::-1]
	"""

	if len(shape(Gtau)) <= 1:
		Gtau_large = Spline_Real(Gtau,tau_mesh, tau_large)
		Gtau_large_inv = Gtau_large[::-1] #G(-tau)
		Ptau_large = -2.*Gtau_large*Gtau_large_inv
		option = 0 #0: simps, 1: naive integration
		Pw = real(Atau_to_Aw(Ptau_large,tau_large,oms_b_log,option))
		return Pw

	else:
		t1 = time.clock()
		M = int(sqrt(len(Gtau[0]))+0.1)
		ist, dict_ist = index_double(M)
		
		Gtau_large = zeros((len(tau_large),M*M),float)
		for i, (i1,i2) in enumerate(ist):
			Gtau_large[:,i] = Spline_Real(Gtau[:,i],tau_mesh, tau_large)
		
		Nist = len(ist)
		Ntau = len(tau_large)
		Ptau_large = zeros((Ntau, Nist, Nist),float)
		Gtau_large_inv = Gtau_large[::-1] #G(-tau)
		for p,(i,l) in enumerate(ist):
			#print p
			for q,(k,j) in enumerate(ist): #note the order kj, not jk
				#P(ijkl) = P(il)(kj) 
				#	=  2*G(ki)(tau)G(lj)(-tau) 
				# = -2*G(ki)(tau)G(lj)(beta-tau)
				# = -2*G(ki)(tau)G(lj)[::-1]
				p1 = dict_ist[k,i]
				q1 = dict_ist[l,j]
				Ptau_large[:,p,q] = -2.*Gtau_large[:,p1]*Gtau_large_inv[:,q1]
		t2 = time.clock()
		#print "t Gtau_to_Ptau = %2.2f sec" %(t2-t1) 
		option = 0 #0: simps, 1: naive integration
		Pw = real(Atau_to_Aw(Ptau_large,tau_large,oms_b_log,option))
		#Pw = Atau_to_Aw(Ptau_large,tau_large,oms_b_log,option)
		t3 = time.clock()
		#print "t Ptau_to_Pw = %2.2f sec" %(t3-t2) 
		return Pw

def FT_matrix(tau,phase,A):
	"""
	Fourier transformation:
			F(iw) = int dtau exp(i*iw*tau) F(tau)
	integration done by simpson's rule
	shape(A) = (Ntau,p,q)
	shape(phase) = (Nw,Ntau)
	"""
	if len(tau)!=len(A):
		print "the # of samplings of X(%2d)\
		and Y(%2d) are different" %(len(tau), len(A))
		sys.exit()

	N = len(tau)
	res = 0.0
	#print N
	if N%2 != 0:
		t1 = tau[1:-1][::2]-tau[:-2][::2]
		t2 = tau[2:][::2]-tau[1:-1][::2]
		T  = (t1+t2)/6.
		T1 = (2.-t2/t1)*T
		T2 = (t1+t2)**2/(t1*t2)*T
		T3 = (2.-t1/t2)*T
		
		A1 = A[:-2][::2]
		A2 = A[1:-1][::2]
		A3 = A[2:][::2]

		TP1 = T1[newaxis]*phase[:,:-2][:,::2] #t_m * exp(iwn*tm)
		TP2 = T2[newaxis]*phase[:,1:-1][:,::2] #t_m * exp(iwn*tm)
		TP3 = T3[newaxis]*phase[:,2:][:,::2] #t_m * exp(iwn*tm)

		res += tensordot(TP1,A[:-2][::2],axes=([1],[0]))
		res += tensordot(TP2,A[1:-1][::2],axes=([1],[0]))
		res += tensordot(TP3,A[2:][::2],axes=([1],[0]))

	else:
		print "Nmesh for Simpsons is even"
		t1 = tau[1:-1][::2]-tau[:-2][::2]
		t2 = tau[2:][::2]-tau[1:-1][::2]
		#print t1
		#print t2
		T  = (t1+t2)/6.
		T1 = (2.-t2/t1)*T
		T2 = (t1+t2)**2/(t1*t2)*T
		T3 = (2.-t1/t2)*T
		
		A1 = A[:-2][::2]
		A2 = A[1:-1][::2]
		A3 = A[2:][::2]
		
		TP1 = T1[newaxis]*phase[:,:-2][:,::2] #t_m * exp(iwn*tm)
		TP2 = T2[newaxis]*phase[:,1:-1][:,::2] #t_m * exp(iwn*tm)
		TP3 = T3[newaxis]*phase[:,2:][:,::2] #t_m * exp(iwn*tm)

		res += tensordot(TP1,A[:-2][::2],axes=([1],[0]))
		res += tensordot(TP2,A[1:-1][::2],axes=([1],[0]))
		res += tensordot(TP3,A[2:][::2],axes=([1],[0]))
		
		#res += 0.5*(tau[-1]-tau[-2])*(P[-1]+P[-2])
	#extrapolate the last term to reduce the error
	res[-1] = res[-2]+(res[-2]-res[-3]) 
	return res



def Create_om_mesh(beta, Nomega, nom, nom_tail):
	oms_f = [] #fermion frequency
	for i in range(Nomega):
		oms_f.append((2*i+1)*pi/beta)
	oms_f = array(oms_f)

	oms_b = [] #boson frequency
	for i in range(Nomega):
		s = (2*i)*pi/beta
		oms_b.append(s)
	oms_b = array(oms_b)

	oms_ind, oms_equal_f = log_mesh(oms_f,nom,nom_tail)

	oms_f_log = []
	for i in oms_ind:
		oms_f_log.append((2*i+1)*pi/beta)
	oms_f_log = array(oms_f_log)

	oms_b_log = []
	for i in oms_ind:
		s = (2*i)*pi/beta
		oms_b_log.append(s)
	oms_b_log = array(oms_b_log)
	return oms_f, oms_f_log, oms_b, oms_b_log, oms_equal_f, oms_ind

	

def Create_tau_mesh(beta,Ntau,ntau,ntau_tail): 
	#mesh is dense near 0 and beta
	deltau = beta/Ntau
	#tau_large = arange(0,beta, deltau)
	tau_large = linspace(0,beta,Ntau+1) #including beta
	
	N_beta1 = Ntau/2
	
	#0 <= tau <= beta/2
	tau1_org = linspace(0.0, 0.5*beta,N_beta1+1)
	del_ntau_reg = 1
	tau1_ind, tau1_equal = log_mesh( tau1_org, ntau, ntau_tail,del_ntau_reg) #logarithmic defined
	#tau1_ind = tau1_ind[::5] #if this is off, we get nan from Spline_Complex.....why?


	#beta/2 < to < beta
	#tau2_ind = (Ntau-1 - array(tau1_ind[::-1])).tolist()[1:]
	tau2_ind = (Ntau  - array(tau1_ind[::-1])).tolist()
	
	tau_ind = tau1_ind[:-1]+tau2_ind
	#print tau_ind
	#print len(tau_ind)
		
	tau_mesh = array([tau_large[i] for i in tau_ind])
	#print tau_mesh[-1] #tau_2nd = -tau_1st[::-1]+beta
	#tau_mesh0 = tau_1st.tolist()+tau_2nd.tolist()[1:]
	#print len(tau_mesh0)
	return tau_mesh,tau_ind ,tau_large


def Create_tau_mesh2(beta,Ntau,ntau,beta_safe=8.): 
	"""
	beta = 1/T , 
	beta_safe = tau range up to which needed to be treated carefully.
	Ntau = beta_safe mesh 
	ntau = other mesh
	The total length going to be therefore 2(Ntau+ntau)-1
	"""
	deltau = beta_safe/Ntau
	
	N_beta1 = int(0.5*beta/deltau)
	
	tau1_org = linspace(0.0, 0.5*beta,N_beta1+1)
	#0 <= tau <= beta/2
	tau1_ind, tau1_equal = log_mesh( tau1_org, Ntau, ntau) #taularge_ind
	tau1_large = tau1_org[tau1_ind]
	
	#beta/2 < to < beta
	tau2_large = beta-tau1_large[::-1]
	tau_large = array( tau1_large.tolist()+tau2_large[1:].tolist())

	
	beta_safe_small = 1.5

	Nsmall = int(beta_safe_small/deltau)
	nsmall = ntau/4
	#0 <= tau <= beta/2
	tau1_ind, tau1_equal = log_mesh( tau1_org, Nsmall, nsmall) #tau_small_ind
	tau1_small = tau1_org[tau1_ind]
	
	#beta/2 < to < beta
	tau2_small = beta-tau1_small[::-1]
	tau_small = array( tau1_small.tolist()+tau2_small[1:].tolist())

	#print tau_large
	#print tau_small
	#print len(tau_large)
	#print 2*(Ntau+ntau)-1
	#print len(tau_small)
	#print 2*(Nsmall+nsmall)-1

	return tau_small,tau_large
	
def Ww_to_Wtau(delWF, tau_mesh, oms_b_log, oms_b,beta):
	MM = len(delWF[0])
	wt = dot(tau_mesh[newaxis].T,oms_b[newaxis])
	cos_wt = 2*cos(wt)
	cos_wt[:,0] -= 1. #for vn=0, it must 1, not 2.

	WFtau = zeros((len(tau_mesh),MM,MM),float)
	for p in range(MM):
		for q in range(MM):
			WFlarge = Spline_Real(delWF[:,p,q], oms_b_log, oms_b)
			WFtau[:,p,q] = 1/beta*dot(cos_wt,WFlarge)
	return WFtau

def Ww_to_Wtau_loc(delWF, tau_mesh, oms_b_log, oms_b,beta):
	wt = dot(tau_mesh[newaxis].T,oms_b[newaxis])
	cos_wt = 2*cos(wt)
	cos_wt[:,0] -= 1. #for vn=0, it must 1, not 2.

	WFlarge = Spline_Real(delWF, oms_b_log, oms_b)
	WFtau = 1/beta*dot(cos_wt,WFlarge)
	return WFtau



if __name__ =="__name__":
	beta = 30.
	Ntau = 2000
	tau_safe = 10.
	ntau = 50
	tau_mesh, tau_large = Create_tau_mesh2(beta,Ntau,ntau,tau_safe)
