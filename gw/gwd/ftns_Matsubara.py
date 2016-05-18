#popen3 -> subprocess.Popen for python ver 2.6
#all codes for matsubara sum
#DMFT codes

import subprocess
import sys
import os
import shutil
import re
from scipy import *
from scipy import weave
from scipy import linalg
from scipy import optimize
from scipy import interpolate
import copy
import time




def create_log_mesh(sigdata, nom, ntail_):
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
    om = sigdata[0]
    
    istart = min(nom, len(om))
    ntail = min(ntail_, len(om)-istart)
        
    istart = min(nom,len(om))
    ntail = min(ntail, len(om)-istart)

    ind_om=[]
    alpha = log((len(om)-1.)/istart)/(ntail-1.)
    for i in range(istart):
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
    #print ind_oms_equal
    #print oms_equal
    
    ssigdata = zeros( (shape(sigdata)[0], len(ind_om)), dtype=float )
    for i in range(len(ind_om)):
        ssigdata[:,i] = sigdata[:,ind_om[i]]

    return (ssigdata,oms_equal,ind_om)

def MatsubaraSum(mu,Ew,om,beta):
    sm=0.0
    for iw,w in enumerate(om): sm += real(1/(w*1j+mu-Ew[iw]))
    sm = 2.*sm/beta+0.5
    # Sommerfeld expansion to correct for finite sum
    # We need to add sum_{iw>Omega0} 1/(iw-E0)
    # which is Integrate[f(x) * Omega0/(Omega0**2+(x-E0)**2)/pi,{x,-oo,oo}]
    E0 = Ew[-1].real
    Omega0 = om[-1]+pi/beta
    sm -= 1./pi*arctan2(E0,Omega0)
    sm += pi/(3*beta**2)*Omega0*E0/(Omega0**2+E0**2)**2
    sm -= 7*pi**3/(15*beta**4)*Omega0*E0*(Omega0**2-E0**2)/(Omega0**2+E0**2)**4
    return sm


def LogMatsubaraSum(mu,Ew,oms,oms_equal,beta):
    code="""
       #include <complex>
       using namespace std;
       double sm=0.0;
       double cmu=mu;
       complex<double> i(0.0,1.0);
       for (int iband=0; iband<Ec.size(); iband++){
           complex<double> cEc = Ec(iband);
           for (int j=0; j<equal.size(); j++){
               double w = equal(j);
               complex<double> G = 1.0/(i*w+cmu-cEc);
               sm += G.real();
           }
       }
       return_val = sm;
    """
    PYTHON=False
    if (PYTHON):
        sm=0.0
        for iw in range(len(oms)):
            for Eb in Ew[iw]:
                for w in oms_equal[iw]:
                    sm += real(1/(w*1j+mu-Eb))
    else:
        sm=0.0
        for iw in range(len(oms)):
            equal = oms_equal[iw]
            Ec = Ew[iw]
            sm += weave.inline(code, ['equal', 'mu', 'Ec'], type_converters=weave.converters.blitz, compiler = 'gcc')
        
    sm *= 2./beta
    for Eb in Ew[-1]:
        # Sommerfeld expansion to correct for finite sum
        # We need to add sum_{iw>Omega0} 1/(iw-E0)
        # which is Integrate[f(x) * Omega0/(Omega0**2+(x-E0)**2)/pi,{x,-oo,oo}]
        E0 = Eb.real-mu
        Omega0 = oms[-1]+pi/beta
        sm += 0.5  # because of 1/omega
        sm -= 1./pi*arctan2(E0,Omega0)
        sm += pi/(3*beta**2)*Omega0*E0/(Omega0**2+E0**2)**2
        sm -= 7*pi**3/(15*beta**4)*Omega0*E0*(Omega0**2-E0**2)/(Omega0**2+E0**2)**4
    return sm


def Cmp_eks(lsigdata,s_oo,Edc,Ener,TProj):

    N = len(Ener)
    eks=zeros((len(lsigdata[0]),N),dtype=complex)
    for iw,w in enumerate(lsigdata[0]):
        iomega = w*1j
        Sigm = (lsigdata[1,iw]+lsigdata[2,iw]*1j)+s_oo[0]
        Sig = identity(2)*(Sigm-Edc)
        ES = zeros((N,N),dtype=complex)
        for i in range(N): ES[i,i] = Ener[i]
        ES += dot(TProj.T , dot( Sig , TProj))
        eks[iw] = linalg.eigvals(ES)
    return eks

def Cmp_eVks(lsigdata,s_oo,Edc,Ener,TProj):
    
    N = len(Ener)
    eks=zeros((len(lsigdata[0]),N),dtype=complex)
    Ars=zeros((len(lsigdata[0]),N,N),dtype=complex)
    for iw,w in enumerate(lsigdata[0]):
        iomega = w*1j
        if iw!=N-1:
            Sigm = (lsigdata[1,iw]+lsigdata[2,iw]*1j)+s_oo[0]
        else:
            Sigm = lsigdata[1,iw]+s_oo[0]
            
        Sig = identity(2)*(Sigm-Edc)
        ES = zeros((N,N),dtype=complex)
        for i in range(N): ES[i,i] = Ener[i]
        ES += dot(TProj.T , dot( Sig , TProj))

        if iw!=N-1:
            (es, Ar) = linalg.eig(ES)
        else:
            (es, Ar) = linalg.eigh(ES)
        
        eks[iw] = es
        Ars[iw,:,:] = Ar
    return (eks,Ars)
    
def Cmp_mu(N0,a,b,eks,oms,oms_equal,beta):

    def Density(mu,N0,eks,oms,oms_equal,beta):
        return LogMatsubaraSum(mu,eks,oms,oms_equal,beta)-N0

    return optimize.brentq(Density,a,b,args=(N0,eks,oms,oms_equal,beta))
    

def Cmp_DensityMatrix(eks,Ars,mu,oms,oms_equal,beta):
	
	code1="""
	  using namespace std;
	  complex<double> i(0,1);
	  complex<double> sm=0.0;
	  for (int jw=0; jw<equal.size(); jw++){
	     double w = equal(jw);
	     sm += 1.0/(w*i+mu-Eb);
	  }
	  return_val = sm;
	"""
	code2="""
	  using namespace std;
	  for (int i1=0; i1<N; i1++){
	     for (int i2=0; i2<N; i2++){
	        for (int l=0; l<N; l++){
	           Glc(i1,i2) += Ar(i1,l)*gii(l)*Al(l,i2);
	        }
	     }
	  }
	"""
	N = len(eks[0])
	Nc0 = zeros((N,N),dtype=complex)
	for iw in range(len(oms)):
		Ar = Ars[iw]
		Al = linalg.inv(Ar)
		gii = zeros((N,),dtype=complex)
		
		for l in range(N):
			Eb = eks[iw,l]
			equal = oms_equal[iw]
			#sm = weave.inline(code1, ['equal','mu','Eb'], type_converters=weave.converters.blitz, compiler = 'gcc')
			#sm = 0.0
			#for w in equal:
			#	sm += 1.0/(1j*w+mu-Eb)
			sm = sum(1./(1j*equal+mu-Eb))
			
			
			if iw==len(oms)-1:
				# Sommerfeld expansion to correct for finite sum
				# We need to add sum_{iw>Omega0} 1/(iw-E0)
				# which is Integrate[f(x) * Omega0/(Omega0**2+(x-E0)**2)/pi,{x,-oo,oo}]
				E0 = Eb.real-mu
				Omega0 = oms[-1]+pi/beta
				sm_oo = 0.5  # because of 1/omega
				sm_oo -= 1./pi*arctan2(E0,Omega0)
				sm_oo += pi/(3*beta**2)*Omega0*E0/(Omega0**2+E0**2)**2
				sm_oo -= 7*pi**3/(15*beta**4)*Omega0*E0*(Omega0**2-E0**2)/(Omega0**2+E0**2)**4
				sm += 0.5*beta*sm_oo
			gii[l] = sm
		
		Glc = zeros((N,N),dtype=complex)
		#weave.inline(code2, ['Glc','N','Ar','Al','gii'], type_converters=weave.converters.blitz, compiler = 'gcc')
		Glc = dot(Ar,dot(diag(gii),Al))
		Nc0 += (Glc+transpose(conjugate(Glc)))/beta
		
	return real(Nc0)

	
	
