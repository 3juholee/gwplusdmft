from PyQuante import *
from PyQuante.LA2 import *
from PyQuante.Ints import get2JmK,getbasis,getints,getJ,getK, fetch_jints,fetch_kints
from scipy import *
from scipy import linalg
from PyQuante.MG2 import MG2 as MolecularGrid

def savedata(mesh,data,output):
	if abs(imag(data[0]))<1e-10:
		data2 = zeros((len(mesh),2),float)
		data2[:,0] = mesh
		data2[:,1] = data
	else:
		data2 = zeros((len(mesh),2),float)
		data2[:,0] = mesh
		data2[:,1] = real(data)
		data2[:,2] = real(data)
	savetxt(output,data2)



def index_double(M):
	ist = []
	#index set
	for i in range(M):
		for j in range(M):
			ist.append( (i,j) )
	#dict of ist, used in cmp_Coulomb in h2_Coulomb.py
	dict_ist = zeros((M,M),int)
	for l, (i,j) in enumerate(ist):
		dict_ist[i,j]=l
	return ist, dict_ist


def four_indice(T):
	#from 2indice trans_matrix T to 4indice trans_matrix T2
	N,M = shape(T)
	ist1, dict_ist1 = index_double(N)
	ist2, dict_ist2 = index_double(M)

	T2 = zeros((len(ist1),len(ist2)),float)
	for p,(i,j) in enumerate(ist1):
		for q,(a,b) in enumerate(ist2):
			T2[p,q] = T[i,a]*T[j,b]
	return T2


def trans_2matrix(H,T):
	# T = transformation matrix ij -> ab (therefore T_{i,a})
	# H = 2indice matrix represented as H_{ij}
	# H' = T^{+} * H * T
	return dot(T.T, dot(H, T))


def trans_4matrix(HH,T):
	# T = transformation matrix ij -> ab (therefore T_{i,a})
	# HH = four indice matrix represented as HH_{pq}, p=(il),q=(jk)
	T2 = four_indice(T)
	return dot(T2.T, dot(HH, T2))
	
def basis_transform(H0,UH,UF, Evec):
	H02 = trans_2matrix(H0,Evec)
	UH2 = trans_4matrix(UH,Evec)
	UF2 = trans_4matrix(UF,Evec)
	return H02, UH2, UF2

class gtos:
	def __init__(self,natom, R, basis_input="ccpvtz",LDA=False):
		if basis_input =='2dp':
			basis_data='6-311g++(2d,2p)'
		elif basis_input =='3dp':
			basis_data='6-311g++(3d,3p)'
		elif basis_input =='3df':
			basis_data='6-311g++(3df,3pd)'
		elif basis_input =='ccpvtz':
			basis_data="ccpvtz"
		elif basis_input =='631ss':
			basis_data =  '6-31g**++'
		elif basis_input =='ccpvdz':
			basis_data="ccpvdz"
		elif basis_input =='p6311ss':
			basis_data =  '6-311g**'
		elif basis_input =='juho_':
			basis_data =  'juho_'
		elif basis_input =='1s':
			basis_data =  'sto-6g'

		atoms = Molecule('H2',atomlist=[(natom,(0,0,0)),(natom,(R,0,0))])
		self.bfs  = getbasis(atoms,basis_data)
		self.S, self.h, self.Ints = getints(self.bfs, atoms)

		M = len(self.S)
		
		ist, dict_ist = index_double(M)
		
		M2 = len(ist)
		UH_g = zeros((M2,M2),float)
		UF_g = zeros((M2,M2),float)
		for il,(i,j) in enumerate(ist):
			UH_g[il] = fetch_jints(self.Ints,i,j,M)
			UF_g[il] = fetch_kints(self.Ints,i,j,M)
		
		#Diagonalize GTO: H2+ basis
		self.eival, self.eivec = geigh(self.h,self.S) 
		
		#h0 within H2+ basis
		h0 = trans_2matrix(self.h,self.eivec)
		#UH and UF within H2+ basis:  T1[ad,il]*UH_g(ijkl)*T1.T[jk,bc]
		UH = trans_4matrix(UH_g, self.eivec)
		UF = trans_4matrix(UF_g, self.eivec)
		self.inputs = (2*h0,2*UH,2*UF)

		if LDA:
			self.gr = gto_lda_inputs(self.bfs,atoms)



def HartreeF(natom, R, H0, UH, UF, Nitt = 20 ):
	#this only applies to symmetric diatomic molecules
	nocc = 2.
	M = len(H0)
	#print "dimension = ", M
	ist, dict_ist = index_double(M)
	Nt2 = zeros((M,M),float)
	for i in range(natom):
		Nt2[i,i] = nocc
	Nc = reshape(Nt2,(M*M,))
	print "Ntot = ", trace(Nt2)
	
	p_Ener = 0.0
	V2 = zeros((M,M),float)
	for itt in range(Nitt):
		VHa = dot(UH, Nc)     # Hartree
		VFx = -0.5*dot(UF,Nc) # Exact exchange
		#for i, (i1,i2) in enumerate(ist): V2[i1,i2]=VHa[i]+VFx[i]
		V2 = reshape(VHa+VFx,(M,M))
		(Ener,Evec) = linalg.eigh(H0+V2)

		Nt2 = zeros((M,M),float)
		for i in range(natom):
			ei = Evec[:,i][newaxis]
			Nt2 += dot(ei.T,ei)*nocc
		Nc = reshape(Nt2,(M*M,))
		#print "Ntot = ", trace(Nt2)
		
		Eone = trace2(H0,Nt2)
		EHa = 0.5*sum(Nc*VHa)
		EFx = 0.5*sum(Nc*VFx)
		EHF = Eone + EHa + EFx + 2./R*natom**2
		#EHF = Eone + EHa + EFx
		diff = EHF-p_Ener
		#print R, "EHF = %2.7f, %2.7f , itt = %d" %(EHF/2, diff,itt)
		if abs(EHF-p_Ener)<1e-5: 
			print "HF calculation is done"
			print "Eone	= ", Eone
			print "EHa	= ", EHa
			print "EFx	= ", EFx
			print "Eion	= ", 2./R*natom**2
			print "e0 = ", H0[0,0]
			print R, "EHF = %2.7f Ry (%2.7f a.u.) " %(EHF, EHF/2)
			print "%2.2f   %2.7f" %(R, EHF)
			print "Ntot = ", trace(Nt2)
			print "HF orbitals = ", Ener
		#	print Nt2
			#sys.exit()
			break
		if itt == Nitt-1:
			print "It does not converge at R = ", R
		
		p_Ener = EHF
	return Ener, Evec, Nt2
		

def gto_lda_inputs(bfs,atoms):
	grid_nrad = 32 # number of radial shells per atom
	grid_fineness = 1 #option
	gr = MolecularGrid(atoms,grid_nrad,grid_fineness) 
	gr.set_bf_amps(bfs)
	return gr

def xc_gto(Nt2, U, gr,(FX,FC)=(1.0,1.0)):
	D = dot(U,dot(Nt2/2.,U.T)) #D = S*(N_g)*S
	gr.setdens(D)
	Exc, Vxc_g =get_X_C(gr,FX,FC)
	Vxc = dot(U.T,dot(Vxc_g,U)) # V = UT* V_g *U
	return 2*Vxc, 2*Exc, gr #Ry


def xc_gto_local(Nf,TPOv,U,gr,(FX,FC)):
	#for DC for LDA+DMFT
	M = len(U)
	Ploc = zeros((M,M),float)
	R = TPOv[0]
	for i in range(M):
		for j in range(M):
			Ploc[i,j] = R[i]*R[j]
	Nloc = Nf*Ploc
	Vxc, Exc_loc, gr = xc_gto(Nloc, U, gr, (FX,FC))
	Vxc_loc = dot(R, dot(Vxc, R))
	print "local Vxc and Exc = ", Vxc_loc,Exc_loc
	#sys.exit()
	return Vxc_loc, Exc_loc #already in Ry


def LDA_gto(natom,R, H0, UH, UF, gr, U, Nitt = 20 ):
	#gr: object grid from PyQuante
	#U is such that |i> = U_ki |g_k>
	nocc = 2.
	M = len(H0)
	ist, dict_ist = index_double(M)
	Nt2 = zeros((M,M),float)
	for i in range(natom):
		Nt2[i,i] = nocc
	Nc = reshape(Nt2,(M*M,))
	M = len(H0)
	p_Ener = 0.0
	V2 = zeros((M,M),float)
	FX = 1.0
	FC = 1.0
	for itt in range(Nitt):
		VHa = dot(UH, Nc)     # Hartree
		VFx = -0.5*dot(UF,Nc) # Exact exchange
		Vxc, Exc, gr = xc_gto(Nt2, U, gr,(FX,FC))
		for i, (i1,i2) in enumerate(ist): 
			V2[i1,i2]=VHa[i]+VFx[i]*(1.-FX)
		V2 += Vxc

		(Ener,Evec) = linalg.eigh(H0+V2)

		Nt2 = zeros((M,M),float)
		for i in range(natom):
			ei = Evec[:,i][newaxis]
			Nt2 += dot(ei.T,ei)*nocc
		Nc = reshape(Nt2,(M*M,))
		#print "Nt2 = ", trace(Nt2)
		
		Eone = sum([H0[i,i]*real(Nt2[i,i]) for i in range(len(H0))])
		EHa = 0.5*sum(Nc*VHa)
		EFx = 0.5*sum(Nc*VFx)
		E_lda = Eone + EHa + EFx*(1.-FX) + Exc + 2./R*natom**2
		print R, "E_lda = ", E_lda, "Exc = ", Exc

		if abs(E_lda-p_Ener)<5e-6: 
			print "LDA calculation is done"
			print R, "E_LDA = ", E_lda
			#sys.exit()
			break
		
		p_Ener = E_lda
	return Ener,Evec
		
def bfs_filter(natom, atoms, basis_data):
	symlist = {
	    'S' : [1],
	    'P' : [1,0,0],
	    #'P' : [1,1,1],
	    'D' : [1,0,0,1,0,1],
	    'F' : [0,0,0,0,0,0,0,0,0,0],
	    }
	import PyQuante.Basis.Tools as tools
	basis_info = tools.get_basis_data(basis_data)[natom]
	bfs_sym = []
	for sym in basis_info:
		bfs_sym += symlist[sym[0]]
	bfs_sym = 2*bfs_sym
	bfs0 = getbasis(atoms,basis_data)
	M = len(bfs0)
	bfs = []
	for il in range(M):
		if bfs_sym[il]==1:
			bfs.append(bfs0[il])
	return bfs


if __name__ == "__main__":

	R = 1.388
	#h0, UH, UF = inputs_gto(R,'ccpvtz')
	h0, UH, UF = inputs_gto(R)
	HartreeF(R, h0,UH, UF)

