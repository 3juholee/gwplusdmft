#popen3 -> subprocess.Popen for python ver 2.6

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
from cix import *
import copy
import time


class IMP:
    def __init__(self,nom,beta,M,nomD=None,RUUN=True,wbroad=0.03,kbroad=0.15):
        self.checkCausality = False
        self.broad = True
        
        self.params = {"exe":   ["./ctqmc",          "# Path to executable"],
                       "workdir":["imp",             "# Working directory for impurity"],
                       "Delta": ["Delta.dat",        "# Input bath function hybridization"],
                       "Sig":   ["Sig.out",          "# Output self-energy"],
                       "cix":   ["one_band.imp",     "# Input file with atomic state"],
                       "U":     [0,                  "# Coulomb repulsion (F0)"],
                       "mu":    [0,                  "# Chemical potential"],
                       "beta":  [100,                "# Inverse temperature"],
                       "M" :    [2e6,               "# Number of Monte Carlo steps"],
                       "nom":   [100,                "# number of Matsubara frequency points to sample"],
                       "nomD":  [0,                 "# number of Matsubara points using the Dyson Equation"],
                       "aom":   [5,                  "# number of frequency points to determin high frequency tail"],
                       "tsample":[300,               "# how often to record the measurements" ],
                       "maxNoise":[1e100,            "# maximum allowed noise is large in simple run"]}
        self.wbroad=wbroad                     # Broadening of the hybridization function
        self.kbroad=kbroad                     # Broadening of the hybridization function

        self.dir = self.params['workdir'][0]
        self.q_exe = self.params['exe'][0]
        self.Signame = self.params['Sig'][0]
        self.PARAMS = 'PARAMS'
        self.fh_info = sys.stdout
        self.mpi_prefix = ''
        mpifile = 'mpi_prefix.dat'
        if os.path.isfile(mpifile):
            self.mpi_prefix = open(mpifile, 'r').next().strip()
            print "DmftEnvironment: mpi_prefix.dat exists -- running in parallel mode."
            print "  ", self.mpi_prefix
        else:
            print "DmftEnvironment: mpi_prefix.dat does not exist -- running in single-processor mode."
        
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        
        # creating impurity cix file
        f = open(self.dir+'/'+self.params['cix'][0], 'w')
        print >> f, one_band_cix
        f.close()

        root = os.getenv('WIEN_DMFT_ROOT')
        shutil.copy2(root+'/ctqmc', self.dir)
        shutil.copy2(root+'/broad', self.dir)
        
        self.params['M'][0]=M
        self.params['nom'][0]=nom
        if nomD:
            self.params['nomD'][0]=nomD
        self.params['beta'][0]=beta
        
    def Run(self,U,mu_QMC,omega_large,ind_om,Delta):

        # Interpolating Delta on entire Matsubara mesh
        Dreal = interpolate.UnivariateSpline(Delta[:,0],Delta[:,1],s=0)
        Dimag = interpolate.UnivariateSpline(Delta[:,0],Delta[:,2],s=0)
        Dr = [Dreal(x) for x in omega_large]
        Di = [Dimag(x) for x in omega_large]
        Delta2 = transpose(array([omega_large,Dr,Di]))
        savetxt(self.dir+'/Delta.dat',Delta2)
        
        # Creates input file (PARAMS) for CT-QMC solver
        self.params['U'][0]=U
        self.params['mu'][0]=mu_QMC
        f = open(self.dir+'/PARAMS', 'w')
        print >> f, '# Input file for continuous time quantum Monte Carlo'
        for p in self.params.keys():
            print >> f, p, self.params[p][0], '\t', self.params[p][1]
        f.close()
       
        # Below we execute ctqmc
        cmd = 'cd '+self.dir+'; '+self.mpi_prefix+' '+self.q_exe+' '+self.PARAMS+' > nohup_imp.out 2>&1 '
        print cmd
        print 'Running ---- ctqmc -----'

        RUUN = True
        if (RUUN):
            p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, close_fds=True)
            stdin, stdout, stderr = (p.stdin,p.stdout,p.stderr)
            print >> self.fh_info, stdout.read(), stderr.read()

        
        # Copying to Sig.outb
        Signame = self.dir+'/'+self.Signame
        shutil.copy2(Signame, Signame+'b')

        nf=0.; TrSigmaG=0.
        first_line = open(Signame,'r').readline()
        m=re.search('nf=(\d*\.\d*)',first_line)
        if m is not None:
            nf = float(m.group(1))
        m=re.search('TrSigmaG=(\d*\.\d*)',first_line)
        if m is not None:
            TrSigmaG = float(m.group(1))


        Sig = loadtxt(Signame)
        if self.broad:
            
            if self.checkCausality:
                for i in range(shape(Sig)[1]/2):
                    for iw in range(len(Sig)):
                        if Sig[iw,2+2*i]>0: Sig[iw,2+2*i]=0.0
            
            savetxt(Signame+'w',Sig)
            
            # adds the first line for header at the beginning
            f0=open(Signame+'w','r')
            dat=f0.readlines()
            f0.close()
            f1=open(Signame+'t','w')
            f1.writelines([first_line]+dat)
            f1.close()
            shutil.move(Signame+'t',Signame+'w')
            
            # Broadening the output self-energy to reduce qmc noise
            cmd = 'cd '+self.dir+'; ./broad -w '+str(self.wbroad)+' -k '+str(self.kbroad)+' '+self.Signame+'w'+' > '+self.Signame
            print cmd
            
            if (RUUN):
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, close_fds=True)
                stdin, stdout, stderr = (p.stdin,p.stdout,p.stderr)
                print >> self.fh_info, stdout.read(), stderr.read()
            
            Sig = loadtxt(Signame)

            
        print shape(Sig)
        Sigs = zeros( (len(ind_om), shape(Sig)[1]), dtype=float )
        for i in range(len(ind_om)):
            Sigs[i,:] = Sig[ind_om[i],:]
        
        return (Sigs, nf, TrSigmaG)
    
    def copyfiles(self,itt):
        shutil.copy2(self.dir+'/'+self.params['Sig'][0], self.dir+'/'+self.params['Sig'][0]+'.'+str(itt))
        shutil.copy2(self.dir+'/nohup_imp.out', self.dir+'/nohup_imp.out.'+str(itt))
        shutil.copy2(self.dir+'/Gf.out', self.dir+'/Gf.out.'+str(itt))
        shutil.copy2(self.dir+'/Delta.dat', self.dir+'/Delta.dat.'+str(itt))
        

    
DMFT=True
