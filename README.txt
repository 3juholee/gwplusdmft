PREINSTALL:
    1. PyQuante library: 
        -"http://pyquante.sourceforge.net/"
    2. CTQMC (continuous-time quantum Monte Carlo) for DMFT calculation
        -"http://hauleweb.rutgers.edu/downloads/"
        -See also additional preinstallations such as python and C++
        -Download "dmft_w2k.tgz" (version 2015)
        -Extract:  "tar -zxvf dmft_w2k.tgz"
        -Install according to "http://hauleweb.rutgers.edu/tutorials/Installation.html"
        -"export WIEN_DMFT_ROOT=where_your_bin_folder_is"

Comments:
    Each folder contains different version of GW+DMFT execution file.
        -> gwd/: fully self-consistent GW+DMFT (or only GW)
        -> gwd_v9/: ver9 of GW+DMFT
        -> gwd_at_lda: G0W0+DMFT with H0 at LDA level
        -> hf+dmft
    src: common libaries which are called by execution files in each GW+DMFT folder






May17/2016
#source codes to be synced to git/project_h2/gw+dmft

May23/2016
The current folder in the server (Beowulf@Rutgers: rupc08)
    /work/naivephya/dimer/project_n2/gw+dmft_svd/new_src/src_gw+dmft

July3/2016
Clean the folders
	
