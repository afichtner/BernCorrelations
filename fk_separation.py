import obspy
import numpy as np
from pandas import read_csv
from pandas import read_csv
import matplotlib.pyplot as plt

def fk_separation(filename, c_min, c_max, f_min, f_max, plot=False):
	"""
	Perform f-k-domain separation of a cross-correlation record section.

	filname: input filename
	c_min, c_max: minimum and maximum phase velocities [m/s]
	f_min, f_max: minimum and maximum frequencies [Hz]
	plot: screen output if wanted

	"""

	#- Load data. ===================================================
	cct=np.load(filename)
	
	# Length of arrays and increments.
	nt=cct.shape[1]
	nx=cct.shape[0]-1
	dt=cct[0,1]-cct[0,0]

	# Receiver spacing.
	dx=1.9411132335662842

	# Normalisation.
	for i in range(1,nx): cct[i,:]/=np.max(np.abs(cct[i,:]))

	# Tapering in space to avoid wavenumber-domain artefacts.
	width=20
	for i in range(width): 
		cct[i+1,:]=(np.float(i+1)/np.float(width+1))*cct[i+1,:]
		cct[nx-i,:]=(np.float(i+1)/np.float(width+1))*cct[nx-i,:]

	#================================================================
	#- f-k separation for the causal side. ==========================
	#================================================================

	#- Extract causal part. =========================================
	cct_cropped=cct[1:,int((nt-1)/2):nt].copy()
	t=np.linspace(0.0,(nt-1)*dt/2.0,int((nt-1)/2)+1)

	#- Plot causal part.
	if plot:

		plt.figure(figsize=(10,20))
		scale=3.0

		for i in range(nx):
			     
			data=cct_cropped[i,:]#/np.max(np.abs(cct_cropped[i,:]))
			dist_var=(i-1)*dx
			
			plt.plot(t,(scale*data)+dist_var,'k-', alpha = 0.4) 
			plt.fill_between(t,(scale*data)+dist_var,y2=np.ones(np.shape(t))*dist_var,where=(data+dist_var>=dist_var), interpolate=True,fc='k',alpha=0.8)

		plt.xlabel('time [s]')
		plt.ylabel('distance [m]')
		plt.title('causal part')
		plt.grid()
		plt.show()

	#- Compute frequency-wavenumber domain representation. ==========
	nt_cropped=len(t)
	ccf=np.fft.fft2(cct_cropped)
	ccfr=np.roll(np.roll(ccf,int((nt_cropped-1)/2),axis=1),int((nx-1)/2),axis=0)
	f=np.linspace(-0.5/dt,0.5/dt,nt_cropped)
	k=np.linspace(-np.pi/dx,np.pi/dx,nx)

	#- Plot f-k domain amplitude spectrum.
	if plot:

		ff,kk=np.meshgrid(f,k)
		plt.subplots(1, figsize=(25,25))
		plt.pcolor(ff,kk,np.abs(ccfr),cmap='Greys')
		plt.xlim(-50.0,50.0)
		plt.ylim(-0.5,0.5)
		plt.xlabel('f [Hz]')
		plt.ylabel('k [1/m]')
		plt.title('f-k power spectrum, causal',pad=25)
		plt.colorbar()
		plt.grid()
		plt.show()

	#- Build f-k mask. ==============================================

	#- Mask for forward-propagating wavefield.
	mask_fwd=np.ones(np.shape(ccfr),dtype='complex64')

	for i in range(nx):
		for j in range(nt_cropped):
		
			#- Compute phase velocity.
			if np.abs(k[i]):
				c=2.0*np.pi*f[j]/k[i]
			else:
				c=1.0e9
			
			#- Maximum and minimum absolute phase velocities.
			if (np.abs(c)>c_max) or (np.abs(c)<c_min): mask_fwd[i,j]=0.0
			#- Remove backward propagation.
			if c>0.0: mask_fwd[i,j]=0.0
			#- Bandpass filter.
			if np.abs(f[j])>f_max: mask_fwd[i,j]=0.0
			if np.abs(f[j])<f_min: mask_fwd[i,j]=0.0
			
	#- Smooth the mask.
	dummy_fwd=mask_fwd.copy()
	mask_fwd_smooth=mask_fwd.copy()

	for l in range(4):
		for i in np.arange(1,nx-1):
			for j in np.arange(1,nt_cropped-1):
				mask_fwd_smooth[i,j]=(dummy_fwd[i,j]+dummy_fwd[i-1,j]+dummy_fwd[i+1,j]+dummy_fwd[i,j-1]+dummy_fwd[i,j+1])/5.0
			dummy_fwd=mask_fwd_smooth

	#- Apply f-k mask and transform back. ===========================
	ccfr_fwd_filtered=ccfr*mask_fwd_smooth

	#- Roll back.
	ccf_fwd_filtered=np.roll(np.roll(ccfr_fwd_filtered,-int((nt_cropped-1)/2),axis=1),-int((nx-1)/2),axis=0)
	cct_fwd_filtered_causal=np.real(np.fft.ifft2(ccf_fwd_filtered))


	#================================================================
	#- f-k separation for the acausal side. ==========================
	#================================================================

	#- Extract acausal part. =========================================
	cct_cropped=cct[1:,int((nt-1)/2)+1:0:-1].copy()

	#- Plot acausal part.
	if plot:

		plt.figure(figsize=(10,20))
		scale=3

		for i in range(nx):
			      
			data=cct_cropped[i,:]
			dist_var=(i-1)*dx
			
			plt.plot(t,(scale*data)+dist_var,'k-', alpha = 0.4) 
			plt.fill_between(t,(scale*data)+dist_var,y2=np.ones(np.shape(t))*dist_var,where=(data+dist_var>=dist_var), interpolate=True,fc='k',alpha=0.8)

		plt.xlabel('time [s]')
		plt.ylabel('distance [m]')
		plt.title('acausal part')
		plt.grid()
		plt.show()

	#- Compute frequency-wavenumber domain representation. ==========
	nt_cropped=len(t)
	ccf=np.fft.fft2(cct_cropped)
	ccfr=np.roll(np.roll(ccf,int((nt_cropped-1)/2),axis=1),int((nx-1)/2),axis=0)
	f=np.linspace(-0.5/dt,0.5/dt,nt_cropped)
	k=np.linspace(-np.pi/dx,np.pi/dx,nx)

	#- Plot f-k domain amplitude spectrum.
	if plot:

		ff,kk=np.meshgrid(f,k)
		plt.subplots(1, figsize=(25,25))
		plt.pcolor(ff,kk,np.abs(ccfr),cmap='Greys')
		plt.xlim(-50.0,50.0)
		plt.ylim(-0.5,0.5)
		plt.xlabel('f [Hz]')
		plt.ylabel('k [1/m]')
		plt.title('f-k power spectrum, acausal',pad=25)
		plt.colorbar()
		plt.grid()
		plt.show()

	#- Apply f-k mask and transform back. ===========================
	ccfr_fwd_filtered=ccfr*mask_fwd_smooth

	#- Roll back.
	ccf_fwd_filtered=np.roll(np.roll(ccfr_fwd_filtered,-int((nt_cropped-1)/2),axis=1),-int((nx-1)/2),axis=0)
	cct_fwd_filtered_acausal=np.real(np.fft.ifft2(ccf_fwd_filtered))

	#- Return. ======================================================
	return t,cct_fwd_filtered_causal, cct_fwd_filtered_acausal


