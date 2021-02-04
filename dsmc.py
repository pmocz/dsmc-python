import matplotlib.pyplot as plt
import numpy as np
from scipy import special

"""
Create Your Own Direct Simulation Monte Carlo (With Python)
Philip Mocz (2021) Princeton Univeristy, @PMocz

Simulate dilute gas with DSMC
Setup: Rayleigh Problem = gas between 2 plates (Alexander & Garcia, 1997)

dimensionless units of m = sigma = k T0 = 1

"""


def main():
	""" Direct Simulation Monte Carlo """
	
	# Simulation parameters
	uw              = 0.2       # lower wall velocity
	Tw              = 1         # wall temperature
	n0              = 0.001     # density
	N               = 50000     # number of sampling particles
	Nsim            = 3         # number of simulations to run
	Ncell           = 50        # number of cells
	Nmft            = 20        # number of mean-free times to run simulation
	plotRealTime    = False # True      # animate
	
	Nt              = Nmft*25                    # number of time steps (25 per mean-free time)
	lambda_mfp      = 1/(np.sqrt(2)*np.pi*n0)    # mean free path ~= 225
	Lz              = 10*lambda_mfp              # height of box  ~= 2250.8
	Kn              = lambda_mfp / Lz            # Knudsen number  = 0.1
	v_mean = (2/np.sqrt(np.pi)) * np.sqrt(2*Tw)  # mean speed
	tau             = lambda_mfp / v_mean        # mean-free time
	dt              = Nmft*tau/Nt                # timestep
	dz              = Lz/Ncell                   # cell height
	vol             = Lz*dz*dz/Ncell             # cell volume
	Ne              = n0*Lz*dz*dz/N   # number of real particles each sampling particle represents
		
	# vector for recording v_y(z=0)
	vy0 = np.zeros((Nsim,Nt))    

	# set the random number generator seed
	np.random.seed(17) 

	# prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	ax = plt.gca()

	# Simulation Main Loop
	
	for sim in range(Nsim):
		
		print('Simulation',sim+1,'of',Nsim)
    
		# Initialize 
		x = dz * np.random.random(N)
		y = dz * np.random.random(N)
		z = Lz * np.random.random(N)
		
		# Maxwellian
		vx = np.random.normal(0, Tw, N) 
		vy = np.random.normal(0, Tw, N) 
		vz = np.random.normal(0, Tw, N) 


		# Evolve
		for i in range(Nt):
			
			print('  timestep',i,'of',Nt,'  (sim',sim+1,'/',Nsim,')')
			
			# drift
			x += dt*vx
			y += dt*vy
			z += dt*vz
			
			
			# collide specular wall (z=Lz)
			# trace the straight-line trajectory to the top wall, bounce it back
			hit_top = z > Lz
			dt_ac = (z[hit_top]-Lz) / vz[hit_top] # time after collision
			vz[hit_top] = -vz[hit_top]  # reverse normal component of velocity
			z[hit_top]  = Lz + dt_ac * vz[hit_top]
			
			
			# collide thermal wall (z=0)
			# reset velocity to a biased maxwellian upon impact
			hit_bot = z < 0
			dt_ac = z[hit_bot] / vz[hit_bot]
			x[hit_bot] -= dt_ac * vx[hit_bot]
			y[hit_bot] -= dt_ac * vy[hit_bot]
			Nbot = np.sum( hit_bot )
			
			vx[hit_bot] = np.sqrt(Tw) * np.random.normal(0, 1, Nbot) 
			vy[hit_bot] = np.sqrt(Tw) * np.random.normal(0, 1, Nbot) + uw
			vz[hit_bot] = np.sqrt( -2 * Tw * np.log(np.random.random(Nbot)) )
			
			x[hit_bot] += dt_ac * vx[hit_bot]
			y[hit_bot] += dt_ac * vy[hit_bot]
			z[hit_bot]  = dt_ac * vz[hit_bot]
			
			# periodic BCs
			x = np.mod(x,dz)
			y = np.mod(y,dz)
			
			# collide particles using acceptance--rejection scheme
			v_rel_max = 6 # (over-)estimate upper limit to relative vel.
			N_collisions = 0
			# loop over cells
			for j in range(Ncell):
				
				in_cell = (j*dz < z) & (z < (j+1)*dz)
				Nc = np.sum( in_cell )
				x_c = x[in_cell]
				y_c = y[in_cell]
				z_c = z[in_cell]
				vx_c = vx[in_cell]
				vy_c = vy[in_cell]
				vz_c = vz[in_cell]
				
				M_cand = np.ceil(Nc**2 * np.pi * v_rel_max * Ne * dt/(2*vol)).astype(int)
					
				# propose collision between i and j
				for k in range(M_cand):
					
					r_fac = np.random.random()
					i_prop = np.random.randint(Nc)
					j_prop = np.random.randint(Nc)
					
					v_rel = np.sqrt((vx_c[i_prop]-vx_c[j_prop])**2 + (vy_c[i_prop]-vy_c[j_prop])**2 + (vz_c[i_prop]-vz_c[j_prop])**2 )
					
					# accept collision with appropriate probability
					if v_rel > r_fac*v_rel_max:
						
						# process collision -- hard sphere
						vx_cm = 0.5 * (vx_c[i_prop] + vx_c[j_prop])
						vy_cm = 0.5 * (vy_c[i_prop] + vy_c[j_prop])
						vz_cm = 0.5 * (vz_c[i_prop] + vz_c[j_prop])
						cos_theta = 2 * np.random.random() - 1
						sin_theta = np.sqrt( 1 - cos_theta**2 )
						phi       = 2 * np.pi * np.random.random()
						vx_p = v_rel * sin_theta * np.cos(phi)
						vy_p = v_rel * sin_theta * np.sin(phi)
						vz_p = v_rel * cos_theta
						vx_c[i_prop] = vx_cm + 0.5*vx_p
						vy_c[i_prop] = vy_cm + 0.5*vy_p
						vz_c[i_prop] = vz_cm + 0.5*vz_p
						vx_c[j_prop] = vx_cm - 0.5*vx_p
						vy_c[j_prop] = vy_cm - 0.5*vy_p
						vz_c[j_prop] = vz_cm - 0.5*vz_p
						
						N_collisions += 1
						
				x[in_cell]  = x_c 
				y[in_cell]  = y_c 
				z[in_cell]  = z_c
				vx[in_cell] = vx_c 
				vy[in_cell] = vy_c 
				vz[in_cell] = vz_c
				
			print('    ',N_collisions,' collisions')
			
			# periodic BCs
			x = np.mod(x,dz)
			y = np.mod(y,dz)
			
			# record v_y(z=0)
			vy0[sim,i] = np.mean( vy[ (0 < z) & (z < dz) ] )
			
			# measure vy along box
			bin_c = dz * np.linspace(0.5,Ncell-0.5,Ncell)
			vy_profile = np.zeros((Ncell,1))
			for j in range(Ncell):
				in_cell = ( j*dz < z ) & ( z < (j+1)*dz )
				vy_profile[j] = np.mean(vy[in_cell])

			
			# plot phase-space slice
			if plotRealTime:
				plt.cla()
				plt.scatter(z[0::20],vy[0::20], color='blue', s=0.1)
				plt.plot(bin_c,vy_profile)
				ax.set(xlim=(0, Lz), ylim=(-3,3))
				plt.xlabel(r'$z$')
				plt.ylabel(r'$v_y$')
				plt.pause(0.001)


	# Plot results: compare v_y(z=0) to BGK theory
	fig2 = plt.figure(figsize=(6,4), dpi=80)
	ax2  = plt.gca()
	tt = dt * np.linspace(1, Nt, num=Nt) / tau
	bgk = np.zeros(tt.shape)
	for i in range(Nt):
		xx = np.linspace(tt[i]/10000, tt[i], num=10000)
		bgk[i] = 0.5*(1 + np.trapz(np.exp(-xx) / xx * special.iv(1,xx), x=xx))
	plt.plot(tt*2.5, bgk, label='BGK theory', color='red')
	plt.plot(tt, np.mean(vy0,axis=0).reshape((Nt,1))/uw, label='DSMC', color='blue')
	plt.xlabel(r'$t/\tau$')
	plt.ylabel(r'$u_y(z=0)/u_w$')
	ax2.set(xlim=(0, Nmft), ylim=(0.5, 1.1))
	ax2.legend(loc='upper left')
			
	# Save figure
	plt.savefig('dsmc.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
  main()

