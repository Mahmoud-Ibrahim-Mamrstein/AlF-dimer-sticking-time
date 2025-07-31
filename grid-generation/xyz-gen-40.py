import numpy as np
from numpy import *
import ase
from ase import io
from ase import Atoms
start=35 #(included)
stop=40  #(not included)
diff=stop-start #number of R values
previous_tot_config=int(36495360+4*(5*32*16.5*24*24*8))
tot_config=int(diff*32*16.5*24*24*8) #total number of configurations between start and stop
Rr1r2_config=int(diff*32*16.5) #number of configurations having unique sets of combinations of R, r1 and r2
previous_no_files=int(792+4*(264)) #number of files already generated
no_files=int(Rr1r2_config/10) #number of files consisting of ten of those unique sets of combinations of R, r1 and r2 each each with tehir angles

def jacobi(R_min,R_max,R_np,r_min,r_max,r_np,theta_np,phi_np):
    R_doamin=np.linspace(R_min, R_max, num=R_np, endpoint=False, retstep=True)[0][start:stop]
    R_space=np.linspace(R_min, R_max, num=R_np, endpoint=False, retstep=True)[1]
    r_doamin=np.linspace(r_min, r_max, num=r_np, endpoint=False, retstep=True)[0]
    r_space=np.linspace(r_min, r_max, num=r_np, endpoint=False, retstep=True)[1]
    theta_doamin=np.polynomial.legendre.leggauss(theta_np)[0]*(np.pi/2)+(np.pi/2)
    legendre_weights=np.polynomial.legendre.leggauss(theta_np)[1]*(np.pi/2)
    phi_doamin=np.polynomial.chebyshev.chebgauss(phi_np)[0]*(np.pi/2)+(np.pi/2)
    chebyshev_weights=np.polynomial.chebyshev.chebgauss(phi_np)[1]*(np.pi/2)
    jacobi_cord=np.vstack(np.meshgrid(R_doamin, r_doamin,r_doamin,theta_doamin,theta_doamin,phi_doamin,indexing='ij')).reshape(6,-1).T
    jacobi_cord = jacobi_cord[(jacobi_cord[:,2] >= jacobi_cord[:,1])]
    jacobi_weights=np.vstack(np.meshgrid(R_doamin, r_doamin,r_doamin,legendre_weights,legendre_weights,chebyshev_weights,indexing='ij')).reshape(6,-1).T
    jacobi_weights = jacobi_weights[(jacobi_weights[:,2] >= jacobi_weights[:,1])]
    jacobi_weights[:,0]=ones(shape(jacobi_weights[:,0]))
    jacobi_weights[:,1]=ones(shape(jacobi_weights[:,1]))
    jacobi_weights[:,2]=ones(shape(jacobi_weights[:,2]))
    return jacobi_cord, jacobi_weights, R_space, r_space, legendre_weights, chebyshev_weights

def xyz(R,r_1,r_2,theta_1,theta_2,phi,number_of_configurations,rand):
    m_al=26.981539 #Atomic Mass of Aluminium in amu
    m_f=18.998403 #Atomic Mass of Fluorine in amu
    m_alf=m_al+m_f
    x_f1=R+(m_al/m_alf)*r_1*np.cos(theta_1)
    y_f1=(m_al/m_alf)*r_1*np.sin(theta_1)
    z_f1=0*x_f1
    x_al1=R-(m_f/m_alf)*r_1*np.cos(theta_1)
    y_al1=-(m_f/m_alf)*r_1*np.sin(theta_1)
    z_al1=0*x_al1
    x_f2=(m_al/m_alf)*r_2*np.cos(theta_2)
    y_f2=(m_al/m_alf)*r_2*np.sin(theta_2)*np.cos(phi)
    z_f2=(m_al/m_alf)*r_2*np.sin(theta_2)*np.sin(phi)
    x_al2=-(m_f/m_alf)*r_2*np.cos(theta_2)
    y_al2=-(m_f/m_alf)*r_2*np.sin(theta_2)*np.cos(phi)
    z_al2=-(m_f/m_alf)*r_2*np.sin(theta_2)*np.sin(phi)
    for i in range(previous_no_files,previous_no_files+no_files):
            atoms=[]
            for j in rand[i-previous_no_files]:
                s=j
                atoms.append(Atoms(['Al','F','Al','F'], positions=[(x_al1[j], y_al1[j], z_al1[j]), (x_f1[j], y_f1[j], z_f1[j]),(x_al2[j], y_al2[j], z_al2[j]),(x_f2[j], y_f2[j], z_f2[j])]))
                while s<(j+4607):
                    s=s+1
                    atoms.append(Atoms(['Al','F','Al','F'], positions=[(x_al1[s], y_al1[s], z_al1[s]), (x_f1[s], y_f1[s], z_f1[s]),(x_al2[s], y_al2[s], z_al2[s]),(x_f2[s], y_f2[s], z_f2[s])]))
            ase.io.write(filename='/gpfs/scratch/maaibrahim/xyz-2/'+str(i)+'.xyz',images=atoms)
    return x_f1, y_f1, z_f1, x_al1, y_al1,z_al1, x_f2, y_f2, z_f2, x_al2, y_al2, z_al2, atoms     
    
#rand=np.load('rand_ind.npy')
   
jacobi_cord, jacobi_weights, R_space, r_space, legendre_weights, chebyshev_weights = jacobi(R_min=0.4,R_max=10.0,R_np=60,r_min=1.4,r_max=6.0,r_np=32,theta_np=24,phi_np=8)

savez_compressed(f"jacobi_coordinates_and_weights_{stop}",jacobi_cordinates=jacobi_cord, jacobi_weights=jacobi_weights,R_space=R_space, r_space=r_space, legendre_weights=legendre_weights, chebyshev_weights=chebyshev_weights)

#rand=np.random.choice(np.linspace(0, 36495360, num=7920, endpoint=False, retstep=True,dtype=int)[0], size=(792,10), replace=False, p=None)
rand=np.linspace(0, tot_config, num=Rr1r2_config, endpoint=False, retstep=True,dtype=int)[0].reshape((no_files,10))

np.save(f'rand_ind_{stop}', previous_tot_config+rand)

x_f1, y_f1, z_f1, x_al1, y_al1,z_al1, x_f2, y_f2, z_f2, x_al2, y_al2, z_al2,atoms=xyz(jacobi_cord[:,0],jacobi_cord[:,1],jacobi_cord[:,2],jacobi_cord[:,3],jacobi_cord[:,4],jacobi_cord[:,5],number_of_configurations=shape(jacobi_cord[:,5])[0],rand=rand)
