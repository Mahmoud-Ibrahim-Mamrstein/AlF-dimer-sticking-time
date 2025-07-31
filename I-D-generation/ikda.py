import numpy as np
from numpy import *
import ase
from ase import io
from ase import Atoms
import os
n=55
jacobi_cord = np.load(f"jacobi_coordinates_and_weights_{n}.npz")['jacobi_cordinates']
"""jacobi_cord = np.concatenate((np.load("jacobi_coordinates_and_weights_15.npz")['jacobi_cordinates'], 
                              np.load("jacobi_coordinates_and_weights_20.npz")['jacobi_cordinates'], 
                              np.load("jacobi_coordinates_and_weights_25.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_30.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_35.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_40.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_45.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_50.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_55.npz")['jacobi_cordinates'],
                              np.load("jacobi_coordinates_and_weights_60.npz")['jacobi_cordinates']))"""

jacobi_weights = np.load(f"jacobi_coordinates_and_weights_{n}.npz")['jacobi_weights']
"""np.concatenate((np.load("jacobi_coordinates_and_weights_15.npz")['jacobi_weights'], 
                                 np.load("jacobi_coordinates_and_weights_20.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_25.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_30.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_35.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_40.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_45.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_50.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_55.npz")['jacobi_weights'],
                                 np.load("jacobi_coordinates_and_weights_60.npz")['jacobi_weights']))"""

R, r1, r2, theta1, theta2, phi = jacobi_cord[:, 0], jacobi_cord[:, 1], jacobi_cord[:, 2], jacobi_cord[:, 3], jacobi_cord[:, 4], jacobi_cord[:, 5]

wt1, wt2, wph = jacobi_weights[:, 3],jacobi_weights[:, 4], jacobi_weights[:, 5]

savez_compressed(f"/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/jacobi_cord_{n}",R=R,r_1=r1,r_2=r2,theta_1=theta1,theta_2=theta2,phi=phi,wt1=wt1,wt2=wt2,wph=wph)


def Ixyz(R,r_1,r_2,theta_1,theta_2,phi,number_of_configurations=shape(R)[0]):
    m_al=26.981539*1822.8885 #Atomic Mass of Aluminium in amu
    m_f=18.998403*1822.8885 #Atomic Mass of Fluorine in amu
    mu_alf=(m_al*m_f)/(m_al+m_f)
    c1=np.cos(theta_1)
    s1=np.sin(theta_1)
    c2=np.cos(theta_2)
    s2=np.sin(theta_2)
    cp=np.cos(phi)
    sp=np.sin(phi)
    c12=c1**2
    s12=s1**2
    c22=c2**2
    s22=s2**2
    cp2=cp**2
    sp2=sp**2
    r12=r_1**2
    r22=r_2**2
    i_xx=mu_alf*(r12*s12+r22*s22)
    i_yy=mu_alf*(r12*c12+r22*c22+r22*s22*sp2)+R**2*(m_f+m_al)/2
    i_zz=mu_alf*(r12+r22*c22+r22*s22*cp2)+R**2*(m_f+m_al)/2
    i_xy=-mu_alf*(r12*c1*s1+r22*c2*s2*cp)
    i_yx=i_xy
    i_xz=-mu_alf*r22*c2*s2*sp
    i_zx=i_xz
    i_yz=-mu_alf*r22*s22*sp*cp
    i_zy=i_yz
    I=np.stack((i_xx,i_xy,i_xz,i_yx,i_yy,i_yz,i_zx,i_zy,i_zz), axis=-1).reshape(np.shape(R)[0],3,3)
    I_det=np.linalg.det(I)
    I_inv=np.linalg.inv(I)
    return I, I_det, I_inv
    
    
def K(R,r_1,r_2,theta_1,theta_2,phi,number_of_configurations=shape(R)[0]):
    m_al=26.981539*1822.8885 #Atomic Mass of Aluminium in amu
    m_f=18.998403*1822.8885 #Atomic Mass of Fluorine in amu
    mu_alf=(m_al*m_f)/(m_al+m_f)
    zero=np.zeros(number_of_configurations)
    one=np.ones(number_of_configurations)
    KmK=np.stack((1/2*(m_al+m_f)*one,zero,zero,zero,zero,zero,zero,mu_alf*one,zero,zero,zero,zero,zero,zero,mu_alf*one,zero,zero,zero,zero,zero,zero,mu_alf*r_1**2,zero,zero,zero,zero,zero,zero,mu_alf*r_2**2,zero,zero,zero,zero,zero,zero,mu_alf*np.sin(theta_2)**2*r_2**2), axis=-1).reshape(number_of_configurations,6,6)

    return KmK



def D(R,r_1,r_2,theta_1,theta_2,phi,number_of_configurations=shape(R)[0]):
 
    m_al=26.981539*1822.8885 #Atomic Mass of Aluminium in amu
    m_f=18.998403*1822.8885 #Atomic Mass of Fluorine in amu
    mu_alf=(m_al*m_f)/(m_al+m_f)
    zero=np.zeros(number_of_configurations)
    one=np.ones(number_of_configurations)
    D=np.stack((zero,zero,zero,zero,zero,mu_alf*np.sin(theta_2)**2*r_2**2,zero,zero,zero,zero,-mu_alf*np.sin(phi)*r_2**2,-mu_alf*np.cos(phi)*np.cos(theta_2)*np.sin(theta_2)*r_2**2,zero,zero,zero,mu_alf*r_1**2,mu_alf*np.cos(phi)*r_2**2,-mu_alf*np.sin(phi)*np.cos(theta_2)*np.sin(theta_2)*r_2**2), axis=-1).reshape(number_of_configurations,3,6)
    
    return D

def A(I_inv,D, KmK,number_of_configurations=shape(R)[0]):
    m_al=26.981539*1822.8885 #Atomic Mass of Aluminium in amu
    
    m_f=18.998403*1822.8885 #Atomic Mass of Fluorine in amu
    
    DI=matmul(np.transpose(D,(0,2,1)),I_inv)
    
    DID=matmul(DI,D)

    A=0.5*(KmK-DID)
    
    det_A=linalg.det(A)
    
    return det_A

bohr=1/0.529177210544

R,r1,r2=R*bohr,r1*bohr,r2*bohr

I, I_det, I_inv=Ixyz(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

savez_compressed(f"/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/I_per_configuration_{n}",I, I_det, I_inv)

KmK=K(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

print('KmK =',KmK)

#save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/KmK_per_configuration_15",KmK)

D=D(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

print('D =' ,D)

#save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration_15",D)


det_A=A(I_inv,D, KmK,number_of_configurations=shape(R)[0])

print('det_A =',det_A)

save(f"/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/det_A_per_configuration_{n}", det_A)

print('shape of det_A: ',np.shape(det_A))

print('shapes: ','shape(IMAT[I]) = ',np.shape(I),'shape(det_A) = ',np.shape(det_A))

print(np.any(np.less(det_A,det_A*0)))

FT=np.less(det_A,det_A*0)

print(len(FT[FT==True]))

    
"""if os.path.exists("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/I_per_configuration.npz"):
    IMAT=np.load("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/I_per_configuration.npz")
    I, I_det, I_inv=IMAT['arr_0'],IMAT['arr_1'],IMAT['arr_2']
else:
    I, I_det, I_inv=Ixyz(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

    savez_compressed("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/I_per_configuration",I, I_det, I_inv)
    

if os.path.exists("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/KmK_per_configuration.npz"):
    KMAT=np.load("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/KmK_per_configuration.npz")
    KMK=KMAT['KmK']
else:
    KmK=K(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

    #save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/KmK_per_configuration",KmK)


if os.path.exists("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration.npz"):
    DMAT=np.load("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration.npz")
    D=DMAT['D']
    if np.shape(DMAT)==np.shape(KMAT):
        print('K shpe is ',np.shape(KMAT))
        print('D shpe is ',np.shape(DMAT))
    else:
        D=D(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

        #save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration",D)  
if not os.path.exists("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration.npz"):
    D=D(R,r1,r2,theta1,theta2,phi,number_of_configurations=shape(R)[0])

    #save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/D_per_configuration",D)   

#print('Jacopi type: ',type(jacobi_cord))
#print('K type: ',type(Kal1))
#print('shape of I_det: ',np.shape(I_det))

det_A=A(I_inv,D, KmK,number_of_configurations=shape(R)[0])

print('det_A =',det_A)

save("/gpfs/scratch/maaibrahim/AlF_dimer/AlF-draft-2/det_A_per_configuration", det_A)

print('shape of det_A: ',np.shape(det_A))

print('shapes: ','shape(IMAT[I]) = ',np.shape(I),'shape(det_A) = ',np.shape(det_A))

print(np.any(np.less(det_A,det_A*0)))

FT=np.less(det_A,det_A*0)

print(len(FT[FT==True]))"""