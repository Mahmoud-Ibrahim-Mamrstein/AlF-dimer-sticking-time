from mpi4py import MPI
import numpy as np
import pandas as pd
import scipy
import os
import ase
#from ase import io
from ase import Atoms
from ase.io import read
from machine_learning_potential import ml_potential
import csv
import sys
start=20 #(included)
stop=25  #(not included)
diff=stop-start #number of R values
previous_tot_config=int(36495360+5*32*16.5*24*24*8)
tot_config=int(diff*32*16.5*24*24*8) #total number of configurations between start and stop
Rr1r2_config=int(diff*32*16.5) #number of configurations having unique sets of combinations of R, r1 and r2
previous_no_files=int(792+264) #number of files already generated
no_files=int(Rr1r2_config/10) #number of files consisting of ten of those unique sets of combinations of R, r1 and r2 each each with tehir angles
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID (rank)
size = comm.Get_size()  # Total number of processes

# Load necessary files
jacobi_cord = np.load(f"jacobi_coordinates_and_weights_{stop}.npz")['jacobi_cordinates']
jacobi_weights = np.load(f"jacobi_coordinates_and_weights_{stop}.npz")['jacobi_weights']
rand = np.load(f"/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/rand_ind_{stop}.npy")
print(rand)
# Paths for ML model and training set 

training_set_filename = '/gpfs/home/maaibrahim/AlF_dimer/data/Al2F2_CCSDt_energy_MP2_force_full_set.xyz' 
trained_ml_potential_model = '/gpfs/home/maaibrahim/AlF_dimer/trained_ml_potential_model.pkl'
ml_parameters = {
    'fd_displacement': 0.0025,
    'ml_potential_model': trained_ml_potential_model,
    'ml_training_set': training_set_filename,
    'ml_gpr_noise_level_bounds': 1e-07
}
ml_calculator = ml_potential(ml_parameters=ml_parameters)

# ML generated PES function with checkpointing and final logging

def generate_v(jacobi_cord,rand, file_indices):

    processed_files = []
    configs_used = []
    local_test_set=[]
    test_set={}
    file_index=[]

    for j in file_indices:

        no_of_config = (j + 1) * 10 * 4608   
        
        ind = []
        for k in rand[j-previous_no_files]:
            s = k
            ind.append(s)
            while s < (k + 4607):
                s += 1
                ind.append(s)

        #R, r1, r2, theta1, theta2, phi = jacobi_cord[np.array(ind)-previous_tot_config, 0], jacobi_cord[np.array(ind)-previous_tot_config, 1], jacobi_cord[np.array(ind)-previous_tot_config, 2], jacobi_cord[np.array(ind)-previous_tot_config, 3], jacobi_cord[np.array(ind)-previous_tot_config, 4], jacobi_cord[np.array(ind)-previous_tot_config, 5]


        v = []
        config_no=[]
        file_path = f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz'
        file_exists = os.path.exists(file_path)
        if file_exists:
        
            
            v_exists = np.load(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz')['v']
            config_no_exists = np.load(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz')['ind']            
            
            if len(v_exists) == 46080:

                continue
            
            elif len(v_exists) < 46080:
                
                config_no.extend(config_no_exists.tolist())
                v.extend(v_exists.tolist())
                
                for i, i_sys in enumerate(read('/gpfs/scratch/maaibrahim/xyz-2/'+str(j)+'.xyz', index=':', format='extxyz',parallel=False)):

            
                    if int(ind[i]) in config_no_exists:
                   
                        continue
                    
                    else:

                        #print('adding a row to an exisited file')
                        config_no.append(int(ind[i]))
                        i_sys.calc = ml_calculator
                        energy = i_sys.get_potential_energy() * 0.036749405469679  # Energy in amu
                        v.append(energy)                
                        if i % 10000 == 0:
                            np.savez(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz', v=np.array(v), ind=np.array(config_no))
                    np.savez(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz', v=np.array(v), ind=np.array(config_no))          
        elif not file_exists:
            
            for i, i_sys in enumerate(read('/gpfs/scratch/maaibrahim/xyz-2/'+str(j)+'.xyz', index=':', format='extxyz',parallel=False)):
                config_no.append(int(ind[i]))
                i_sys.calc = ml_calculator
                energy = i_sys.get_potential_energy() * 0.036749405469679  # Energy in amu
                v.append(energy)
                if i == 0:
                    np.savez(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz', v=np.array(v), ind=np.array(config_no))
                if i % 10000 == 0:
                    np.savez(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz', v=np.array(v), ind=np.array(config_no))
        np.savez(f'/gpfs/home/maaibrahim/AlF_dimer/AlF-draft-2/v_npz/{j}_v.npz', v=np.array(v), ind=np.array(config_no))    


    return 0

# Distribute work: split files evenly among processes

file_indices_per_rank = np.arange(previous_no_files,previous_no_files+no_files)[rank::size]

generate_v(jacobi_cord, rand, file_indices_per_rank) # Execute the function
