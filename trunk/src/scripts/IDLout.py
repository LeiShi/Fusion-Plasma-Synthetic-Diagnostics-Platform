"""script for write out ECEI plasma parameters and result to a file for IDL to read 

file contains (in order):
    NR: number of grid along the path
    NF: number of selected frequencies
    R_arr: 2D array (NF,NR),R coordinates on the grids
    Z_arr: 2D array (NF,NR),Z coordinates on the grids
    ne_arr: 2D array (NF,NR),electron density on the grids
    Te_arr: 2D array (NF,NR),electron temperature on the grids
    B_arr: 2D array (NF,NR),total B field on the grids
    f_arr: array contains all the frequencies
    PYalpha: 2D array (NF,NR), contains calculated alpha values, note that in python the later index is faster varying, and in IDL the opposite.  
"""
import numpy as np
def IDLoutput(Profs,alphas,filename = "IDLout.dat"):
    NF = len(Profs)
    R_arr = []
    Z_arr = []
    ne_arr = []
    Te_arr = []
    B_arr = []
    f_arr = []
    PYalpha = []
    for i in range(NF):    
        R_arr.append(Profs[i]['Grid'].R2D[0,:])
        Z_arr.append(Profs[i]['Grid'].Z2D[0,:])
        ne_arr.append(Profs[i]['ne'][0,:])
        Te_arr.append(Profs[i]['Te'][0,:])
        B_arr.append(Profs[i]['B'][0,:])
        f_arr.append(Profs[i]['omega'][0]/(2*np.pi))
        PYalpha.append(alphas[i][0,0,:])
    NR = len(R_arr[0])
    arrs = [R_arr,Z_arr,ne_arr,Te_arr,B_arr,PYalpha]
    with open(filename,'w') as fout:
        fout.write(str(NR)+'\n')
        fout.write(str(NF)+'\n')
        for i in range(NF):
            fout.write(str(f_arr[i])+'\t')
        fout.write('\n')
        for con in arrs:
            for i in range(NF):
                for j in range(NR):
                    fout.write(str(con[i][j])+'\t')
                fout.write('\n')
            fout.write('\n')
        fout.flush()
    
                
    