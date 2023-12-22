import os
import sys
#sys.path.append("/home/alireza/CODES/My_scr")
import numpy as np
#from interface_xmls import *
<<<<<<< HEAD:My_Scripts/my_functions.py
#import interface_xmls
=======
# import interface_xmls
>>>>>>> 06f68684216e875f47e2ca60df16f23f17f234ff:my_functions.py
# import P_interface_xmls
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import AbinitToTHz
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
import ase
from ase.data import atomic_numbers 
from ase import build, Atoms
from ase.units import Ha, Bohr
from ase.io import write, read
from ase.build import make_supercell
import spglib.spglib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
# from SC_xml_potential import *
from os.path import exists
#from pyDFTutils.perovskite import frozen_mode as FM
# from mpl_toolkits.mplot3d import axes3d
#import abipy
#from abipy import abilab
#from abipy.dfpt import converters as cnvtr
from ase.neighborlist import NeighborList
#from My_ddb import ddb_reader
#import My_ddb
thz_cm = 33.356/1.374673102
from collections import OrderedDict
Hatoev=Ha/Bohr
import time
import My_Scripts.SC_xml_potential as SCXML
from My_Scripts.My_simulations import MB_sim
import My_Scripts.my_supercells as my_SClls
from My_Scripts import mync
import  My_Scripts.xml_io as xml_io
import  My_Scripts.missfit_terms as missfit_terms

def get_dist_strc(SL_atms,with_bloch_comp,dim,atom_to_disp,mat_id_To_displace,ref_atm_sym,STRC_inv_uc_cell,dom_const,my_dic,wall_size=1):
    SC_dist = make_supercell(SL_atms, dim)
    chem_sym = SC_dist.get_chemical_symbols()
    tag_id = SC_dist.get_array('tag_id')
    a_dir = dim[0][0]
    new_pos = []
    
    
    if with_bloch_comp: 
        dm_size = int((a_dir/2)-wall_size)
        dm_size_0 = int((dm_size/2))
        dm_size_l = dm_size-dm_size_0
        dm_size_0_clls = list(range(0,dm_size_0))
        dm_size_l_clls = list(range(a_dir-1,a_dir-dm_size_l-1,-1))
        cell_dm_2 = list(range(dm_size_0+wall_size,dm_size_0+wall_size+dm_size))
        wall_1 = list(range(dm_size_0,dm_size_0+wall_size))
        wall_2 = list(range(dm_size_0+wall_size+dm_size,dm_size_0+2*wall_size+dm_size))
        wall_cells = [*wall_1,*wall_2]
        cell_dm_1 = [*dm_size_0_clls,*dm_size_l_clls]

        # all_cells = list(range(a_dir))
        chem_sym = SC_dist.get_chemical_symbols()
        for atm_cnt, atm_pos in enumerate(SC_dist.get_positions()):
            if chem_sym[atm_cnt] in atom_to_disp and tag_id[atm_cnt][1]==mat_id_To_displace:
                ref_atm = find_clesest_atom(SC_dist,[0,0,0],atm_sym=ref_atm_sym[chem_sym[atm_cnt]])
                dists = atm_pos - SC_dist.get_positions()[ref_atm]
                cell_atm = np.dot((1/0.95)*STRC_inv_uc_cell, dists)
                cell_atm = list(map(int,cell_atm))

                if cell_atm[0] in wall_cells:  #Wall part
                    new_pos.append([atm_pos[0],atm_pos[1]+dom_const[1]*my_dic[1],atm_pos[2]]) 
                   
                    # if cell_atm[0] == int(a_dir/2):
                    #     new_pos.append([atm_pos[0],atm_pos[1]+dom_const[1]*my_dic[1],atm_pos[2]]) 

                elif cell_atm[0] in cell_dm_1:
                    new_pos.append([atm_pos[0],atm_pos[1],atm_pos[2]+dom_const[2]*my_dic[2]])
                else:
                    new_pos.append([atm_pos[0],atm_pos[1],atm_pos[2]+my_dic[2]])

            else:
                new_pos.append(atm_pos)
    else:
        for atm_cnt, atm_pos in enumerate(SC_dist.get_positions()):
            if chem_sym[atm_cnt] in atom_to_disp and tag_id[atm_cnt][1]==mat_id_To_displace:
                ref_atm = find_clesest_atom(SC_dist,[0,0,0],atm_sym=ref_atm_sym[chem_sym[atm_cnt]])
                dists = atm_pos - SC_dist.get_positions()[ref_atm]
                cell_atm = np.dot((1/0.95)*STRC_inv_uc_cell, dists)
                cell_atm = list(map(int,cell_atm))
                if cell_atm[0] in list(range(0,int(a_dir/2))):
                    new_pos.append([atm_pos[0]+dom_const[0]*my_dic[0],atm_pos[1]+dom_const[1]*my_dic[1],atm_pos[2]+dom_const[2]*my_dic[2]])
                else:
                    new_pos.append([atm_pos[0]+my_dic[0],atm_pos[1]+my_dic[1],atm_pos[2]+my_dic[2]])                        
            else:
                new_pos.append(atm_pos)
    my_sl_distor = Atoms(numbers=SC_dist.get_atomic_numbers(),positions=new_pos, cell=SC_dist.get_cell(), pbc=True)
    return(my_sl_distor)

def get_str_disp(mode,Phonon_obj,qpt = [0,0,0],get_disp=False,amp=1,path='./'):  
    os.chdir(path)        
    evecs,freqs = get_Evecs_Freqa_from_phonon_obj(Phonon_obj,qpt=qpt)
    HS_STR = Atoms(numbers = Phonon_obj.unitcell.get_atomic_numbers(),scaled_positions=Phonon_obj.unitcell.scaled_positions,cell = Phonon_obj.unitcell.cell)
    natom = len(Phonon_obj.unitcell.get_atomic_numbers())             
    freq = freqs[mode]
    # print(mode,f'   >>> Freq :  {freq:.2f} ', freq)
    # os.system(f'mkdir Mod_{mode}_freq_{freq:.2f}')         
    dis =  eigvec_to_eigdis(evecs[mode],HS_STR)
    dis = amp*np.array(dis.real).reshape(natom,3)  
    if get_disp:
        dist_str = dis
    else:
        dist_str = Atoms(numbers = Phonon_obj.unitcell.get_atomic_numbers(),scaled_positions=Phonon_obj.unitcell.scaled_positions+dis,cell = Phonon_obj.unitcell.cell)
    return(dist_str,freq)


def plot_figs_Dips(modes,Phonon_obj,str_file,qpt = [0,0,0],amp=1,prj_dirs = [0,1,2,3],path='./'):  
    vec_type=3  
    os.chdir(path)        
    evecs,freqs = get_Evecs_Freqa_from_phonon_obj(Phonon_obj,qpt=qpt)
    # print(10*'---','EVECS LENGTH')
    # print(len(evecs))
    # Phonon_obj.set_irreps([qpt])
    # Labels = Phonon_obj._irreps._get_ir_labels()
    HS_STR = Atoms(numbers = Phonon_obj.unitcell.get_atomic_numbers(),scaled_positions=Phonon_obj.unitcell.scaled_positions,cell = Phonon_obj.unitcell.cell)
    cell_p_Found = False
    natom = len(Phonon_obj.unitcell.get_atomic_numbers())             
    pol_consts = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
    chars_plot = ['x_','y_','z_','']
    vst_in=open(str_file,'r')
    for mod in modes:  
        freq = freqs[mod]
        print(mod,f'   >>> Freq :  {freq:.2f} ', freq)
        os.system(f'mkdir Mod_{mod}_freq_{freq:.2f}')         
        dis =  eigvec_to_eigdis(evecs[mod],HS_STR)
        dis = np.array(dis.real).reshape(natom,3)        
        for prj_dir,pol_const in zip(prj_dirs,pol_consts):
            vst_in.seek(0,0)
            vst_ot=open(f'Mod_{mod}_freq_{freq:.2f}/Mode_{mod}_{chars_plot[prj_dir]}proj.vesta','w')
            disn=np.zeros((natom,3))
            l=vst_in.readline()
            vst_ot.write(l)
            while l:
                l=vst_in.readline()
                vst_ot.write(l)
                if l.strip().startswith('CELLP') and not cell_p_Found:
                    l=vst_in.readline()
                    lp=[float(l.strip().split()[0]),float(l.strip().split()[1]),float(l.strip().split()[2])]
                    vst_ot.write(l)
                    for i in range(len(dis)):
                        disn[i,:]=dis[i,0]*lp[0],dis[i,1]*lp[1],dis[i,2]*lp[2]
                        if i ==0:
                            bg_vec=(disn[i,0])**2+(disn[i,1])**2+(disn[i,2])**2
                        elif (bg_vec<(disn[i,0])**2+(disn[i,1])**2+(disn[i,2])**2):
                            bg_vec=(disn[i,0])**2+(disn[i,1])**2+(disn[i,2])**2   
                    mdls=amp/(bg_vec**0.5)  
                if l.strip().startswith('BOUND'):
                    vst_ot.write(' -0.1      1.1      -0.1      1.1      -0.1      1.1   ')

                if l.strip().startswith('VECTR'):                                              
                    for i in range(len(dis)):
                        vst_ot.write(f'   {i+1}    {disn[i,0]*mdls*pol_const[0]:.6f}    {disn[i,1]*mdls*pol_const[1]:.6f}    {disn[i,2]*mdls*pol_const[2]:.6f} {0}\n')
                        vst_ot.write(' {} 0 0 0 0\n'.format(i+1))
                        vst_ot.write(' 0 0 0 0 0\n')
                if l.strip().startswith('VECTT'):                
                    if prj_dir==3:
                        for i in range(len(dis)):
                            vst_ot.write(' {} 0.300 255  0  0  {}\n'.format(i+1,vec_type))
                    else:
                        for i in range(len(dis)):
                            if (disn[i,prj_dir]*mdls>0):
                                vst_ot.write(' {} 0.300 255  0  0  {}\n'.format(i+1,vec_type))
                            else:
                                vst_ot.write(' {} 0.300 0  0  255  {}\n'.format(i+1,vec_type))
            vst_ot.close() 

    vst_in.close()

def my_timer(my_Func):
    def time_wrapper(*args,**kwrags):
        print(f'Function {my_Func.__name__} ...')
        t0 = time.perf_counter()
        result = my_Func(*args,**kwrags)
        t1 = time.perf_counter()
        print(f'time(s) spend in {my_Func.__name__} is :',f'{t1-t0:.2f}')
        return(result)
    return(time_wrapper)

# def plot_decorator(plt):
#     SMALL_SIZE = 13
#     MEDIUM_SIZE = 13
#     BIGGER_SIZE = 13
#     my_dpi=300
#     plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=11)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def ase_to_sclup(atoms,SC_MAT,output='sclup_strc.out',SCLL_fnl_str=None):  
    tag_atm = []
    SC_MAT = np.array(SC_MAT)
    for i in range(len(atoms)):
        tag_atm.append(i+1)
    atoms.set_array('tag', np.array(tag_atm))
    super_cell,lat_points = my_SClls.make_supercell(atoms, SC_MAT, if_return_lp=True)
    chem_sym = atoms.get_chemical_symbols()
    num_atom = len(chem_sym)
    type_atom = len(set(chem_sym))
    dict_typ = OrderedDict()
    isym = 0
    typ_list = ''
    for sym in chem_sym:
        if sym not in dict_typ.keys():
            isym += 1
            dict_typ[sym] = isym
            typ_list += f'{sym}        ' 
    my_latice_point = []
    for my_lp in lat_points:
        for i in range(len(atoms)):
            tmp_lp = list(map(int,np.dot(SC_MAT,my_lp)))
            my_latice_point.append(tmp_lp)
    out_pt = open(output,'w')
    out_pt.write(f'    {int(SC_MAT[0,0])}    {int(SC_MAT[1,1])}    {int(SC_MAT[2,2])}\n')
    out_pt.write(f'    {num_atom}    {type_atom}\n')
    out_pt.write(typ_list+'\n')
    SC_cell = super_cell.get_cell()
    if SCLL_fnl_str == None:
        for i in range(3):
            for j in range(3):
                out_pt.write(f'    {SC_cell[i,j]/Bohr:0.9E}')
                position_to_write = super_cell.get_positions()
    else:
        scl_fnl = SCLL_fnl_str.get_cell()
        strain_mat = np.dot(np.linalg.inv(SC_cell),scl_fnl)-np.eye(3)
        voigt_str = missfit_terms.get_strain(strain=strain_mat)
        final_strc = get_mapped_strcs(SCLL_fnl_str, super_cell)
        scld_position_diff = final_strc.get_scaled_positions()-super_cell.get_scaled_positions()
        position_to_write = np.dot(scld_position_diff,SC_cell)
        for strn in voigt_str:
            out_pt.write(f'    {strn:0.9E}')                 
    out_pt.write('\n')
    symbls_SC = super_cell.get_chemical_symbols()
    tag_atom = super_cell.get_array('tag')
    for ipos,pos in enumerate(position_to_write):
        out_pt.write(f'    {my_latice_point[ipos][0]}    {my_latice_point[ipos][1]}    {my_latice_point[ipos][2]}    {tag_atom[ipos]}    {dict_typ[symbls_SC[ipos]]}    {pos[0]/Bohr:0.9E}    {pos[1]/Bohr:0.9E}    {pos[2]/Bohr:0.9E} \n')
    out_pt.close()

def sclup_to_ase(sclup_file):
    inpt_f = open(sclup_file,'r')
    l = inpt_f.readline().split()
    SC_mat = list(map(int,l))
    l = inpt_f.readline().split()
    dta = list(map(int,l))
    num_atom_UC,atm_typ = dta[0],dta[1]
    chem_sym = inpt_f.readline().split()
    cell_strc = np.array(list(map(float,inpt_f.readline().split()))).reshape(3,3)*Bohr
    num_atm = num_atom_UC*(SC_mat[0]*SC_mat[1]*SC_mat[2])
    pos = []
    sym = []
    l = inpt_f.readline().split()
    natm_tmp = 0
    while l:  
        natm_tmp += 1      
        dta = list(map(float,l))
        pos.append([dta[5]*Bohr,dta[6]*Bohr,dta[7]*Bohr])
        sym.append(chem_sym[int(dta[4])-1])
        l = inpt_f.readline().split()
    numbers = [atomic_numbers[i] for i in sym]
    if natm_tmp != num_atm:
        raise('sclup_strc_to_atoms : Number of atoms are different')
    atoms = Atoms(numbers=numbers,positions=pos, cell=cell_strc, pbc=True)
    return(atoms)
 
class Get_Pol():
    def __init__(self,Str_ref,BEC_ref,proj_dir = [1,1,1],cntr_at = ['Ti'],trans_mat = [1,1,1],dim_1=1,fast=False):
        self.wght = {'O':1/6,'Ba':1/8,'Sr':1/8,'Pb':1/8,'Ti':1} 
        self.proj_dir=proj_dir
        self.cntr_at=cntr_at
        self.dim_1=dim_1
        self.BEC_ref=BEC_ref 
        self.Str_ref= Str_ref   
        self.trans_mat = trans_mat
        self.get_NL()
        self.fast = fast
    @my_timer
    def get_NL(self):
        self.chmsym = self.Str_ref.get_chemical_symbols()
        self.Ti_indxs = []
        for atm_indx,atm_sym in enumerate(self.chmsym):
            if atm_sym in self.cntr_at:
                self.Ti_indxs.append(atm_indx)
        cutof_dic = {}
        ABC_SL = self.Str_ref.cell.cellpar()[0:3]
        ABC_SL = [ABC_SL[0]/self.trans_mat[0],ABC_SL[1]/self.trans_mat[1],ABC_SL[2]/self.trans_mat[2]] 
        cutof_dic['Ti'] = ABC_SL[0]/2 - 0.5
        cutof_dic['Ba'] = self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
        cutof_dic['Sr'] = self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
        cutof_dic['Pb'] = self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
        cutof_dic['O'] = self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
        sccuof = [cutof_dic[sym] for sym in self.chmsym]        
        self.mnl = NeighborList(sccuof,sorted=False,self_interaction = False,bothways=True)
        self.mnl.update(self.Str_ref)
    @my_timer
    def get_pol_mat(self,Str_dist):  
        Prim_Str_Hist = Atoms(numbers=self.Str_ref.get_atomic_numbers(),scaled_positions=self.Str_ref.get_scaled_positions(), cell=Str_dist.get_cell(), pbc=True)
        ABC_SL0=Prim_Str_Hist.cell.cellpar()[0:3]
        v0 = np.linalg.det(Prim_Str_Hist.get_cell())/(self.trans_mat[0]*self.trans_mat[1]*self.trans_mat[2])
        ABC_SL = [ABC_SL0[0]/self.trans_mat[0],ABC_SL0[1]/self.trans_mat[1],ABC_SL0[2]/self.trans_mat[2]]   
        ref_positions =   Prim_Str_Hist.get_positions() 

        if self.fast:
            Fnl_str=map_strctures(Str_dist,Prim_Str_Hist,tol=0.2)
        else:
            Fnl_str=get_mapped_strcs(Str_dist,Prim_Str_Hist,Ret_index=False)
        
        # cm_ref = Prim_Str_Hist.get_center_of_mass()
        # cm_fnl = Fnl_str.get_center_of_mass()
        disp = Fnl_str.get_positions()-ref_positions
        disp_proj = self.proj_dir*disp
        pol_mat = np.zeros((self.trans_mat[0],self.trans_mat[1],self.trans_mat[2],3))
        for aa in self.Ti_indxs:
            NNs,offsets = self.mnl.get_neighbors(aa)
            if ref_positions[aa,2] <= ABC_SL[2]*self.dim_1 :
                k=0
            else:
                k=1
            
            a,b,c=int(abs(ref_positions[aa,0]-ref_positions[1,0])/(ABC_SL[0]*0.99)),int(abs(ref_positions[aa,1]-ref_positions[1,1])/(ABC_SL[1]*0.99)),int(abs(ref_positions[aa,2]-ref_positions[1,2])/(ABC_SL[2]*0.99))
            # print(a,b,c,ref_positions[aa,2],ref_positions[1,2])
            NNs = np.append(aa,NNs)
            for j in NNs:                
                pol_mat[a,b,c,:] += self.wght[self.chmsym[j]]*np.dot(disp_proj[j],self.BEC_ref[k,j%5,:,:]) 
                
        pol_mat=16*pol_mat/v0
        return(pol_mat)

    @my_timer
    def get_tot_pol(self,Str_dist):  
        Prim_Str_Hist = Atoms(numbers=self.Str_ref.get_atomic_numbers(),scaled_positions=self.Str_ref.get_scaled_positions(), cell=Str_dist.get_cell(), pbc=True)
        ABC_SL0=Prim_Str_Hist.cell.cellpar()[0:3]
        v0 = np.linalg.det(Prim_Str_Hist.get_cell()) #/(self.trans_mat[0]*self.trans_mat[1]*self.trans_mat[2])
        ABC_SL = [ABC_SL0[0]/self.trans_mat[0],ABC_SL0[1]/self.trans_mat[1],ABC_SL0[2]/self.trans_mat[2]]   
        ref_positions =   Prim_Str_Hist.get_positions() 

        if self.fast:
            Fnl_str=map_strctures(Str_dist,Prim_Str_Hist,tol=0.2)
        else:
            Fnl_str=get_mapped_strcs(Str_dist,Prim_Str_Hist,Ret_index=False)

        disp = Fnl_str.get_positions()-ref_positions
        disp_proj = self.proj_dir*disp
        tot_pol = np.zeros(3)

        for j in range(len(disp_proj)):
            if ref_positions[j%5,2] <= ABC_SL[2]*self.dim_1 :
                k=0
            else:
                k=1
            tot_pol[:] += self.wght[self.chmsym[j]]*np.dot(disp_proj[j],self.BEC_ref[k,j%5,:,:])                 
        tot_pol=16*tot_pol/v0
        return(tot_pol)

    def get_layer_pol(self,Str_dist):

        pass

def get_BEC_SCLL(SCLL,SCLL_MAT,har_xml):
    my_xml_sys = xml_io.Xml_sys_reader(har_xml)
    my_xml_sys.get_ase_atoms()
    UC_atms = my_xml_sys.ase_atoms
    my_SCLL = make_supercell(UC_atms,SCLL_MAT)
    indexes=get_mapped_strcs(SCLL,my_SCLL,Ret_index=True)
    my_xml_sys.get_BEC()
    my_xml_sys.get_eps_inf()
    BEC = [my_xml_sys.BEC[key] for key in my_xml_sys.BEC.keys()]
    my_xml_sys.get_eps_inf()
    eps_inf = my_xml_sys.eps_inf
    BEC_SCLL = []
    for i in indexes:   #FIXME
        BEC_SCLL.append(BEC[i%(len(UC_atms))])
    return(BEC_SCLL,eps_inf)

def eigvec_eigdis(eigvec,atoms):
    natm=len(atoms)
    masses = atoms.get_masses()
    eig_temp=eigvec.reshape(3*natm)
    eigdis=np.zeros(3*natm,dtype=complex)
    for i in range(natm):
        for j in range(3):
            cntr = i*3+j
            eigdis[cntr]=eig_temp[cntr]*(masses[i]**0.5)
    return(eigvec.reshape(natm,3))

def get_image_dta(nc_path,istep=-1):
    NC_DTA = mync.hist_reader(nc_path)
    etotal =NC_DTA.get_etotal()[istep]
    atomic_number = NC_DTA.get_tot_numbers()
    Nimages = len(etotal)
    xred_f = NC_DTA.get_xred()[istep]
    RSET_f_Bhr = NC_DTA.get_RSET()[istep]    
    image_atoms = []
    for ii in range(Nimages):
        atms_n = Atoms(numbers=atomic_number,cell=RSET_f_Bhr[ii]*Bohr,scaled_positions=xred_f[ii],pbc=True)
        #write(f'POSCAR_{ii}',atms_n,format='vasp')
        image_atoms.append(atms_n) 
    return(image_atoms,etotal) 

def eigvec_to_eigdis(eigvec,atoms):
    natm=len(atoms)
    masses = atoms.get_masses()
    eig_temp=eigvec.reshape(3*natm)
    eigdis=np.zeros(3*natm,dtype=complex)
    for i in range(natm):
        for j in range(3):
            cntr = i*3+j
            eigdis[cntr]=eig_temp[cntr]/(masses[i]**0.5)
    return(eigdis.reshape(natm,3))

def eigdis_to_eigvec(eigdis,atoms):
    natm=len(atoms)
    masses = atoms.get_masses()
    eig_temp=eigdis.reshape(3*natm)
    eigvec=np.zeros(3*natm,dtype=complex)
    for i in range(natm):
        for j in range(3):
            cntr = i*3+j
            eigvec[cntr]=eig_temp[cntr]*(masses[i]**0.5)
    return(eigvec.reshape(natm,3))

def get_Evecs_Freqa_from_phonon_obj(Phonon,qpt=[0,0,0]):
    natom = len(Phonon.unitcell.numbers)
    evecs0 = np.array(Phonon.get_frequencies_with_eigenvectors(qpt)[1])
    Freqs = np.array(Phonon.get_frequencies_with_eigenvectors(qpt)[0])
    evecs = []
    for i in range(3*natom):
        evecs.append(np.reshape(evecs0[:,i],(natom,3)))
    return(evecs,Freqs)


@my_timer
def get_mapped_strcs(str_to_be_map,str_to_map_to,Ret_index=False):
    natom = len(str_to_map_to.get_scaled_positions())
    natom2 = len(str_to_be_map.get_scaled_positions())
    # if natom!=natom2:
    #     print('wrong structures')
    #     return(0)
    str_cell = np.array(str_to_be_map.get_cell())
    map_index = np.zeros(natom,dtype=int)
    xred_maped = np.zeros((natom,3))
    for ia,xred_a in enumerate(str_to_map_to.get_scaled_positions()):
        diff_xred = np.zeros((natom,3))
        shift = np.zeros((natom,3))
        list_dist = np.zeros(natom)
        list_absdist = np.zeros((natom,3))
        diff_xred = str_to_be_map.get_scaled_positions()-xred_a
        for ib,b in enumerate(str_to_be_map.get_scaled_positions()):
            if diff_xred[ib,0] > 0.5:
                diff_xred[ib,0] = 1 - diff_xred[ib,0]
                shift[ib,0] = -1
            if diff_xred[ib,1] > 0.5:
                diff_xred[ib,1] = 1 - diff_xred[ib,1]
                shift[ib,1] = -1
            if diff_xred[ib,2] > 0.5:
                diff_xred[ib,2] = 1 - diff_xred[ib,2]
                shift[ib,2] = -1
            if diff_xred[ib,0] < -0.5:
                diff_xred[ib,0] = -1-diff_xred[ib,0]
                shift[ib,0] = 1
            if diff_xred[ib,1] < -0.5:
                diff_xred[ib,1] = -1-diff_xred[ib,1]
                shift[ib,1] = 1
            if diff_xred[ib,2] < -0.5:
                diff_xred[ib,2] = -1-diff_xred[ib,2]
                shift[ib,2] = 1
            list_absdist[ib,:] = np.dot(str_cell,diff_xred[ib,:])
            list_dist[ib] = np.sqrt(np.dot(list_absdist[ib,:],np.transpose(list_absdist[ib,:])))

        map_index[ia] = np.where(list_dist==min(list_dist))[0][0]
        xred_maped[ia,:] = str_to_be_map.get_scaled_positions()[map_index[ia],:] + shift[map_index[ia],:]

    if Ret_index:
        return(map_index)

    maped_str = Atoms(numbers = str_to_map_to.get_atomic_numbers(),scaled_positions = xred_maped, cell = str_to_be_map.get_cell())
    return(maped_str)

# @my_timer
def map_strctures(str_1,str_2,tol=0.5):  # the ordering is correct
    red_1 = str_1.get_scaled_positions()
    red_2 = str_2.get_scaled_positions()
    tmp_red1 = np.zeros((len(red_1),3))
    for i in range(len(red_1)):
        for j in range(3):
            diff = red_1[i,j]-red_2[i,j]
            if abs(diff) > tol:
                if diff>0:
                #    print('Here  1')
                   tmp_red1[i,j] = red_1[i,j]-1
                elif diff<0:
                #    print('Here  2')
                   tmp_red1[i,j] = 1+red_1[i,j]
            else:
               tmp_red1[i,j] = red_1[i,j]
    Nstr_1 = Atoms(numbers=str_1.get_atomic_numbers(), scaled_positions=tmp_red1, cell=str_1.get_cell())
    return(Nstr_1)

def my_wrap(atoms,tol=0.02):
    scld_pos=atoms.get_scaled_positions()
    trans_scld_pos=[]
    for i in range(atoms.get_global_number_of_atoms()):
        vec=np.zeros((3))
        for j in range(3):
            if scld_pos[i,j]>=1-tol:
                #print('transforming')
                vec[j]=scld_pos[i,j]-1
            elif scld_pos[i,j] < -1+tol:
                #print('transforming')
                vec[j]=scld_pos[i,j]+1
            else:
                vec[j]=scld_pos[i,j]
        trans_scld_pos.append(vec)
    Trans_atms = Atoms(numbers = atoms.get_atomic_numbers(),scaled_positions = trans_scld_pos, cell = atoms.get_cell() , pbc=True)
    return(Trans_atms)


def find_disp(str_dist,str_ref):
    Nstr_dist = get_mapped_strcs(str_dist,str_ref,Ret_index=False)
    disp = Nstr_dist.get_positions()-str_ref.get_positions()
    return(np.array(disp))


def my_make_SL(a1,a2):
    cell_1=a1.get_cell()
    cell_2=a2.get_cell()
    cll_diff0=0 #(cell_2[0][0]-cell_1[0][0])
    cll_diff1=0 #(cell_2[1][1]-cell_1[1][1])
    cll_diff2=0 #(cell_2[2][2]-cell_1[2][2])
    shift=[0,0,0]
    cell_SL=[cell_1[0][0],cell_1[1][1],cell_1[2][2]+cell_2[2][2]]
    pos1=a1.get_positions()
    pos2=a2.get_positions()
    car_SL=[]
    nij=[0,0,0]
    for i in pos1:
        car_SL.append(i)
    for i in pos2:
        for j in range(3):
            if abs(i[j]-cell_2[j][j]) < 0.1:
                nij[j]=i[j]-cell_2[j][j]
            else:
                nij[j]=i[j]
        car_SL.append([nij[0],nij[1],nij[2]+cell_1[2][2]])
        #car_SL.append([i[0],i[1],i[2]+cell_1[2][2]])
    numbers1=a1.get_atomic_numbers()
    numbers2=a2.get_atomic_numbers()
    numbers_SL= [*numbers1, *numbers2]
    my_SL=Atoms(numbers=numbers_SL,positions=car_SL, cell=cell_SL, pbc=True)
    return(my_SL)

def ref_SL(atm_1,SC_mat1,atm_2,SC_mat2):
    tmp_SC1=make_supercell(atm_1,SC_mat2)
    mySC1=make_supercell(atm_1,SC_mat1)
    SL1=my_make_SL(mySC1,tmp_SC1)
    ABC_SL1=SL1.cell.cellpar()[0:3]
    ScPOS_SL1=SL1.get_scaled_positions()
    tmp_SC2=make_supercell(atm_2,SC_mat1)
    mySC2=make_supercell(atm_2,SC_mat2)
    SL2=my_make_SL(mySC2,tmp_SC1)
    ABC_SL2=SL2.cell.cellpar()[0:3]
    ScPOS_SL2=SL2.get_scaled_positions()
    cell_1=mySC1.get_cell()
    cell_2=mySC2.get_cell()
    weight = 1/(cell_1[2][2]+cell_2[2][2])
    cell_a = weight*(cell_1[2][2]*ABC_SL1[0]+cell_2[2][2]*ABC_SL2[0])
    cell_b = weight*(cell_1[2][2]*ABC_SL1[1]+cell_2[2][2]*ABC_SL2[1])
    cell_c = weight*(cell_1[2][2]*ABC_SL1[2]+cell_2[2][2]*ABC_SL2[2])
    cell_SL=[cell_a, cell_b, cell_c]
    SL_SCp = []
    for i in range(len(ScPOS_SL1)):
        new_pa = weight*(cell_1[2][2]*ScPOS_SL1[i][0]+cell_2[2][2]*ScPOS_SL2[i][0])
        new_pb = weight*(cell_1[2][2]*ScPOS_SL1[i][1]+cell_2[2][2]*ScPOS_SL2[i][1])
        new_pc = weight*(cell_1[2][2]*ScPOS_SL1[i][2]+cell_2[2][2]*ScPOS_SL2[i][2])
        SL_SCp.append([new_pa,new_pb,new_pc])
    numbers1 = mySC1.get_atomic_numbers()
    numbers2 = mySC2.get_atomic_numbers()
    numbers_SL= [*numbers1, *numbers2]
    my_SL = Atoms(numbers=numbers_SL,scaled_positions = SL_SCp, cell = cell_SL, pbc=True)
    return(my_SL)

def find_index(Mcor,Vec,tol=0.001):
    index=-1
    for m in range(len(Mcor)):
        flg=[]
        for v in range(len(Mcor[m])):
            diff=Mcor[m,v]-Vec[v]
            if abs(diff) < tol:
                flg.append(True)
            else:
                flg.append(False)
        if all (flg) :
            index=int(m)
    return(index)

def mapping_index(str_to_bemaped, str_to_map_to):
    POS_SC1 = str_to_bemaped.get_positions()
    POS_SC2 = str_to_map_to.get_positions()
    indx_list2=[]
    for i in range(len(POS_SC1)):
        indx=find_index(POS_SC2,POS_SC1[i],tol = 0.1)
        if indx==-1:
            print(POS_SC1[0])
            print(POS_SC2[0])
            print('---mapping failed')
            #return(-1)
            break
        indx_list2.append(indx)
    return(indx_list2)

def remap_fset(str_to_bemaped, str_to_map_to,FSET):   ########
    indx_list = get_mapped_strcs(str_to_map_to,str_to_bemaped,Ret_index=True)
    #indx_list = mapping_index(str_to_bemaped, str_to_map_to)
    #print(indx_list1)
    #print(indx_list)
    FC_trans=[]
    for i, ii in enumerate(FSET):
        tmp_FC=np.zeros((len(str_to_bemaped.get_positions()),3))
        for j, jj in enumerate(indx_list):    ######
            tmp_FC[j,:]=ii[jj,:]*Hatoev
        FC_trans.append(tmp_FC)
    return(FC_trans)

def find_ref_centr(SC_ase):
    prim_cell = spg.get_symmetry_dataset(SC_ase)['std_lattice']
    prim_scld_pos = spg.get_symmetry_dataset(SC_ase)['std_positions']
    prim_pos = np.dot(prim_scld_pos, prim_cell)
    super_cell = spg.get_symmetry_dataset(SC_ase)['transformation_matrix']
    transform = []
    for i in super_cell.flatten():
        if i%2 == 0 and i!=0:
            transform.append(int(i/2))
        elif i%2 != 0 and i!=0:
            transform.append(int((i-1)/2))
    cent_pos = np.zeros((len(prim_pos)))
    for i in range(len(prim_pos)):
        my_pos = [transform[0]*prim_cell[0][0]+prim_pos[i][0],transform[1]*prim_cell[1][1]+prim_pos[i][1],transform[2]*prim_cell[2][2]+prim_pos[i][2]]
        cent_pos[i] = find_index(SC_ase.get_positions(),my_pos)
    return(cent_pos)

def get_FC_dic_from_phonon(phonon):
    FC = phonon.force_constants
    SC = phonon.supercell
    SC_ase = Atoms(numbers = SC.get_atomic_numbers(), scaled_positions=SC.get_scaled_positions(),cell = SC.cell, pbc = True)
    maped_to_prim = spg.get_symmetry_dataset(SC_ase)['mapping_to_primitive']
    prim_cell = spg.get_symmetry_dataset(SC_ase)['std_lattice']
    cen_pos = find_ref_centr(SC_ase)
    cen_pos = [int(i) for i in cen_pos]
    natom_sc = SC_ase.get_global_number_of_atoms()
    pos_SC = SC_ase.get_positions()
    tot_SC_FCDIC={}
    my_map = {}
    for i in (cen_pos):
        my_map[maped_to_prim[i]]=i

    for i in (cen_pos):
        for j in range(natom_sc):
            tmp_pos = pos_SC[my_map[maped_to_prim[j]]]
            dist = pos_SC[j]-tmp_pos
            my_cell = [int(dist[0]/(0.99*prim_cell[0,0])),int(dist[1]/(0.99*prim_cell[1,1])),int(dist[2]/(0.99*prim_cell[2,2]))]
            UC_key='{}_{}_{}_{}_{}'.format(maped_to_prim[i],maped_to_prim[j],my_cell[0],my_cell[1],my_cell[2])
            if UC_key not in (tot_SC_FCDIC.keys()):
                tot_SC_FCDIC[UC_key] = FC[i,j]*0.010290311970481884
    return(tot_SC_FCDIC)

def get_phonon(har_xml, Anh_xml, phon_scll=None, str_ten=np.zeros((3,3)), factor=AbinitToTHz*thz_cm, ngqpt=[4,4,4], nproc=1, my_EXEC='MB_Feb21',asr=2,
               dipdip=1,path='/tmp/',UC=None, SIM_cell=None, disp_amp=0.01, Files_prefix='phonon', opt_cell=2,cal_nac=False,BEC_in=None,eps_inf_in=None,dipdip_range=None):
    if UC is None:
        try:
            my_xml_sys = xml_io.Xml_sys_reader(har_xml)
            my_xml_sys.get_ase_atoms()
            xml_atms = my_xml_sys.ase_atoms
            if cal_nac:
                my_xml_sys.get_BEC()
                my_xml_sys.get_eps_inf()
                BEC = [my_xml_sys.BEC[key] for key in my_xml_sys.BEC.keys()]
                my_xml_sys.get_eps_inf()
                eps_inf = my_xml_sys.eps_inf
        except:
            my_ddb = My_ddb.ddb_reader(har_xml)
            my_ddb.get_atoms()
            xml_atms = my_ddb.atoms            
    else:
        xml_atms = UC
        if cal_nac:
            if BEC_in is None and eps_inf_in is None:
                my_xml_sys = xml_io.Xml_sys_reader(har_xml)
                BEC = [my_xml_sys.BEC[key] for key in my_xml_sys.BEC.keys()]
                my_xml_sys.get_eps_inf()
                eps_inf = my_xml_sys.eps_inf
            else:
                BEC = BEC_in
                eps_inf = eps_inf_in

    if phon_scll is None:
        phon_scll=[[ngqpt[0],0,0],[0,ngqpt[1],0],[0,0,ngqpt[2]]]
    if SIM_cell is None:
        SIM_cell=phon_scll

    # print(xml_atms)
    # print(10*'---')
    # print(xml_atms.get_cell())
    unitcell= PhonopyAtoms(symbols=xml_atms.get_chemical_symbols(),
           cell = xml_atms.get_cell()+np.dot(str_ten,xml_atms.get_cell()),
           scaled_positions = xml_atms.get_scaled_positions())
    # print(10*'**')  #
    print('cell_of_phonons     \n',xml_atms.get_cell()+np.dot(str_ten,xml_atms.get_cell()))

    phonon = Phonopy(unitcell, phon_scll,factor = factor )

    if cal_nac:
        '>>>>>>>>>>>>>>>>>>>   setting NAC'
        NAC={'born':BEC,
                    'dielectric':eps_inf,
                   'factor':14.4 }    ### FACTORS??                
        phonon.nac_params = NAC

    print('symmetry in phonopy  >> ',phonon.get_symmetry().get_pointgroup(),phonon._symprec)
    #Supercells with displacemetns
    phonon.generate_displacements(distance = disp_amp, is_plusminus  = "True")

    supercells = phonon.supercells_with_displacements
    Super_cell = phonon.supercell
    path_phon = 'tmp_phonon'
    
    sim_path = f'{path}/{path_phon}'
    os.makedirs(sim_path,exist_ok=True)
    os.chdir(sim_path)

    print(10*'--')
    print('number of Disps : ',len(supercells))
    for sc,isc in enumerate(supercells):
        if sc ==0:
            ph_ase_scll = Atoms(numbers=isc.get_atomic_numbers(),scaled_positions=isc.get_scaled_positions(), cell=isc.get_cell(), pbc=True)
            ph_ase_scll.wrap()
        tmp_ase_scll= Atoms(numbers=isc.get_atomic_numbers(),scaled_positions=isc.get_scaled_positions(), cell=isc.get_cell(), pbc=True)
        #tmp_ase_scll=get_mapped_strcs(tmp_ase_scll,Super_cell,Ret_index=False)
        tmp_ase_scll=map_strctures(tmp_ase_scll,Super_cell)
        #tmp_ase_scll.wrap()
        #write(f'POSCAR_{sc}',tmp_ase_scll,format='vasp')
        mync.write_hist(tmp_ase_scll,'Phonon_dist_HIST.nc',sc)

    # MB_SIMS
    my_sim = MB_sim(my_EXEC, har_xml, Anhar_coeffs=Anh_xml, ngqpt=ngqpt, ncell=[int(SIM_cell[0][0]),int(SIM_cell[1][1]),int(SIM_cell[2][2])], ncpu=nproc, test_set='no',prefix = Files_prefix)
    my_sim.test_dta()
    my_sim.inpt['asr'] = asr
    if dipdip_range is not None:
        my_sim.inpt['dipdip_range'] = dipdip_range
    my_sim.inpt['dipdip'] = dipdip    
    my_sim.inpt['prt_model']  = 2
    my_sim.inpt['dipdip_prt']  =  1
    # RUNING MB
    os.chdir(sim_path)
    my_sim.test_set = 'Phonon_dist_HIST.nc'
    my_sim.inpt['optcell'] = opt_cell
    my_sim.write_run_data()
    os.system(f'sh MB_run.sh')  
    my_nc = mync.hist_reader(f'{sim_path}/ph_test.nc')
    str_from_nc = my_nc.get_ase_str(0)
    FSET= my_nc.get_fcart()

    ##remapping FSET
    # print(10*'111')
    FC_trans = remap_fset(ph_ase_scll,str_from_nc,FSET)
    # print(10*'222')
    phonon.set_forces(FC_trans)
    phonon.produce_force_constants()
    if asr:
        phonon.symmetrize_force_constants()   #ASR
    return(phonon)

def plot_phonon(phonon,name='plt',cpath='./'):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 10
    my_dpi=300
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=11)            # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    os.chdir(f'{cpath}')
    phonon.symmetrize_force_constants()   #ASR
    phonon.auto_band_structure()

    phonon.plot_band_structure().savefig('{}/{}.png'.format(cpath,name))

def get_dict_ifc(IFC, loc=True):
    if loc:
        tmp_FC = IFC.ifc_cart_coord_short_range
        shape = np.shape(tmp_FC)
    else:
        tmp_FC = IFC.ifc_cart_coord
    ifc_coor = IFC.atoms_cart_coord
    atm_indx = IFC.neighbours_indices
    prim_coor = IFC.structure.cart_coords/Bohr
    prim_cell = IFC.structure.lattice.matrix/Bohr
    FCDIC = {}
    for atm_a in range(len(tmp_FC)):
        for atm_b in range(len(tmp_FC[atm_a])):
            indx_b = atm_indx[atm_a,atm_b]
            coor_b = ifc_coor[atm_a,atm_b]
            dist = coor_b-prim_coor[indx_b]  #####
            my_cell =  np.dot(dist,np.linalg.inv(prim_cell))  #[int(dist[0]/(0.99*prim_cell[0,0])),int(dist[1]/(0.99*prim_cell[1,1])),int(dist[2]/(0.99*prim_cell[2,2]))]
            UC_key='{}_{}_{}_{}_{}'.format(atm_a,indx_b,(my_cell[0]),(my_cell[1]),(my_cell[2]))
            if UC_key not in (FCDIC.keys()):
                FCDIC[UC_key] =  np.matrix.transpose(tmp_FC[atm_a,atm_b])

    return(FCDIC)

def get_pair_key(IFC, loc=True):
    if loc:
        tmp_FC = IFC.ifc_cart_coord_short_range
        shape = np.shape(tmp_FC)
    else:
        tmp_FC = IFC.ifc_cart_coord
    ifc_coor = IFC.atoms_cart_coord
    atm_indx = IFC.neighbours_indices
    prim_coor = IFC.structure.cart_coords/Bohr
    prim_cell = IFC.structure.lattice.matrix/Bohr
    key_pair = {}
    for atm_a in range(len(tmp_FC)):
        for atm_b in range(len(tmp_FC[atm_a])):
            indx_b = atm_indx[atm_a,atm_b]
            coor_b = ifc_coor[atm_a,atm_b]
            dist = coor_b-prim_coor[indx_b]  #####
            my_cell = [int(dist[0]/(0.99*prim_cell[0,0])),int(dist[1]/(0.99*prim_cell[1,1])),int(dist[2]/(0.99*prim_cell[2,2]))]
            UC_key='{}_{}_{}_{}_{}'.format(atm_a,indx_b,my_cell[0],my_cell[1],my_cell[2])
            if UC_key not in (FCDIC.keys()):
                key_pair[UC_key] = f'{atm_a} {atm_b}'

    return(key_pair)

def xml_IFC_phonon(har_xml,phonon,ngqpt,str_ten,out_put,UC=None,BEC_in=None,eps_inf_in=None,tot_fc=True,mod_cell=False,ddb_path='/tmp/',ddb_0=None):
    ''' converting phonon object to DDB using the Born Effective charges as in har_xml the phonons are with no NAC'''

    if BEC_in is not None:
        BEC = BEC_in
    if eps_inf_in is not None:
        eps_inf = eps_inf_in
    
    if UC is not None:
        xml_atms=UC
    else:        
        my_xml_sys = xml_sys(har_xml)
        my_xml_sys.get_ase_atoms()
        xml_atms = my_xml_sys.ase_atoms
        my_xml_sys.get_BEC()
        BEC = [my_xml_sys.BEC[key] for key in my_xml_sys.BEC.keys()]
        my_xml_sys.get_eps_inf()
        eps_inf = my_xml_sys.eps_inf
    
    if mod_cell:
        my_cell = xml_atms.get_cell()+np.dot(str_ten,xml_atms.get_cell())
    else:
        my_cell = xml_atms.get_cell()

    ase_UC = Atoms(numbers=xml_atms.numbers,
           cell = my_cell,
           scaled_positions = xml_atms.get_scaled_positions())
    NAC={'born':BEC,
                 'dielectric':eps_inf}
    write('DDB_STR.cif',ase_UC,format='cif')
    UC_abistruct = abipy.abio.factories.Structure.from_ase_atoms(ase_UC)
    if ddb_0 ==None:
        new_ddb = cnvtr.phonopy_to_abinit(UC_abistruct, phonon.get_supercell_matrix(), ddb_path , ngqpt=ngqpt,
                      force_constants=phonon.force_constants, born=NAC)
    else:
        new_ddb = ddb_0
    IFC = new_ddb.anaget_ifc()
    loc_ifc_dic = get_dict_ifc(IFC, loc=True)
    tot_ifc_dic = get_dict_ifc(IFC, loc=False)
    myxml_SC = my_sc_maker(har_xml,np.eye(3))
    myxml_SC.set_loc_UCFC(loc_ifc_dic)
    myxml_SC.set_tot_UCFC(tot_ifc_dic,flag = tot_fc)
    myxml_SC.set_SC(ase_UC)
    myxml_SC.reshape_FCDIC()
    myxml_SC.write_xml(out_put)


def DDB_IFC_phonon(phonon,ngqpt,str_ten,UC=None,BEC_in=None,eps_inf_in=None,mod_cell=False,ddb_path='/tmp/'):
    ''' converting phonon object to DDB using the Born Effective charges as in har_xml the phonons are with no NAC'''

    if BEC_in is not None:
        BEC = BEC_in
        NAC={'born':BEC}
    if eps_inf_in is not None:
        eps_inf = eps_inf_in
        NAC={'dielectric':eps_inf}
    else:
        NAC=None
    

    xml_atms=UC

    
    if mod_cell:
        my_cell = xml_atms.get_cell()+np.dot(str_ten,xml_atms.get_cell())
    else:
        my_cell = xml_atms.get_cell()

    ase_UC = Atoms(numbers=xml_atms.numbers,
           cell = my_cell,
           scaled_positions = xml_atms.get_scaled_positions())


    write('DDB_STR.cif',ase_UC,format='cif')
    UC_abistruct = abipy.abio.factories.Structure.from_ase_atoms(ase_UC)

    new_ddb = cnvtr.phonopy_to_abinit(UC_abistruct, phonon.get_supercell_matrix(), ddb_path , ngqpt=ngqpt,
                      force_constants=phonon.force_constants, born=NAC)

    IFC = new_ddb.anaget_ifc()


def str_mult(a,b):
    my_list = [*a.split(),*b.split()]
    my_list.sort()
    return(' '.join(my_list))

def terms_mult(T_1,T_2):
    T1T2 = {}
    for i in T_1.keys():
        for j in T_2.keys():
            my_key = str_mult(i,j)
            if my_key in T1T2.keys():
                T1T2[my_key] = T1T2[my_key]+ T_1[i]*T_2[j]
            else:
                T1T2[my_key] = T_1[i]*T_2[j]
    return(T1T2)

def get_pwr_N(T1,n):
    if n-1!=0:
        return(terms_mult(get_pwr_N(T1,n-1),T1))
    else:
        return(T1)

def re_order_terms(T1):
    fin_dict = {}
    for key in T1.keys():
        my_pwr_list = {}
        tmp_key = 0
        char_0 = ' '
        pwr_1 = 0
        for char in key.split():
            if char_0 == ' ':
                char_0 = char
                my_pwr_list[char_0] = 1
            elif char_0 == char:
                my_pwr_list[char_0] += 1
            else:
                char_0 = char
                my_pwr_list[char_0] = 1
        New_key = [tmp_key+'^'+str(my_pwr_list[tmp_key]) for tmp_key in my_pwr_list.keys()]
        New_key = ' '.join(New_key)
        fin_dict[New_key] = T1[key]
    return(fin_dict)

def get_terms_diff(trms1,trms2):
    diff_list = []
    for i1 in range(len(trms1)-1):
        if trms1[i1] not in trms2:
            diff_list.append(trms1[i1])
    for i2 in range(len(trms2)-1):
        if trms2[i2] not in trms1:
            diff_list.append(trms2[i2])
    return(diff_list)

def my_write_vasp(filename, atoms, label='', direct=False, sort=True, symbol_count = None, long_format=True, vasp5=True):
    """
    Hexu: alter the sort method from 'quicksort' to 'mergesort' so that the order of same symbol is kept; sort default -> True, vasp default ->True.

    Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordiantes is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """

    import numpy as np
    from ase.constraints import FixAtoms, FixScaled

    if isinstance(filename, str):
        f = open(filename, 'w')
    else: # Assume it's a 'file-like object'
        f = filename

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError("Don't know how to save more than "+
                               "one image to VASP input")
        else:
            atoms = atoms[0]

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions()
    else:
        coord = atoms.get_positions()

    if atoms.constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]

    if sort:
        ind = np.argsort(atoms.get_chemical_symbols(),kind='mergesort')
        symbols = np.array(atoms.get_chemical_symbols())[ind]
        coord = coord[ind]
        if atoms.constraints:
            sflags = sflags[ind]
    else:
        symbols = atoms.get_chemical_symbols()

    # Create a list sc of (symbol, count) pairs
    if symbol_count:
        sc = symbol_count
    else:
        sc = []
        psym = symbols[0]
        count = 0
        for sym in symbols:
            if sym != psym:
                sc.append((psym, count))
                psym = sym
                count = 1
            else:
                count += 1
        sc.append((psym, count))

    # Create the label
    if label == '':
        for sym, c in sc:
            label += '%2s ' % sym
    f.write(label + '\n')

    # Write unitcell in real coordinates and adapt to VASP convention
    # for unit cell
    # ase Atoms doesn't store the lattice constant separately, so always
    # write 1.0.
    f.write('%19.16f\n' % 1.0)
    if long_format:
        latt_form = ' %21.16f'
    else:
        latt_form = ' %11.6f'
    for vec in atoms.get_cell():
        f.write(' ')
        for el in vec:
            f.write(latt_form % el)
        f.write('\n')

    # If we're writing a VASP 5.x format POSCAR file, write out the
    # atomic symbols
    if vasp5:
        for sym, c in sc:
            f.write(' %3s' % sym)
        f.write('\n')

    # Numbers of each atom
    for sym, count in sc:
        f.write(' %3i' % count)
    f.write('\n')

    if atoms.constraints:
        f.write('Selective dynamics\n')

    if direct:
        f.write('Direct\n')
    else:
        f.write('Cartesian\n')

    if long_format:
        cform = ' %19.16f'
    else:
        cform = ' %9.6f'
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            f.write(cform % dcoord)
        if atoms.constraints:
            for flag in sflags[iatom]:
                if flag:
                    s = 'F'
                else:
                    s = 'T'
                f.write('%4s' % s)
        f.write('\n')

    if type(filename) == str:
        f.close()

def to_same_cell(pos, ref_pos):
    """
    make every ion  position in pos in the same cell as ref_pos
    pos, ref_pos: array or list of positions.
    """
    pos = np.array(pos)
    for i, position in enumerate(pos):
        for j, xj in enumerate(position):
            if xj - ref_pos[i][j] > 0.5:
                pos[i][j] -= 1.0
            elif xj - ref_pos[i][j] < -0.5:
                pos[i][j] += 1.0
    return pos

def IFsimilar_terms(T1,T2):
    if T1[-1]['dips']==T2[-1]['dips'] and T1[-1]['strain']==T2[-1]['strain']:
        found = [False for i in range(int(T1[-1]['dips']+ T1[-1]['strain']))]
        found = np.array(found)
        for disp_1 in range(int(T1[-1]['dips'])):
            for disp_2 in range(int(T2[-1]['dips'])):
                if float(T1[disp_1]['weight']) == float(T2[disp_2]['weight']):
                    #print('weight_equal')
                    cell_b1 = [int(i) for i in T1[disp_1]['cell_b'].split()]
                    cell_b2 = [int(i) for i in T2[disp_2]['cell_b'].split()]
                    cell_b2N = [-int(i) for i in T2[disp_2]['cell_b'].split()]
                    cell_b1_flg = [True if l!=0 else False for l in cell_b1]
                    cell_b2_flg = [True if l!=0 else False for l in cell_b2]
                    if cell_b1==cell_b2:
                        #print('cells equal')
                        if T1[disp_1]['atom_a'] == T2[disp_2]['atom_a'] and T1[disp_1]['atom_b'] == T2[disp_2]['atom_b'] and T1[disp_1]['direction'] == T2[disp_2]['direction']:
                            #print('atoms equal')
                            if  T1[disp_1]['power'] == T2[disp_2]['power']:
                                found[disp_1]=True
                    elif cell_b1==cell_b2N:
                        #print('oposite cells')
                        if T1[disp_1]['atom_a'] == T2[disp_2]['atom_b'] and T1[disp_1]['atom_b'] == T2[disp_2]['atom_a'] and T1[disp_1]['direction'] == T2[disp_2]['direction']:
                            #print('atoms equal')
                            if  T1[disp_1]['power'] == T2[disp_2]['power']:
                                found[disp_1]=True

                elif float(T1[disp_1]['weight']) == -float(T2[disp_2]['weight']) :
                    #print('weight_oposite')
                    cell_b1 = [int(i) for i in T1[disp_1]['cell_b'].split()]
                    cell_b2 = [int(i) for i in T2[disp_2]['cell_b'].split()]
                    cell_b1_flg = [True if l!=0 else False for l in cell_b1]
                    cell_b2_flg = [True if l!=0 else False for l in cell_b2]
                    if any(cell_b1_flg) or any(cell_b2_flg):
                        if T1[disp_1]['atom_a'] == T2[disp_2]['atom_b'] and T1[disp_1]['atom_b'] == T2[disp_2]['atom_a'] and T1[disp_1]['power'] == T2[disp_2]['power']  and T1[disp_1]['direction'] == T2[disp_2]['direction']:
                            #print('atoms_checked')
                            cell_b1 = [int(i) for i in T1[disp_1]['cell_b'].split()]
                            cell_b2 = [-int(i) for i in T2[disp_2]['cell_b'].split()]
                            if cell_b1==cell_b2:
                                #print('oposite cells')
                                found[disp_1]=True
                    else:
                        if T1[disp_1]['atom_a'] == T2[disp_2]['atom_a'] and T1[disp_1]['atom_b'] == T2[disp_2]['atom_b'] and T1[disp_1]['power'] == T2[disp_2]['power']  and T1[disp_1]['direction'] == T2[disp_2]['direction']:
                            #print('atoms_checked')
                            if cell_b1==cell_b2:
                                #print('same cells')
                                found[disp_1]=True

        for disp_1 in range(int(T1[-1]['dips']),int(T1[-1]['strain'])):
             for disp_2 in range(int(T2[-1]['dips']),int(T2[-1]['strain'])):
                 if T1[disp_1]==T2[disp_2]:
                       found[disp_1]=True

        if all(found):
            return(True)
        else:
            return(False)
    else:
        return(False)

def IFsimilar_Coeff(c1,c2):
    sim_trms = []
    if len(c1)==len(c2):
        found= [False for i in range(len(c1))]
        for i in range(len(c1)):
            for j in range(len(c2)):
                if IFsimilar_terms(c1[i],c2[j]):
                    found[i]=True
                    sim_trms.append((i,j))
        if any(found):
            return(True,np.array(sim_trms))
        else:
            return(False,np.array(sim_trms))
    else:
        return(False,np.array(sim_trms))

def get_SATs(hxml, my_term):
    hxml.get_ase_atoms()
    hxml.get_str_cp()
    my_strain = hxml.strain
    my_atms = hxml.ase_atoms
    sds = spg.get_symmetry_dataset(my_atms)
    print(f"space group number = {sds['number']}  INT = {sds['international']} ")
    rot = sds['rotations']
    trans = sds['translations']
    vgt_mat=np.zeros((7,3,3))
    vgt_mat[1,:,:] = np.reshape([float(v) for v in my_strain[0].split()],(3,3))
    vgt_mat[2,:,:] = np.reshape([float(v) for v in my_strain[1].split()],(3,3))
    vgt_mat[3,:,:] = np.reshape([float(v) for v in my_strain[2].split()],(3,3))
    vgt_mat[4,:,:] = np.reshape([float(v) for v in my_strain[3].split()],(3,3))
    vgt_mat[5,:,:] = np.reshape([float(v) for v in my_strain[4].split()],(3,3))
    vgt_mat[6,:,:] = np.reshape([float(v) for v in my_strain[5].split()],(3,3))
    dir_dic = {'x':1,'y':2,'z':3}
    pos_prm = my_atms.get_positions()
    wrapPos = ase.geometry.wrap_positions
    cell_prm = my_atms.get_cell()
    my_trms=[]
    for my_rot in range(len(rot)):
        disp_lst=[]
        dis_tmp_weight = 1
        wrong_acell = False
        this_term = False
        for disp in range(my_term[-1]['dips']):
            atm_a = int(my_term[disp]['atom_a'])
            cell_a = [ int(cll) for cll in (my_term[disp]['cell_a'].split())]
            atm_b = int(my_term[disp]['atom_b'])
            cell_b = [ int(cll) for cll in (my_term[disp]['cell_b'].split())]
            power = my_term[disp]['power']
            vec_dir = dir_dic[my_term[disp]['direction']]
            rot_vec_dir =vgt_mat[vec_dir,:,:]
            rot_vec_dir = np.dot(np.dot(rot[my_rot],rot_vec_dir),(alg.inv(rot[my_rot])))

            for vgt_cnt in range(1,len(vgt_mat)):
                diff = np.matrix.flatten(abs(rot_vec_dir) - vgt_mat[vgt_cnt,:,:])
                if not any(diff):
                    rot_voit_key = vgt_cnt

            pos_a = pos_prm[atm_a] + np.dot(cell_a,cell_prm)
            pos_b = pos_prm[atm_b] + np.dot(cell_b,cell_prm)

            rot_pos_a = np.dot(rot[my_rot],pos_a)+np.dot(trans[my_rot],cell_prm)
            rot_pos_b = np.dot(rot[my_rot],pos_b)+np.dot(trans[my_rot],cell_prm)

            wrp_a = wrapPos([rot_pos_a], cell_prm)[0]
            wrp_b = wrapPos([rot_pos_b], cell_prm)[0]

            ncell_a0 = rot_pos_a - wrp_a
            ncell_b =  rot_pos_b - wrp_b  - ncell_a0
            ncell_a = [0,0,0] #ncell_a

            atm_a_ind = find_index(pos_prm, wrp_a ,tol=0.0001)
            atm_b_ind = find_index(pos_prm, wrp_b ,tol=0.0001)

            dst0 = [pos_a[0]-pos_b[0],pos_a[1]-pos_b[1],pos_a[2]-pos_b[2]]
            dst = [rot_pos_a[0]-rot_pos_b[0],rot_pos_a[1]-rot_pos_b[1],rot_pos_a[2]-rot_pos_b[2]]

            if abs(np.linalg.norm(dst0)-np.linalg.norm(dst)) > 0.00001:
                wrong_acell = True
                break

            dis_tmp_weight *= (rot[my_rot][rot_voit_key-1,vec_dir-1])**int(power)

            ndis = {
                'atom_a':atm_a_ind,
                'cell_a': f"{round(ncell_a[0]/cell_prm[0,0])} {round(ncell_a[1]/cell_prm[1,1])} {round(ncell_a[2]/cell_prm[2,2])}",
                'atom_b':atm_b_ind,
                'cell_b':f"{round(ncell_b[0]/cell_prm[0,0])} {round(ncell_b[1]/cell_prm[1,1])} {round(ncell_b[2]/cell_prm[2,2])}",
                'direction':get_key(dir_dic,rot_voit_key),
                'power':power
                }
            disp_lst.append(ndis)
        tem_vogt_wgt = 1
        if not(wrong_acell):
            if my_term[-1]['dips'] != 0 and my_term[-1]['strain'] != 0 and len(disp_lst) > 0:
                for strain in range(my_term[-1]['strain']):
                    voigt =int(my_term[disp+strain+1]['voigt'])
                    pwr_str = int(my_term[disp+strain+1]['power'])
                    vgt_trans = np.dot(np.dot(rot[my_rot],vgt_mat[voigt,:,:]),(alg.inv(rot[my_rot])))
                    for vgt_cnt in range(1,len(vgt_mat)):
                        diff = np.matrix.flatten(-(vgt_trans) - vgt_mat[vgt_cnt,:,:])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = -1
                        diff = np.matrix.flatten((vgt_trans) - vgt_mat[vgt_cnt,:,:])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = 1
                    tem_vogt_wgt *= vogt_wgt**int(pwr_str)
                    disp_lst.append({'power':f' {pwr_str}' , 'voigt':f' {rot_voit_key}'})

            for i in range(my_term[-1]['dips']):
                disp_lst[i]['weight'] = " %2.6f" %(dis_tmp_weight*tem_vogt_wgt)
            #if  my_term[-1]['dips']==1 and my_term[-1]['strain']==0:
                #disp_lst[i]['weight'] = " %2.6f" %1
            if  len(disp_lst)>0:
                disp_lst.append(my_term[-1])
                found = False
                prm_lst =list(permutations(disp_lst[:]))
                for prm_mem in prm_lst:
                    my_temp = [prm for prm in prm_mem]
                    if my_temp in  my_trms :
                        found = True
                if disp_lst not in my_trms and not found:
                    if this_term:
                        print('this term printed')
                    my_trms.append(disp_lst)

    for my_rot in range(len(rot)):
        tem_vogt_wgt = 1
        if my_term[-1]['dips'] == 0 and my_term[-1]['strain'] != 0:
            disp_lst = []
            disp_lst.append(my_term[0])
            for strain in range(my_term[-1]['strain']):
                voigt = int(my_term[strain+1]['voigt'])
                pwr_str = int(my_term[strain+1]['power'])
                vgt_trans = np.dot(np.dot(rot[my_rot],vgt_mat[voigt,:,:]),(alg.inv(rot[my_rot])))
                for vgt_cnt in range(1,len(vgt_mat)):
                    diff = np.matrix.flatten(-(vgt_trans) - vgt_mat[vgt_cnt,:,:])
                    if not any(diff):
                        rot_voit_key = vgt_cnt
                        vogt_wgt = -1
                    diff = np.matrix.flatten((vgt_trans) - vgt_mat[vgt_cnt,:,:])
                    if not any(diff):
                        rot_voit_key = vgt_cnt
                        vogt_wgt = 1
                disp_lst.append({'power':f' {pwr_str}' ,'voigt':f' {rot_voit_key}'})
            tem_vogt_wgt *= vogt_wgt**int(pwr_str)
            disp_lst[0]['weight']=" %2.6f" %tem_vogt_wgt

            disp_lst.append(my_term[-1])
            prm_lst =list(permutations(disp_lst[:]))
            found = False
            for prm_mem in prm_lst:
                my_temp = [prm for prm in prm_mem]
                if my_temp in  my_trms :
                    found = True

            if disp_lst not in my_trms and len(disp_lst)>0 and not found:
                my_trms.append(disp_lst)

    return(my_trms)

def wrt_anxml(coeff,trms,fout):
    output=open(fout,'w')
    output.write('<?xml version="1.0" ?>\n')
    output.write('<Heff_definition>\n')
    for i in range(len(coeff)):
        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coeff[i]['number'],coeff[i]['value'],coeff[i]['text'],))
        for j in range(len(trms[i])):
            for k in range(int(trms[i][j][-1]['dips'])):
                if (k==0):
                    output.write('    <term weight="{}">\n'.format(trms[i][j][k]['weight']))
                output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(trms[i][j][k]['atom_a'],trms[i][j][k]['atom_b'],trms[i][j][k]['direction'],trms[i][j][k]['power']))
                output.write('        <cell_a>{}</cell_a>\n'.format(trms[i][j][k]['cell_a']))
                output.write('        <cell_b>{}</cell_b>\n'.format(trms[i][j][k]['cell_b']))
                output.write('      </displacement_diff>\n')
            if int(trms[i][j][-1]['dips']) ==0 :
                k=int(trms[i][j][-1]['strain'])
                output.write('    <term weight="{}">\n'.format(trms[i][j][k]['weight']))
                k=-1
            for l in range(int(trms[i][j][-1]['strain'])):
                output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms[i][j][k+l+1]['power'],trms[i][j][k+l+1]['voigt']))
            output.write('    </term>\n')
        output.write('  </coefficient>\n')
    output.write('</Heff_definition>\n')

def find_clesest_atom(Super_cell,pos,atm_sym=None):
    min_dist = 100
    Positions = Super_cell.get_positions()
    symbls = Super_cell.get_chemical_symbols()
    for i,position in enumerate(Positions):
        dd = position-pos
        dist = np.dot(dd,dd)**0.5
        if dist <= min_dist:
            if atm_sym is None:
                min_dist = dist
                atm_min = i
            elif symbls[i] in atm_sym:             
                min_dist = dist
                atm_min = i      
    return(atm_min)

def anh_terms_mani(har_xml, anh_xml, output='test_mani.xml', terms_to_write=None):
    xmlsys = xml_io.Xml_sys_reader(har_xml)
    xmlsys.get_ase_atoms()
    atoms = xmlsys.ase_atoms
    anhXml = SCXML.Anh_sc_maker(har_xml, anh_xml)
    SC_mat = np.eye(3, dtype=int)
    anhXml.SC_trms(atoms, SC_mat)
    new_terms = []
    new_coeffs = {}
    for i, ii in enumerate(terms_to_write):
        new_terms.append(anhXml.SC_terms[ii])
        new_coeffs[i] = anhXml.SC_coeff[ii]
    anhXml.SC_terms = new_terms
    anhXml.SC_coeff = new_coeffs
    anhXml.wrt_anxml(output)
