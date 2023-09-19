import netCDF4 as nc
import numpy as np
from ase.units import Ha, Bohr
from ase import Atoms
from ase.build import make_supercell
from ase.neighborlist import NeighborList
import xml.etree.ElementTree as ET
from ase.data import atomic_masses
from mayavi import mlab
from ase.io import write
import mync
import matplotlib.pyplot as plt

# def get_NC_str(NC_HIST,stp=0):
#     dso=nc.Dataset(NC_HIST)
#     RSET=dso.variables['rprimd'][:]
#     xcart=dso.variables['xcart'][:]
#     typ0=dso.variables['typat'][:]
#     numbers0=dso.variables['znucl'][:]
#     numbers=[numbers0[:][int(tt)-1] for tt in typ0[:]]
#     sum_str=np.zeros((len(xcart[0]),3))
#     sum_Rset=np.zeros((3,3))
#     # print(len(xcart))
#     My_strct = Atoms(numbers=numbers,positions=xcart[stp]*Bohr, cell=RSET[stp]*Bohr, pbc=True)
#     My_strct.wrap(eps=0.008)
#     return(My_strct)

def get_mapped_strcs(str_to_be_map,str_to_map_to,Ret_index=False):
    natom = len(str_to_map_to.get_scaled_positions())
    natom2 = len(str_to_be_map.get_scaled_positions())
    if natom!=natom2:
        print('wrong structures')
        return(0)
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

def map_strctures(str_1,str_2,tol=0.5):  # the ordering is correct
    red_1 = str_1.get_scaled_positions()
    red_2 = str_2.get_scaled_positions()
    tmp_red1 = np.zeros((len(red_1),3))
    for i in range(len(red_1)):
        for j in range(3):
            diff = red_1[i,j]-red_2[i,j]
            if abs(diff) > tol:
                if diff>0:
                   tmp_red1[i,j] = red_1[i,j]-1
                elif diff<0:
                   tmp_red1[i,j] = 1+red_1[i,j]
            else:
               tmp_red1[i,j] = red_1[i,j]
    Nstr_1 = Atoms(numbers=str_1.get_atomic_numbers(), scaled_positions=tmp_red1, cell=str_1.get_cell())
    return(Nstr_1)

def get_atom_num(atomic_mass, tol=0.1):
    if abs(atomic_mass-208) < 1:
        tol = 0.001
    for i in range(len(atomic_masses)):
        if abs(atomic_masses[i]-atomic_mass) < tol:
            mynum = i
    return(mynum)

class xml_sys():
    def __init__(self, xml_file):
        # self.xml_file=xml_file
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()

    # atomic mass and Positions        >>>>>>>>             self.atm_pos          self.natm
    def get_atoms(self):
        self.natm = 0
        self.atm_pos = []
        for Inf in self.root.iter('atom'):
            mass = Inf.attrib['mass']
            pos = (Inf.find('position').text)
            chrgs = (Inf.find('borncharge').text)
            self.atm_pos.append([mass, pos, chrgs])
            self.natm += 1
        self.atm_pos = np.array(self.atm_pos)

    def get_BEC(self):
        self.get_atoms()
        atm_pos = self.atm_pos
        self.BEC = {}
        for i in range(len(self.atm_pos)):
            brn_tmp = [float(j) for j in atm_pos[i][2].split()[:]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            self.BEC[i] = brn_tmp

    # number of cells in the super cell and local forces    >>>>  self.ncll

    def get_Per_clls(self):
        self.ncll = [0, 0, 0]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            if cll.split()[1] == '0' and cll.split()[2] == '0':
                self.ncll[0] += 1
            if cll.split()[0] == '0' and cll.split()[2] == '0':
                # print(cll)
                self.ncll[1] += 1
            if cll.split()[0] == '0' and cll.split()[1] == '0':
                self.ncll[2] += 1

    # number of cells in the super cell and local forces    >>>>  self.ncll
    def get_tot_Per_clls(self):
        self.tot_ncll = [0, 0, 0]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            if cll.split()[1] == '0' and cll.split()[2] == '0':
                self.tot_ncll[0] += 1
            if cll.split()[0] == '0' and cll.split()[2] == '0':
                # print(cll)
                self.tot_ncll[1] += 1
            if cll.split()[0] == '0' and cll.split()[1] == '0':
                self.tot_ncll[2] += 1

    # getting total forces    add a condition if exists   >>  self.tot_fc
    def get_tot_forces(self):
        self.has_tot_FC = 0
        self.tot_fc = {}
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.tot_fc[cll_txt] = np.array([float(i) for i in data])
        if len(self.tot_fc) > 2:
            self.has_tot_FC = 1

    # getting local forces        add a condition if exists  >>>> self.loc_fc
    def get_loc_forces(self):
        self.loc_fc = {}
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.loc_fc[cll_txt] = np.array([float(i) for i in data])

    # refrence energy        >>>>     self.ref_eng
    def get_ref_energy(self):
        self.ref_eng = 0
        for Inf in self.root.iter('energy'):
            self.ref_eng = float(Inf.text)

    # Unit cell        >>>>    self.prim_vects
    def get_prim_vects(self):
        self.prim_vects = np.zeros((3, 3))
        for Inf in self.root.iter('unit_cell'):
            # print(Inf.attrib['units'])
            for i in range(3):
                self.prim_vects[i, :] = float((Inf.text).split(
                )[3*i+0]), float((Inf.text).split()[3*i+1]), float((Inf.text).split()[3*i+2])
        self.prim_vects = np.array(self.prim_vects)

    # epsilon infinity        >>>>     self.eps_inf
    def get_eps_inf(self):
        self.eps_inf = np.zeros((3, 3))
        for Inf in self.root.iter('epsilon_inf'):
            # print(Inf.attrib['units'])
            for i in range(3):
                self.eps_inf[i, :] = (Inf.text).split()[
                    3*i+0], (Inf.text).split()[3*i+1], (Inf.text).split()[3*i+2]
        self.eps_inf = np.array(self.eps_inf)

    def comp_tot_and_loc_FC(self):
        self.get_loc_forces()
        self.get_tot_forces()
        if self.has_tot_FC:
            if self.loc_sc.keys() == tot_fc.keys():
                self.similar_cells = True
            else:
                self.similar_cells = False

    # elastic constants        >>>>   self.ela_cons
    def get_ela_cons(self):
        ndim = 6
        self.ela_cons = np.zeros((6, 6))
        for Inf in self.root.iter('elastic'):
            # print(Inf.attrib['units'])
            for i in range(6):
                self.ela_cons[i, :] = (Inf.text).split()[6*i+0], (Inf.text).split()[6*i+1], (Inf.text).split()[
                    6*i+2], (Inf.text).split()[6*i+3], (Inf.text).split()[6*i+4], (Inf.text).split()[6*i+5]
        self.ela_cons = np.array(self.ela_cons)

    # reading phonons data        >>>>     self.dymat     self.freq     self.qpoint
    def get_phonos(self):
        self.has_phonons = 0
        self.freq = []
        self.dymat = []
        for Inf in self.root.iter('phonon'):
            data = (Inf.find('qpoint').text)
            self.qpoint = (data)
            data = (Inf.find('frequencies').text).split()
            self.freq.append([float(i) for i in data])
            data = (Inf.find('dynamical_matrix').text).split()
            self.dymat.append([float(i) for i in data])
        self.freq = np.reshape(self.freq, (self.natm, 3))
        self.dymat = np.reshape(self.dymat, (3*self.natm, 3*self.natm))
        if len(self.freq) > 2:
            self.has_phonons = 1

    # reading strain phonon data        >>>>      self.corr_forc       self.strain
    def get_str_cp(self):
        self.corr_forc = {}
        self.strain = {}
        for Inf in self.root.iter('strain_coupling'):
            voigt = float(Inf.attrib['voigt'])
            data = (Inf.find('strain').text)
            # np.reshape(([float(i) for i in data]),(1,9))
            self.strain[voigt] = data
            data = (Inf.find('correction_force').text).split()
            self.corr_forc[voigt] = np.reshape(
                ([float(i) for i in data]), (self.natm, 3))

    def set_tags(self):
        self.get_ase_atoms()
        my_tags = []
        cntr_list = {}
        my_atoms = self.ase_atoms
        my_symbols = my_atoms.get_chemical_symbols()
        counts = [my_symbols.count(i) for i in my_symbols]
        for i in range(len(counts)):
            my_cntr = f'{my_symbols[i]}_cntr'
            if f'{my_symbols[i]}_cntr' in cntr_list.keys():
                cntr_list[f'{my_symbols[i]}_cntr'] += 1
            else:
                cntr_list[f'{my_symbols[i]}_cntr'] = 1
            if cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i] > 1:
                my_char = 1
            elif cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i] == 1:
                my_char = ''
            else:
                my_char = cntr_list[f'{my_symbols[i]}_cntr']
            my_tags.append(f'{my_symbols[i]}{my_char}')
        self.tags = my_tags

    def get_ase_atoms(self):
        self.get_prim_vects()
        self.get_atoms()
        atm_pos = self.atm_pos
        natm = self.natm
        car_pos = []
        amu = []
        for i in range(natm):
            car_pos.append([float(xx)*Bohr for xx in atm_pos[i][1].split()])
            amu.append(float(atm_pos[i][0]))
        atom_num = [get_atom_num(x) for x in amu]
        car_pos = np.array(car_pos)
        self.ase_atoms = Atoms(numbers=atom_num, positions=car_pos,
                               cell=Bohr*np.transpose(self.prim_vects), pbc=True)
        self.ase_atoms.set_masses((amu))

    def get_loc_FC_dic(self):
        self.loc_FC_dic = {}
        self.get_ase_atoms()
        natm = self.ase_atoms.get_global_number_of_atoms()
        self.get_loc_forces()
        for key in self.loc_fc.keys():
            my_cell = [int(x) for x in key.split()]
            my_fc = np.reshape(self.loc_fc[key], (3*natm, 3*natm))
            for atm_a in range(natm):
                for atm_b in range(natm):
                    FC_mat = np.zeros((3, 3))
                    int_key = '{}_{}_{}_{}_{}'.format(
                        atm_a, atm_b, my_cell[0], my_cell[1], my_cell[2])
                    FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                    self.loc_FC_dic[int_key] = FC_mat

    def get_tot_FC_dic(self):
        self.get_tot_forces()
        self.tot_FC_dic = {}
        if self.has_tot_FC:
            self.get_ase_atoms()
            natm = self.ase_atoms.get_global_number_of_atoms()
            for key in self.tot_fc.keys():
                my_cell = [int(x) for x in key.split()]
                my_fc = np.reshape(self.tot_fc[key], (3*natm, 3*natm))
                for atm_a in range(natm):
                    for atm_b in range(natm):
                        FC_mat = np.zeros((3, 3))
                        int_key = '{}_{}_{}_{}_{}'.format(
                            atm_a, atm_b, my_cell[0], my_cell[1], my_cell[2])
                        FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                        self.tot_FC_dic[int_key] = FC_mat

    # number of cells in the super cell and local forces    >>>>  self.loc_cells
    def get_loc_cells(self):
        self.loc_cells = [[0, 0], [0, 0], [0, 0]]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell = [int(x) for x in cll.split()]
            if my_cell[0] <= self.loc_cells[0][0]:
                self.loc_cells[0][0] = my_cell[0]
            if my_cell[0] >= self.loc_cells[0][1]:
                self.loc_cells[0][1] = my_cell[0]
            if my_cell[1] <= self.loc_cells[1][0]:
                self.loc_cells[1][0] = my_cell[1]
            if my_cell[1] >= self.loc_cells[1][1]:
                self.loc_cells[1][1] = my_cell[1]
            if my_cell[2] <= self.loc_cells[2][0]:
                self.loc_cells[2][0] = my_cell[2]
            if my_cell[2] >= self.loc_cells[2][1]:
                self.loc_cells[2][1] = my_cell[2]

    # number of cells in the super cell and local forces    >>>>  self.loc_cells

    def get_tot_cells(self):
        self.tot_cells = [[0, 0], [0, 0], [0, 0]]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell = [int(x) for x in cll.split()]
            if my_cell[0] <= self.tot_cells[0][0]:
                self.tot_cells[0][0] = my_cell[0]
            if my_cell[0] >= self.tot_cells[0][1]:
                self.tot_cells[0][1] = my_cell[0]
            if my_cell[1] <= self.tot_cells[1][0]:
                self.tot_cells[1][0] = my_cell[1]
            if my_cell[1] >= self.tot_cells[1][1]:
                self.tot_cells[1][1] = my_cell[1]
            if my_cell[2] <= self.tot_cells[2][0]:
                self.tot_cells[2][0] = my_cell[2]
            if my_cell[2] >= self.tot_cells[2][1]:
                self.tot_cells[2][1] = my_cell[2]

class Get_Pol():
    def __init__(self,Str_ref,BEC_ref,proj_dir = [1,1,1],cntr_at = ['Ti'],trans_mat = [1,1,1],dim_1=1,fast=False):
        self.proj_dir=proj_dir
        self.cntr_at=cntr_at
        self.dim_1=dim_1
        self.BEC_ref=BEC_ref 
        self.Str_ref= Str_ref   
        self.trans_mat = trans_mat
        self.get_NL()
        self.fast = fast

    def get_NL(self):
        self.chmsym = self.Str_ref.get_chemical_symbols()
        # write('POSCAR.cif',self.Str_ref,format='cif')
        self.cntr_indxs = []
        for atm_indx,atm_sym in enumerate(self.chmsym):
            if atm_sym in self.cntr_at:
                self.cntr_indxs.append(atm_indx)
        cutof_dic = {}
        ABC_SL = self.Str_ref.cell.cellpar()[0:3]
        ABC_SL = [ABC_SL[0]/self.trans_mat[0],ABC_SL[1]/self.trans_mat[1],ABC_SL[2]/self.trans_mat[2]] 
        if 'Ti' in self.cntr_at:
            self.wght = {'O':1/2,'Ba':1/8,'Sr':1/8,'Pb':1/8,'Ti':1,'Ca':1/8} 
            self.ref_atm = 1
            cutof_dic['Ti'] = ABC_SL[0]/2 - 0.5
            cutof_dic['Ba'] = ABC_SL[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Sr'] = ABC_SL[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Pb'] = ABC_SL[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['O'] = ABC_SL[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
            cutof_dic['Ca'] = ABC_SL[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
        else: # self.cntr_at in ['Pb','Ba','Ca']:
            self.wght = {'O':1/4,'Ba':1,'Sr':1,'Pb':1,'Ti':1/8,'Ca':1} 
            self.ref_atm = 0
            cutof_dic['Ti'] = ABC_SL[0]/2 - 0.5 + 0.1  #
            cutof_dic['Ba'] = ABC_SL[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Sr'] = ABC_SL[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Pb'] = ABC_SL[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['O'] =  ABC_SL[0]/2 - 0.5 + 0.1 #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
            cutof_dic['Ca'] = ABC_SL[0]/2 - 0.5 + 0.1 # self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1

        sccuof = [cutof_dic[sym] for sym in self.chmsym]        
        self.mnl = NeighborList(sccuof,sorted=False,self_interaction = False,bothways=True)
        self.mnl.update(self.Str_ref)

    def get_pol_mat(self,Str_dist):  
        ABC_SL0=self.Str_ref.cell.cellpar()[0:3]
        v0 = np.linalg.det(self.Str_ref.get_cell())/(self.trans_mat[0]*self.trans_mat[1]*self.trans_mat[2])
        ABC_SL = [ABC_SL0[0]/self.trans_mat[0],ABC_SL0[1]/self.trans_mat[1],ABC_SL0[2]/self.trans_mat[2]]   
        ref_positions = self.Str_ref.get_positions()
        
        disp_proj = self.get_disp(Str_dist)
        pol_mat = np.zeros((self.trans_mat[0],self.trans_mat[1],self.trans_mat[2],3))  
        ## TODO count number of unique centers and then for each of them define a wight with respect to number of neighbours
        for aa in self.cntr_indxs:
            NNs,offsets = self.mnl.get_neighbors(aa)
            # print(len(NNs))
            if ref_positions[aa,2] <= ABC_SL[2]*self.dim_1 :
                k=0
            else:
                k=1            
            a,b,c=int(abs(ref_positions[aa,0]-ref_positions[self.ref_atm,0])/(ABC_SL[0]*0.95)),int(abs(ref_positions[aa,1]-ref_positions[self.ref_atm,1])/(ABC_SL[1]*0.95)),int(abs(ref_positions[aa,2]-ref_positions[self.ref_atm,2])/(ABC_SL[2]*0.95))
            NNs = np.append(aa,NNs)            
            # print('Number of Neghbours  =  ',len(NNs))
            syms_NNS = []
            for j in NNs:  
                syms_NNS.append(self.chmsym[j])
                pol_mat[a,b,c,:] += self.wght[self.chmsym[j]]*np.dot(disp_proj[j],self.BEC_ref[k,j%5,:,:])   # Bohr
            
        pol_mat=16*pol_mat/v0
        return(pol_mat)

    def get_disp(self,Str_dist):  
        # Prim_Str_Hist = Atoms(numbers=Str_dist.get_atomic_numbers(),scaled_positions=self.Str_ref.get_scaled_positions(), cell=Str_dist.get_cell(), pbc=True)
        dist_str = Atoms(numbers=Str_dist.get_atomic_numbers(),scaled_positions=Str_dist.get_scaled_positions(), cell=self.Str_ref.get_cell(), pbc=True)
        if self.fast:
            Fnl_str=map_strctures(dist_str,self.Str_ref,tol=0.2)
        else:
            Fnl_str=get_mapped_strcs(dist_str,self.Str_ref,Ret_index=False)
        self.chmsym = Fnl_str.get_chemical_symbols()
        red_disp = Fnl_str.get_scaled_positions()-self.Str_ref.get_scaled_positions()   
        disp = np.dot(red_disp,dist_str.get_cell())               
        disp_proj = self.proj_dir*disp
        return(disp_proj)

def plot_3d_pol_vectrs_pol(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False):
    pol_mat = get_pol_vectrs(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4)
    max_pol = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                scl_pol = (np.dot(pol_mat[i,j,k,:],pol_mat[i,j,k,:]))**0.5
                if scl_pol>max_pol:
                    max_pol = scl_pol
    print(10*'---','Max Polarization length',max_pol)
    if plot_3d:
        mlab.quiver3d(pol_mat[:,:,:,0], pol_mat[:,:,:,1], pol_mat[:,:,:,2])
    else:
        src = mlab.pipeline.vector_field(pol_mat[:,:,:,0], pol_mat[:,:,:,1], pol_mat[:,:,:,2])
        mlab.pipeline.vector_cut_plane(src, mask_points=1, scale_factor=1)
    mlab.show() 
          
def get_pol_vectrs(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4):
    plot_Lpol = True
    myxml1=xml_sys(xml_file)
    myxml1.get_atoms()
    atm_pos1=myxml1.atm_pos
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]]) 

    if xml_file2 is not None:
        myxml2=xml_sys(xml_file2)
        myxml2.get_atoms()
        atm_pos2=myxml2.atm_pos
    else:
        atm_pos2=atm_pos1

    atm_pos=[atm_pos1,atm_pos2]
    BEC=np.zeros((2,5,3,3))
    for bb in range(2):
        for aa in range(my_atms1.get_global_number_of_atoms()):
            brn_tmp=[float(k) for k in atm_pos[bb][aa][2].split()[:]]
            brn_tmp=np.array(brn_tmp)
            brn_tmp=np.reshape(brn_tmp,(3,3))
            BEC[bb,aa,:,:]=brn_tmp

    if ave_str:
        final_Str_Hist = mync.get_avg_str(NC_FILE_STR,init_stp=NC_stp)
    else:
        final_Str_Hist = mync.get_NC_str(NC_FILE_STR,stp=NC_stp)
    Prim_Str_Hist = Atoms(numbers=ref_str.get_atomic_numbers(),scaled_positions=ref_str.get_scaled_positions(), cell=final_Str_Hist.get_cell(), pbc=True)
    my_pol = Get_Pol(Prim_Str_Hist,BEC,proj_dir = plot_dire,cntr_at = cntr_at,trans_mat = dim,dim_1=1,fast=Fast_map)
    # write('M2_POSCAR',final_Str_Hist,format='vasp')
    pol_mat=my_pol.get_pol_mat(final_Str_Hist)
    # print(pol_mat)
    return(pol_mat)

def plot_3d_pol_vectrs_dis(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4):
    myxml1=xml_sys(xml_file)
    myxml1.get_atoms()
    atm_pos1=myxml1.atm_pos
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]]) 

    if xml_file2 is not None:
        myxml2=xml_sys(xml_file2)
        myxml2.get_atoms()
        atm_pos2=myxml2.atm_pos
    else:
        atm_pos2=atm_pos1

    atm_pos=[atm_pos1,atm_pos2]
    BEC=np.zeros((2,5,3,3))
    for bb in range(2):
        for aa in range(my_atms1.get_global_number_of_atoms()):
            brn_tmp=[float(k) for k in atm_pos[bb][aa][2].split()[:]]
            brn_tmp=np.array(brn_tmp)
            brn_tmp=np.reshape(brn_tmp,(3,3))
            BEC[bb,aa,:,:]=brn_tmp

    if ave_str:
        final_Str_Hist = mync.get_avg_str(NC_FILE_STR,init_stp=NC_stp)
    else:
        final_Str_Hist = mync.get_NC_str(NC_FILE_STR,stp=NC_stp)
    Prim_Str_Hist = Atoms(numbers=ref_str.get_atomic_numbers(),scaled_positions=ref_str.get_scaled_positions(), cell=final_Str_Hist.get_cell(), pbc=True)
    my_pol = Get_Pol(Prim_Str_Hist,BEC,proj_dir = plot_dire,cntr_at = cntr_at,trans_mat = dim,dim_1=1,fast=Fast_map)
    
    disps_vectrs = my_pol.get_disp(final_Str_Hist)
    positions_vectrs = final_Str_Hist.get_positions()  
    # length_mul = 4 
    chem_sym = final_Str_Hist.get_chemical_symbols()
    colors_dic = {'O':'r','Ti':'blue','Pb':'k','Sr':'green'}
    max_x,max_y,max_z = max(positions_vectrs[:,0]),max(positions_vectrs[:,1]),max(positions_vectrs[:,2])
    min_x,min_y,min_z = min(positions_vectrs[:,0]),min(positions_vectrs[:,1]),min(positions_vectrs[:,2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    out_pt = open('my_pos_vec.txt','w')
    for ii in range(len(positions_vectrs)):
        out_pt.write(f'{positions_vectrs[ii][0]:.4f}  {positions_vectrs[ii][1]:.4f}  {positions_vectrs[ii][2]:.4f}  {length_mul*disps_vectrs[ii][0]:.4f}  {length_mul*disps_vectrs[ii][1]:.4f}  {length_mul*disps_vectrs[ii][2]:.4f} \n ')
        v = np.array([disps_vectrs[ii][0],disps_vectrs[ii][1],disps_vectrs[ii][2]])
        vlength=length_mul*np.linalg.norm(v)
        ax.quiver(positions_vectrs[ii][0],positions_vectrs[ii][1],positions_vectrs[ii][2],length_mul*disps_vectrs[ii][0],length_mul*disps_vectrs[ii][1],length_mul*disps_vectrs[ii][2],
                pivot='tail',length=vlength,arrow_length_ratio=0.3/vlength,color=colors_dic[chem_sym[ii]])

    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    ax.set_zlim([min_z,max_z])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()  

# def plot_3d_pol_vectrs_dis_00(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4):
#     myxml1=xml_sys(xml_file)
#     myxml1.get_atoms()
#     atm_pos1=myxml1.atm_pos
#     myxml1.get_ase_atoms()
#     my_atms1=myxml1.ase_atoms
#     ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]]) 

#     if xml_file2 is not None:
#         myxml2=xml_sys(xml_file2)
#         myxml2.get_atoms()
#         atm_pos2=myxml2.atm_pos
#     else:
#         atm_pos2=atm_pos1

#     atm_pos=[atm_pos1,atm_pos2]
#     BEC=np.zeros((2,5,3,3))
#     for bb in range(2):
#         for aa in range(my_atms1.get_global_number_of_atoms()):
#             brn_tmp=[float(k) for k in atm_pos[bb][aa][2].split()[:]]
#             brn_tmp=np.array(brn_tmp)
#             brn_tmp=np.reshape(brn_tmp,(3,3))
#             BEC[bb,aa,:,:]=brn_tmp

#     if ave_str:
#         final_Str_Hist = mync.get_avg_str(NC_FILE_STR,init_stp=NC_stp)
#     else:
#         final_Str_Hist = mync.get_NC_str(NC_FILE_STR,stp=NC_stp)
#     Prim_Str_Hist = Atoms(numbers=ref_str.get_atomic_numbers(),scaled_positions=ref_str.get_scaled_positions(), cell=final_Str_Hist.get_cell(), pbc=True)
#     my_pol = Get_Pol(Prim_Str_Hist,BEC,proj_dir = plot_dire,cntr_at = cntr_at,trans_mat = dim,dim_1=1,fast=Fast_map)
    
#     disps_vectrs = my_pol.get_disp(final_Str_Hist)
#     positions_vectrs = final_Str_Hist.get_positions()  
#     plot_3d_vector_field(positions_vectrs,disps_vectrs)

# def plot_3d_vector_field(positions, vectors):
#     # Extract x, y, and z coordinates from the positions
#     x = positions[:, 0]
#     y = positions[:, 1]
#     z = positions[:, 2]

#     # Extract u, v, and w components from the vectors
#     u = vectors[:, 0]
#     v = vectors[:, 1]
#     w = vectors[:, 2]

#     # Compute vector magnitudes
#     magnitudes = np.sqrt(u**2 + v**2 + w**2)

#     # Compute vector directions
#     direction_x = np.sign(u)
#     direction_y = np.sign(v)
#     direction_z = np.sign(w)

#     # Create a figure
#     # fig = mlab.figure()

#     # Plot the vector field
#     magnitude=10
#     for ii in range(len(positions)):
#         mlab.quiver3d(positions[ii][0], positions[ii][1], positions[ii][2], magnitude*vectors[ii][0], magnitude*vectors[ii][1], magnitude*vectors[ii][2], scale_factor=1, mode='arrow')

#     # Set the axis labels
#     mlab.xlabel('X')
#     mlab.ylabel('Y')
#     mlab.zlabel('Z')

#     # Set the plot title
#     mlab.title('3D Vector Field')

#     # Display the plot
#     mlab.show()

