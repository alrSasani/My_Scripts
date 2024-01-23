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
import My_Scripts.mync as mync
from My_Scripts import xml_io
import matplotlib.pyplot as plt
from ase.build import stack

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

def make_SL(a1, a2,ref_cell = 'cell_1',cell_parr_diff = 0):
    cell_1 = a1.get_cell()
    cell_2 = a2.get_cell()

    if ref_cell == 'cell_1':
        cell_SL = [cell_1[0][0], cell_1[1][1], cell_1[2][2]+cell_2[2][2]]
    else:
        cell_SL = [cell_2[0][0], cell_2[1][1], cell_1[2][2]+cell_2[2][2]]

    pos1 = a1.get_positions()
    tags_1 = a1.get_array('tag_id')
    BEC_1 = a1.get_array('BEC')
    str_ph1 = a1.get_array('str_ph')

    pos2 = a2.get_positions()
    tags_2 = a2.get_array('tag_id')
    BEC_2 = a2.get_array('BEC')
    str_ph2 = a2.get_array('str_ph')

    str_ph = []
    SL_tags = []
    SL_BEC = []
    car_SL = []
    for i, cor in enumerate(pos1):
        SL_tags.append(tags_1[i])
        SL_BEC.append(BEC_1[i])
        str_ph.append(str_ph1[i])
        car_SL.append(cor)
    for i, cor in enumerate(pos2):
        car_SL.append([cor[0], cor[1], cor[2]+cell_1[2][2]+cell_parr_diff])
        SL_tags.append(tags_2[i])
        SL_BEC.append(BEC_2[i])
        str_ph.append(str_ph2[i])

    numbers1 = a1.get_atomic_numbers()
    numbers2 = a2.get_atomic_numbers()
    numbers_SL = [*numbers1, *numbers2]
    my_SL = Atoms(numbers=numbers_SL, positions=car_SL, cell=cell_SL, pbc=True)
    my_SL.set_array('tag_id', np.array(SL_tags))
    my_SL.set_array('BEC', np.array(SL_BEC))
    my_SL.set_array('str_ph', np.array(str_ph))
    return(my_SL)

def map_strctures(str_1,str_2,tol=0.05):  # the ordering is correct
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


class xml_sys:
    ''' Harmonic xml reader:
    This class is used to reas harmonic xml file  and return the data needed for construction of the SL'''
    def __init__(self, xml_file, mat_id='', extract_dta=False):
        self.id = mat_id
        tree = ET.parse(xml_file)
        self.root = tree.getroot()
        if extract_dta:
            self.get_atoms()
            self.get_ref_energy()
            self.get_eps_inf()
            self.get_ela_cons()
            self.get_str_cp()
            self.get_ase_atoms()
            self.get_loc_FC_tags()
            self.get_tot_FC_tags()
            self.set_tags()

    # atomic mass and Positions        >>>>>>>>             self.atm_pos          self.natm
    def get_atoms(self):
        self.natm = 0
        self.atm_pos = []
        for Inf in self.root.iter('atom'):
            mass = Inf.attrib['mass']
            # print(Inf.attrib['massunits'])
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

    # getting total forces    add a condition if exists   >>  self.tot_fc
    def get_tot_forces(self):
        self.has_tot_FC = 0
        self.tot_fc = {}
        cells = [[], [], []]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.tot_fc[cll_txt] = np.array([float(i) for i in data])
            cells[0].append(int(cll.split()[0]))
            cells[1].append(int(cll.split()[1]))
            cells[2].append(int(cll.split()[2]))

        if len(self.tot_fc) > 2:
            self.has_tot_FC = True
            self.tot_cells = [[min(cells[0]), max(cells[0])], [min(
                cells[1]), max(cells[1])], [min(cells[2]), max(cells[2])]]
        else:
            self.tot_cells = [[0, 0], [0, 0], [0, 0]]

    # getting local forces        add a condition if exists  >>>> self.loc_fc
    def get_loc_forces(self):
        self.loc_fc = {}
        cells = [[], [], []]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.loc_fc[cll_txt] = np.array([float(i) for i in data])
            cells[0].append(int(cll.split()[0]))
            cells[1].append(int(cll.split()[1]))
            cells[2].append(int(cll.split()[2]))
        self.loc_cells = [[min(cells[0]), max(cells[0])], [min(
            cells[1]), max(cells[1])], [min(cells[2]), max(cells[2])]]

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

    # elastic constants        >>>>   self.ela_cons
    def get_ela_cons(self):
        ndim = 6
        self.ela_cons = np.zeros((ndim, ndim))
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
            voigt = int(Inf.attrib['voigt'])
            data = (Inf.find('strain').text)
            # np.reshape(([float(i) for i in data]),(1,9))
            self.strain[voigt] = data
            data = (Inf.find('correction_force').text).split()
            self.corr_forc[voigt] = np.reshape(
                ([float(i) for i in data]), (self.natm, 3))

    # making ase_atoms object from the structure of the xml file
    def get_ase_atoms(self):
        self.get_str_cp()
        self.get_prim_vects()
        self.get_atoms()
        atm_pos = self.atm_pos
        natm = self.natm
        car_pos = []
        amu = []
        BEC = []
        tag_id = []
        str_ph_cp = []
        for i in range(natm):
            str_ph_cp.append([])
            for j in range(6):
                str_ph_cp[i].append(self.corr_forc[j][i, :])
            amu.append(float(atm_pos[i][0]))
            car_pos.append([float(xx)*Bohr for xx in atm_pos[i][1].split()])
            brn_tmp = [float(j) for j in atm_pos[i][2].split()[:]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            BEC.append(brn_tmp)
            tag_id.append([i, self.id])
            # tag.append(i)
        atom_num = [get_atom_num(x) for x in amu]
        self.ase_atoms = Atoms(numbers=atom_num, positions=np.array(
            car_pos), cell=Bohr*self.prim_vects, pbc=True)
        self.ase_atoms.set_array('BEC', np.array(BEC))
        self.ase_atoms.set_array('tag_id', np.array(tag_id))
        self.ase_atoms.set_array('str_ph', np.array(str_ph_cp))
        #self.ase_atoms.set_array('tag', tag)
        self.ase_atoms.set_masses((amu))

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

    def get_loc_FC_tags(self):
        self.get_ase_atoms()
        # tags = self.ase_atoms.get_array('tags') #self.tags
        tag_id = self.ase_atoms.get_array('tag_id')
        # print(tag_id)
        self.loc_FC_tgs = {}
        natm = self.natm  # self.ase_atoms.get_global_number_of_atoms()
        self.get_loc_forces()
        # lc_fc=self.loc_fc
        for key in self.loc_fc.keys():
            my_cell = [int(x) for x in key.split()]
            my_fc = np.reshape(self.loc_fc[key], (3*natm, 3*natm))
            for atm_a in range(natm):
                for atm_b in range(natm):
                    # print(tag_id[atm_a][0])
                    FC_mat = np.zeros((3, 3))
                    int_key = f'{tag_id[atm_a][1]}{tag_id[atm_a][0]}_{tag_id[atm_b][1]}{tag_id[atm_b][0]}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
                    FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                    self.loc_FC_tgs[int_key] = FC_mat

    # this function return the total force constants in the xml file as dictionary
    def get_tot_FC_tags(self):
        self.get_ase_atoms()
        #tags = self.ase_atoms.get_array('tags')
        tag_id = self.ase_atoms.get_array('tag_id')
        self.get_tot_forces()
        self.tot_FC_tgs = {}
        if self.has_tot_FC:
            natm = self.natm  # self.ase_atoms.get_global_number_of_atoms()
            # tot_fc=self.tot_fc
            for key in self.tot_fc.keys():
                my_cell = [int(x) for x in key.split()]
                my_fc = np.reshape(self.tot_fc[key], (3*natm, 3*natm))
                for atm_a in range(natm):
                    for atm_b in range(natm):
                        FC_mat = np.zeros((3, 3))
                        int_key = f'{tag_id[atm_a][1]}{tag_id[atm_a][0]}_{tag_id[atm_b][1]}{tag_id[atm_b][0]}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
                        FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                        self.tot_FC_tgs[int_key] = FC_mat

###########################################################

class Get_Pol():
    def __init__(self,Str_ref,BEC_ref,proj_dir = [1,1,1],cntr_at = ['Ti'],trans_mat = [1,1,1],dim_1=1,fast=False,cal_c_ov_a=False,origin_atm=['Pb','Sr']):
        self.proj_dir=proj_dir
        self.cntr_at=cntr_at
        self.dim_1=dim_1
        self.BEC_ref=BEC_ref 
        self.Str_ref= Str_ref   
        self.trans_mat = trans_mat
        self.origin_atm = origin_atm
        self.get_NL()
        self.fast = fast
        self.cal_c_ov_a = cal_c_ov_a

    def get_NL(self):
        self.chmsym = self.Str_ref.get_chemical_symbols()
        # write('POSCAR.cif',self.Str_ref,format='cif')
        self.cntr_indxs = []
        for atm_indx,atm_sym in enumerate(self.chmsym):
            if atm_sym in self.cntr_at:
                self.cntr_indxs.append(atm_indx)
        cutof_dic = {}
        ABC_SL = self.Str_ref.cell.cellpar()[0:3]
        ABC_UC = [ABC_SL[0]/self.trans_mat[0],ABC_SL[1]/self.trans_mat[1],ABC_SL[2]/self.trans_mat[2]] 
        if 'Ti' in self.cntr_at:
            self.wght = {'O':1/2,'Ba':1/8,'Sr':1/8,'Pb':1/8,'Ti':1,'Ca':1/8} 
            self.ref_atm = 1
            cutof_dic['Ti'] = ABC_UC[0]/2 - 0.5
            cutof_dic['Ba'] = ABC_UC[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Sr'] = ABC_UC[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Pb'] = ABC_UC[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['O'] = ABC_UC[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
            cutof_dic['Ca'] = ABC_UC[0]/2 - 0.5 + 0.1  #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
        else: # self.cntr_at in ['Pb','Ba','Ca']:
            self.wght = {'O':1/4,'Ba':1,'Sr':1,'Pb':1,'Ti':1/8,'Ca':1} 
            self.ref_atm = 0
            cutof_dic['Ti'] = ABC_UC[0]/2 - 0.5 + 0.1  #
            cutof_dic['Ba'] = ABC_UC[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Sr'] = ABC_UC[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['Pb'] = ABC_UC[0]/2 - 0.5 #self.Str_ref.get_distance(1,0)-cutof_dic['Ti']+0.1
            cutof_dic['O'] =  ABC_UC[0]/2 - 0.5 + 0.1 #self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
            cutof_dic['Ca'] = ABC_UC[0]/2 - 0.5 + 0.1 # self.Str_ref.get_distance(1,2)-cutof_dic['Ti']+0.1
        sccuof = [cutof_dic[sym] for sym in self.chmsym]        
        self.mnl = NeighborList(sccuof,sorted=False,self_interaction = False,bothways=True)
        self.mnl.update(self.Str_ref)

    def get_pol_mat(self,Str_dist):  
        ABC_SL0=self.Str_ref.cell.cellpar()[0:3]
        ABC_UC = [ABC_SL0[0]/self.trans_mat[0],ABC_SL0[1]/self.trans_mat[1],ABC_SL0[2]/self.trans_mat[2]] 
        v0 = np.linalg.det(self.Str_ref.get_cell())/(self.trans_mat[0]*self.trans_mat[1]*self.trans_mat[2])          
        ref_positions = self.Str_ref.get_positions()        
        disp_proj = self.get_disp(Str_dist)
        self.c_ov_a_data = np.zeros((self.trans_mat[0],self.trans_mat[1],self.trans_mat[2],4)) 
        # self.c_ov_a = np.zeros((self.trans_mat[0],self.trans_mat[1],self.trans_mat[2])) 
        pol_mat = np.zeros((self.trans_mat[0],self.trans_mat[1],self.trans_mat[2],3))  
        pos_final_strc = self.Final_strc.get_positions()
        for aa in self.cntr_indxs:
            NNs,offsets = map(list,self.mnl.get_neighbors(aa))
            a,b,c=int(abs(ref_positions[aa,0]-ref_positions[self.ref_atm,0])/(ABC_UC[0]*0.99)),int(abs(ref_positions[aa,1]-ref_positions[self.ref_atm,1])/(ABC_UC[1]*0.99)),int(abs(ref_positions[aa,2]-ref_positions[self.ref_atm,2])/(ABC_UC[2]*0.99))
            NNs.append(aa) 
            offsets.append([0,0,0])
            syms_NNS = []
            pos_c_ov_a = []
            for j,offset in zip(NNs,offsets): 
                syms_NNS.append(self.chmsym[j])
                pol_mat[a,b,c,:] += self.wght[self.chmsym[j]]*np.dot(disp_proj[j],self.BEC_ref[j])   
                if self.cal_c_ov_a and self.chmsym[j] in self.origin_atm:
                    pos_c_ov_a.append(pos_final_strc[j]+(offset*ABC_SL0))
            if self.cal_c_ov_a:
                temp_ca_data = self.get_c_ov_a(pos_c_ov_a,ABC_UC)
                self.c_ov_a_data[a,b,c,:] = [temp_ca_data[0],temp_ca_data[1],temp_ca_data[2],temp_ca_data[2]/(0.5*temp_ca_data[0]+0.5*temp_ca_data[1])]                
                v0 = abs(self.c_ov_a_data[a,b,c,0]*self.c_ov_a_data[a,b,c,1]*self.c_ov_a_data[a,b,c,2])
            pol_mat[a,b,c,:] = 16*pol_mat[a,b,c,:]/v0       
        return(pol_mat)

    def get_c_ov_a(self,pos_c_ov_a,ABC_UC):
        tol_c = 0.5*ABC_UC[2]
        tol_a = 0.5*ABC_UC[0]
        tol_b = 0.5*ABC_UC[1]
        c_tmp,c_cntr = 0,0
        a_tmp,a_cntr = 0,0
        b_tmp,b_cntr = 0,0        
        for i,ipos in enumerate(pos_c_ov_a):
            for j,jpos in enumerate(pos_c_ov_a):
                if i!=j:
                    abs_dist = abs(ipos-jpos)
                    if abs_dist[2]>tol_c and abs_dist[0]<tol_a and abs_dist[1]<tol_b:
                        c_tmp+=abs_dist[2]
                        c_cntr+=1
                    elif abs_dist[0]>tol_a and  abs_dist[1]<tol_b and abs_dist[2]<tol_c :
                        a_tmp+=abs_dist[0]
                        a_cntr+=1
                    elif abs_dist[1]>tol_b and abs_dist[0]<tol_a and abs_dist[2]<tol_c:
                        b_tmp+=abs_dist[1]
                        b_cntr+=1
        c_ov_a_data = [a_tmp/a_cntr,b_tmp/b_cntr,c_tmp/c_cntr]
        return(c_ov_a_data)
                
    def get_disp(self,Str_dist):  
        dist_str = Atoms(numbers=Str_dist.get_atomic_numbers(),scaled_positions=Str_dist.get_scaled_positions(), cell=self.Str_ref.get_cell(), pbc=True)
        if self.fast:
            Fnl_str=map_strctures(dist_str,self.Str_ref,tol=0.2)
        else:
            Fnl_str=get_mapped_strcs(dist_str,self.Str_ref,Ret_index=False)
        self.Final_strc = Fnl_str
        self.chmsym = Fnl_str.get_chemical_symbols()
        red_disp = Fnl_str.get_scaled_positions()-self.Str_ref.get_scaled_positions()   
        disp = np.dot(red_disp,dist_str.get_cell())               
        disp_proj = self.proj_dir*disp
        return(disp_proj)

def plot_3d_pol_vectrs_pol(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,
                           cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False):
    pol_mat = get_pol_vectrs(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4)
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
          
def get_pol_vectrs(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,dim_1=0,Fast_map=True,cntr_at = ['Ti'],
                   plot_dire=[1,1,1],cal_c_ov_a=False,origin_atm=['Pb','Sr'],ave_str=False,length_mul=4):
    myxml1=xml_io.Xml_sys_reader(xml_file)
    myxml1.get_atoms()
    # atm_pos1=myxml1.atm_pos
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    # ref_str = my_atms1.repeat([1,1,dim_1]) # make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim_1]])  #my_atms1.repeat([dim[0],dim[1],dim_1]) #

    if xml_file2 is not None:
        if dim_1==0:
            raise 'dim_1 should be provided for xml 2'
        myxml2=xml_io.Xml_sys_reader(xml_file2)
        myxml2.get_atoms()
        myxml2.get_ase_atoms()
        ref_str_2 = my_atms1.repeat([1,1,dim[2]-dim_1])
        ref_strt = stack(ref_str,ref_str_2, axis = 2)
        ref_str = ref_strt.repeat([dim[0],dim[1],1]) 
    
    else:
        ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]])
    BEC = ref_str.get_array('BEC')
    print('reffff >>>>',len(ref_str))
    if ave_str:
        final_Str_Hist = mync.get_avg_str(NC_FILE_STR,init_stp=NC_stp)
    else:
        final_Str_Hist = mync.get_NC_str(NC_FILE_STR,stp=NC_stp)
    if dim_1!=0:
        dim[2] = dim_1
    Prim_Str_Hist = Atoms(numbers=ref_str.get_atomic_numbers(),scaled_positions=ref_str.get_scaled_positions(), cell=final_Str_Hist.get_cell(), pbc=True)
    print('>>>>',len(final_Str_Hist))
    my_pol = Get_Pol(Prim_Str_Hist,BEC,proj_dir = plot_dire,cntr_at = cntr_at,trans_mat = dim,dim_1=0,fast=Fast_map,cal_c_ov_a=cal_c_ov_a,origin_atm=origin_atm)
    # write('POSCAR_Finall_Strc_Pol',final_Str_Hist,vasp5=True,sort=True)
    pol_mat = my_pol.get_pol_mat(final_Str_Hist)

    if cal_c_ov_a==True:
        return(pol_mat,my_pol.c_ov_a_data)
    else:
        return(pol_mat)

def plot_3d_pol_vectrs_dis(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_3d=False,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False,length_mul=4):
    myxml1=xml_io.Xml_sys_reader(xml_file)
    myxml1.get_atoms()
    atm_pos1=myxml1.atm_pos
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]]) 

    if xml_file2 is not None:
        myxml2=xml_io.Xml_sys_reader(xml_file2)
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

def get_pol(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False):
    myxml1=xml_io.Xml_sys_reader(xml_file)
    myxml1.get_atoms()
    atm_pos1=myxml1.atm_pos
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    ref_str = make_supercell(my_atms1,[[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]]) 

    if xml_file2 is not None:
        myxml2=xml_io.Xml_sys_reader(xml_file2)
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
    disps = my_pol.get_disp(final_Str_Hist)
    return(my_pol)
   
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

def get_layer_atoms(strc,c_z,dir=2,tol=0.1):
    positions = strc.get_positions()
    layer_atms = []
    for ipos,position in enumerate(positions):
        if abs(position[dir]-c_z) < tol:
            layer_atms.append(ipos)
    return(layer_atms)

def get_atoms_of_type(strc,ipositions=None,atom_type='Ti'):
    atoms_of_type = []
    chem_syms = strc.get_chemical_symbols()
    if ipositions==None:
        ipositions = range(len(chem_syms))
    for ipos in ipositions:
        if chem_syms[ipos]==atom_type:
            atoms_of_type.append(ipos)
    return(atoms_of_type)

def get_z_atoms(strc,dir = 2,tol=1):
    positions = strc.get_positions()
    z_atoms = [0]
    all_zs = [positions[0,2]]            
    for iposition,position in enumerate(positions): 
        found = False
        for z in all_zs: 
            if abs(position[dir]-z)<tol:
                found = True
        if not found:
            all_zs.append(position[2])
            z_atoms.append(iposition)

    return(z_atoms)

def layers_props(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,plot_dire=[1,1,1],ave_str=False):
    my_pol = get_pol(xml_file,NC_FILE_STR,dim,xml_file2=None,NC_stp=-1,Fast_map=True,cntr_at = ['Ti'],plot_dire=[1,1,1],ave_str=False)
    direction = 2
    if ave_str:
        final_Str_Hist = mync.get_avg_str(NC_FILE_STR,init_stp=NC_stp)
    else:
        final_Str_Hist = mync.get_NC_str(NC_FILE_STR,stp=NC_stp)

    Str_ref = my_pol.Str_ref

    disps = my_pol.get_disp(final_Str_Hist)

    Fnl_str = my_pol.Final_strc

    z_atom = get_z_atoms(Fnl_str,tol=1)
    chem_syms = Fnl_str.get_chemical_symbols()
    layers = []
    positions = Fnl_str.get_positions()
    for z_cntr in z_atom:
        z = positions[z_cntr,2]
        layers.append(get_layer_atoms(Fnl_str,z,tol=0.5))
        
    dis_dic = []
    avg_dis_dic = []
    chem_sym_lyrs = []
    beta_layers = []
    eta_layers = []

    for ilayer,layer in enumerate(layers):
        chem_sym_lyrs.append([])
        disp_dict_lyr = {}
        disp_avg_lyr = {}
        temp_symsbs = []
        dis_dic.append([])
        avg_dis_dic.append([])
        beta_layers.append([])
        eta_layers.append([])
        for atm in layer:
            temp_symsbs.append(chem_syms[atm])
            
            if chem_syms[atm] in disp_dict_lyr.keys():
                disp_dict_lyr[chem_syms[atm]] += disps[atm][direction]
            else:
                disp_dict_lyr.update({chem_syms[atm] : disps[atm][direction]})

        for mykey in disp_dict_lyr.keys():
            num_atms = temp_symsbs.count(mykey)
            disp_avg_lyr.update({mykey:disp_dict_lyr[mykey]/num_atms})

        if 'Ti' in disp_dict_lyr.keys():
            beta_layers[ilayer].append((disp_avg_lyr['Ti']+disp_avg_lyr['O'])/2)
            eta_layers[ilayer].append((disp_avg_lyr['Ti']-disp_avg_lyr['O'])/2)
        elif 'Sr' in disp_dict_lyr.keys():
            beta_layers[ilayer].append((disp_avg_lyr['Sr']+disp_avg_lyr['O'])/2)
            eta_layers[ilayer].append((disp_avg_lyr['Sr']-disp_avg_lyr['O'])/2)
        elif 'Pb' in disp_dict_lyr.keys():
            beta_layers[ilayer].append((disp_avg_lyr['Pb']+disp_avg_lyr['O'])/2)
            eta_layers[ilayer].append((disp_avg_lyr['Pb']-disp_avg_lyr['O'])/2)

        chem_sym_lyrs[ilayer].append(temp_symsbs)          
        dis_dic[ilayer].append(disp_dict_lyr)
        avg_dis_dic[ilayer].append(disp_avg_lyr)
       
    print('  AVG_disp   == \n',avg_dis_dic)
    print(10*'---')
    print('  beta   == \n',beta_layers)
    print(10*'---')
    print('  eta   == \n',eta_layers)
        
