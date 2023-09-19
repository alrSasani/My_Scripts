from itertools import permutations
import spglib as spg
import SC_xml_potential
import numpy as np
import os
import xml.etree.ElementTree as ET
from math import ceil
import ase
from ase import Atoms
from ase.units import Bohr
#from ase.data import atomic_numbers, atomic_masses
from ase.build import make_supercell
from ase.io import write
from phonopy.units import AbinitToTHz
#from my_functions import *
import my_functions
thz_cm = 33.356/1.374673102
import copy

###########################################################

class Xml_sys:
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

class Har_interface:
    ''' interface harmonic potential generation: This class is given two xml file for two materilas for the interface two 3x3 matirixes(SC_mat1 and SC_mat2) which give the matirx for
     one side of the SL (these should have SC_mat1[0,0]=SC_mat2[0,0] and SC_mat1[1,1]=SC_mat2[1,1] while SC_mat1[2,2] and SC_mat2[2,2] define the thickness of the each material
     in two sides of the SL) '''
    
    def __init__(self, xml_file1, SCMAT_1, xml_file2, SCMAT_2, symmetric=False,negelect_A_SITE=False,negelect_Tot_FCs=True):
        self.negelect_A_SITE = negelect_A_SITE
        #self.negelect_Tot_FCs = negelect_Tot_FCs
        #print(self.negelect_Tot_FCs )
        self.xmls_objs = {}
        self.uc_atoms = {}
        self.SCMATS = {}
        self.loc_FCs = {}
        self.tot_FCs = {}
        self.has_tot_FC = not negelect_Tot_FCs
        self.__Curnt_id = 0
        self.add_material(xml_file1, SCMAT_1)
        self.add_material(xml_file2, SCMAT_2)
        self.symmetric = symmetric
        self.Constr_SL(symmetric)
        self.has_weight = False
        self.loc_SL_FCDIC = {}
        self.tot_SL_FCDIC = {}
        self.tot_mykeys = []
        self.loc_mykeys = []

    @property
    def Curnt_id(self):
        return(self.__Curnt_id)

    def add_material(self, xml_file, SCMAT):
        my_xml_obj = Xml_sys(xml_file, mat_id=str(
            self.__Curnt_id), extract_dta=True)
        self.SCMATS[str(self.__Curnt_id)] = np.array(SCMAT)
        self.xmls_objs[str(self.__Curnt_id)] = my_xml_obj
        self.uc_atoms[str(self.__Curnt_id)] = my_xml_obj.ase_atoms
        self.loc_FCs[str(self.__Curnt_id)] = my_xml_obj.loc_FC_tgs
        self.tot_FCs[str(self.__Curnt_id)] = my_xml_obj.tot_FC_tgs
        self.has_tot_FC *= my_xml_obj.has_tot_FC
        self.__Curnt_id += 1

    def get_cells(self):  # FIXME GENERALISE THIS FUNCTION:
        temp_loc_keys1 = self.xmls_objs[str(0)].loc_cells
        SC_mat1 = self.SCMATS[str(0)]
        temp_loc_keys2 = self.xmls_objs[str(1)].loc_cells
        temp_tot_keys1 = self.xmls_objs[str(0)].tot_cells
        temp_tot_keys2 = self.xmls_objs[str(1)].tot_cells
        SC_mat2 = self.SCMATS[str(1)]
        ##############################
        minxl = min(ceil(temp_loc_keys1[0][0]/SC_mat1[0][0]), ceil(temp_loc_keys2[0][0]/SC_mat1[0][0]), ceil(
            temp_tot_keys1[0][0]/SC_mat1[0][0]), ceil(temp_tot_keys2[0][0]/SC_mat1[0][0]))
        maxxl = max(ceil(temp_loc_keys1[0][1]/SC_mat1[0][0]), ceil(temp_loc_keys2[0][1]/SC_mat1[0][0]), ceil(
            temp_tot_keys1[0][1]/SC_mat1[0][0]), ceil(temp_tot_keys2[0][1]/SC_mat1[0][0]))
        minyl = min(ceil(temp_loc_keys1[1][0]/SC_mat1[1][1]), ceil(temp_loc_keys2[1][0]/SC_mat1[1][1]), ceil(
            temp_tot_keys1[1][0]/SC_mat1[1][1]), ceil(temp_tot_keys2[1][0]/SC_mat1[1][1]))
        maxyl = max(ceil(temp_loc_keys1[1][1]/SC_mat1[1][1]), ceil(temp_loc_keys2[1][1]/SC_mat1[1][1]), ceil(
            temp_tot_keys1[1][1]/SC_mat1[1][1]), ceil(temp_tot_keys2[1][1]/SC_mat1[1][1]))
        minzl = min(ceil(temp_loc_keys1[2][0]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(temp_loc_keys2[2][0]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(
            temp_tot_keys1[2][0]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(temp_tot_keys2[2][0]/(SC_mat1[2][2]+SC_mat2[2][2])))
        maxzl = max(ceil(temp_loc_keys1[2][1]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(temp_loc_keys2[2][1]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(
            temp_tot_keys1[2][1]/(SC_mat1[2][2]+SC_mat2[2][2])), ceil(temp_tot_keys2[2][1]/(SC_mat1[2][2]+SC_mat2[2][2])))
        return(np.array([[minxl, maxxl], [minyl, maxyl], [minzl, maxzl]]))
        ###############################

    def find_tag_index(self, tags, tag):
        for i, ii in enumerate(tags):
            if tag[0] == ii[0] and tag[1] == ii[1]:
                return(i)

    def get_atm_ij_diff_in_UC(self):
        STRC = self.STRC
        natom = len(STRC)
        STRC_uc_cell = self.STRC_uc_cell
        tag_id = STRC.get_array('tag_id')
        indx_tag = []
        for i in range(natom):
            indx_tag.append(self.find_tag_index(
                self.uc_atoms[tag_id[i][1]].get_array('tag_id'), tag_id[i]))
        atm_ij_diff_in_mat = np.zeros((natom, natom, 3))
        for i in range(natom):
            for j in range(natom):
                atm_ij_diff_in_mat[j, i] = np.dot(STRC_uc_cell, (self.uc_atoms[tag_id[i][1]].get_scaled_positions()[
                                                  indx_tag[i]]-self.uc_atoms[tag_id[j][1]].get_scaled_positions()[indx_tag[j]]))
        return(atm_ij_diff_in_mat)

    def get_match_pairs(self):  # FIXME
        self.maped_strs = {}
        self.maped_strs['0'] = get_mapped_strcs(
            self.uc_atoms['0'], self.uc_atoms['1'], Ret_index=True)
        self.maped_strs['1'] = get_mapped_strcs(
            self.uc_atoms['1'], self.uc_atoms['0'], Ret_index=True)
        symbls_0 = self.uc_atoms['0'].get_chemical_symbols()
        symbls_1 = self.uc_atoms['1'].get_chemical_symbols()
        self.diff_elements = []
        for i in range(len(self.uc_atoms['0'])):
            if symbls_0[i] != symbls_1[self.maped_strs['0'][i]]:
                self.diff_elements.append(symbls_0[i])
                self.diff_elements.append(symbls_1[self.maped_strs['0'][i]])
        if len(self.diff_elements) == 0:
            self.diff_elements.append(symbls_0[0])

    def get_NL(self, cutoff=0):
        if cutoff == 0:
            sccuof = ase.neighborlist.natural_cutoffs(self.STRC, mult=1.5)
        else:
            # cutoff = np.linalg.norm(self.STRC_uc_cell)*(3)**0.5/4
            chmsym = self.STRC.get_chemical_symbols()
            uniq_syms = list(set(chmsym))
            cutof_dic = {}
            for sym in uniq_syms:
                cutof_dic[sym] = cutoff
            sccuof = [cutof_dic[sym] for sym in chmsym]
        #self.nl = ase.neighborlist.NeighborList(sccuof,sorted=False,self_interaction = False,bothways=True)
        self.nl = ase.neighborlist.NewPrimitiveNeighborList(
            sccuof, sorted=False, self_interaction=False, bothways=True)
        self.nl.update([1, 1, 1], self.STRC.get_cell(),
                       self.STRC.get_positions())

    def get_FC_weight(self, req_symbs=[]):
        if len(req_symbs) == 0:
            req_symbs = self.diff_elements
        cutoff = self.STRC_uc_cell[0][0]
        self.get_NL()
        natom = len(self.STRC)
        tag_id = self.STRC.get_array('tag_id')
        chmsym = self.STRC.get_chemical_symbols()
        self.FC_weights = []
        self.AVG_flags = []
        for aa in range(natom):
            NNs, offsets = self.nl.get_neighbors(aa)
            weght = np.zeros(self.__Curnt_id)
            ids = []
            if chmsym[aa] not in req_symbs:
                for bb, offset in zip(NNs, offsets):
                    dist = np.linalg.norm(self.STRC.get_distance(
                        aa, bb, vector=True)+np.dot(self.STRC.get_cell(), offset))
                    # self.STRC_uc_cell[0][0]:
                    if dist <= cutoff and chmsym[bb] in req_symbs:
                        for id in range(self.__Curnt_id):
                            if tag_id[bb][1] == f'{id}':
                                weght[id] += 1
                                ids.append(id)
            else:
                weght[int(tag_id[aa][1])] = 1
                ids.append(int(tag_id[aa][1]))
            ids = list(set(ids))
            tmp_sum = sum(np.array(weght))
            if len(ids) > 1:
                self.AVG_flags.append(True)
            else:
                self.AVG_flags.append(False)
            weght = np.array(weght)/tmp_sum
            self.FC_weights.append(weght)
        self.has_weight = True
        # print(self.FC_weights)

    def get_LocTot_FC(self, UC_key, id):
        loc_found, tmp_has_tot_FC = False, False
        temp_loc1 = np.zeros((3, 3))
        temp_tot1 = np.zeros((3, 3))
        if UC_key in self.loc_FCs[id].keys():
            loc_found = True
            temp_loc1 = self.loc_FCs[id][UC_key]

        if self.has_tot_FC and UC_key in self.tot_FCs[id].keys():
            temp_tot1 = self.tot_FCs[id][UC_key]
            tmp_has_tot_FC = True
        return(temp_loc1, temp_tot1, [loc_found, tmp_has_tot_FC])

    def get_Uclls_in_STRC(self):
        STRC = self.STRC
        natom = len(STRC)
        STR_POSs = STRC.get_positions()
        atm_ij_diff_in_mat = self.get_atm_ij_diff_in_UC()
        STRC_inv_uc_cell = np.linalg.inv(self.STRC_uc_cell)
        cells_vecs = np.zeros((natom, natom, 3))
        for atm_i in range(natom):
            for atm_j in range(natom):
                dists = STR_POSs[atm_i]-STR_POSs[atm_j] - \
                    atm_ij_diff_in_mat[atm_j, atm_i]
                cells_vecs[atm_i, atm_j, :] = np.dot(
                    (1/0.98)*STRC_inv_uc_cell, dists)
        return(cells_vecs)

    def get_STR_FCDIC(self):
        id_pars = {'0': '1', '1': '0'}
        self.get_match_pairs()
        maped_strs = self.maped_strs
        Cells = self.get_cells()
        STRC = self.STRC
        natom = len(STRC)
        STRC_inv_uc_cell = np.linalg.inv(self.STRC_uc_cell)
        STRC_Cell = STRC.get_cell()
        if not self.has_weight:
            req_elemtsns = self.diff_elements
            # = self.diff_elements  # FIXME CHANGE IT WITH PROPER MATRIX
            self.get_FC_weight(req_symbs=req_elemtsns)
        FC_weights = self.FC_weights
        tag_id = STRC.get_array('tag_id')
        Ucells_vecs_in_STRC = self.get_Uclls_in_STRC()
        for prd1 in range(Cells[0, 0]-1, Cells[0, 1]+1):
            for prd2 in range(Cells[1, 0]-1, Cells[1, 1]+1):
                for prd3 in range(Cells[2, 0]-1, Cells[2, 1]+1):
                    per_dist = np.dot(np.array([prd1, prd2, prd3]), STRC_Cell)
                    Per_cells = np.dot((1/0.98)*STRC_inv_uc_cell, per_dist)
                    STR_cell_key = '{} {} {}'.format(prd1, prd2, prd3)
                    for atm_i in range(natom):
                        tag_i = tag_id[atm_i][0]
                        id_i = tag_id[atm_i][1]
                        for atm_j in range(natom):
                            tag_j = tag_id[atm_j][0]
                            id_j = tag_id[atm_j][1]
                            cell_b = Ucells_vecs_in_STRC[atm_j,
                                                         atm_i] + Per_cells
                            cell_b = list(map(int, cell_b))
                            SC_key = f'{id_i}{atm_i}_{id_j}{atm_j}_{prd1}_{prd2}_{prd3}'
                            if id_i == id_j:
                                # csr = '11'
                                id_j = id_pars[id_i]
                                tag_ni = str(maped_strs[id_j][int(tag_i)])
                                tag_nj = str(maped_strs[id_j][int(tag_j)])
                                UC_key1 = f'{id_i}{tag_i}_{id_i}{tag_j}_{cell_b[0]}_{cell_b[1]}_{cell_b[2]}'
                                UC_key2 = f'{id_j}{tag_ni}_{id_j}{tag_nj}_{cell_b[0]}_{cell_b[1]}_{cell_b[2]}'
                            else:
                                # csr = '22'
                                # Mapping of atoms between two structures:
                                # FIXME TO BE CHECKED
                                tag_ni = str(maped_strs[id_j][int(tag_i)])
                                # FIXME TO BE CHECKED
                                tag_nj = str(maped_strs[id_i][int(tag_j)])
                                UC_key1 = f'{id_i}{tag_i}_{id_i}{tag_nj}_{cell_b[0]}_{cell_b[1]}_{cell_b[2]}'
                                UC_key2 = f'{id_j}{tag_ni}_{id_j}{tag_j}_{cell_b[0]}_{cell_b[1]}_{cell_b[2]}'

                                # ##### Neglecting A site A site interactions

                                if tag_i=='0' and tag_j=='0' and self.negelect_A_SITE:
                                   UC_key1 = '0' 
                                   UC_key2 = '0'

                            temp_loc1, temp_tot1, found_flgs1 = self.get_LocTot_FC(
                                UC_key1, id_i)
                            temp_loc2, temp_tot2, found_flgs2 = self.get_LocTot_FC(
                                UC_key2, id_j)
                            tmp_weights = np.array(
                                FC_weights[atm_i])+np.array(FC_weights[atm_j])
                            avg_weights = tmp_weights/sum(tmp_weights)

                            # if self.negelect_Tot_FCs:
                            #     found_flgs1[1] = False
                            #     found_flgs2[1] = False

                            if found_flgs1[0] or found_flgs2[0]:
                                self.loc_SL_FCDIC[SC_key] = (
                                    avg_weights[int(id_i)]*temp_loc1 + avg_weights[int(id_j)]*temp_loc2)
                                if STR_cell_key not in (self.loc_mykeys):
                                    self.loc_mykeys.append(STR_cell_key)

                            if found_flgs1[1] or found_flgs2[1]:
                                self.tot_SL_FCDIC[SC_key] = (
                                    avg_weights[int(id_i)]*temp_tot1 + avg_weights[int(id_j)]*temp_tot2)
                                if STR_cell_key not in (self.tot_mykeys):
                                    self.tot_mykeys.append(STR_cell_key)

        self.has_SC_FCDIC = True

    def reshape_FCDIC(self, STRC=0):
        # FIXME ADD MAPPING TO REQUIRED STRUCTURE
        STRC = self.STRC
        if not self.has_SC_FCDIC:
            self.get_SC_FCDIC()
        self.Fin_loc_FC = {}
        if self.has_tot_FC:
            self.Fin_tot_FC = {}
            my_keys = self.tot_mykeys
        else:
            my_keys = self.loc_mykeys
        tag_id = STRC.get_array('tag_id')
        natom = len(STRC)
        for my_key in (my_keys):
            loc_key_found = False
            tot_key_found = False
            my_cell = [int(x) for x in my_key.split()]
            tmp_loc_FC = np.zeros((3*natom, 3*natom))
            tmp_tot_FC = np.zeros((3*natom, 3*natom))
            cnt_a = 0
            for atm_a_tag, atm_a_id in tag_id:
                cnt_b = 0
                for atm_b_tag, atm_b_id in tag_id:
                    my_index = f'{atm_a_id}{cnt_a}_{atm_b_id}{cnt_b}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
                    if my_index in self.loc_SL_FCDIC.keys():
                        loc_key_found = True
                        tmp_loc_FC[cnt_a*3:cnt_a*3+3, cnt_b *
                                   3:cnt_b*3+3] = self.loc_SL_FCDIC[my_index]
                    if my_index in self.tot_SL_FCDIC.keys() and (self.xmls_objs[atm_a_id].has_tot_FC or self.xmls_objs[atm_b_id].has_tot_FC)  :
                        tot_key_found = True
                        tmp_tot_FC[cnt_a*3:cnt_a*3+3, cnt_b *
                                   3:cnt_b*3+3] = self.tot_SL_FCDIC[my_index]
                    cnt_b += 1
                cnt_a += 1
            if loc_key_found:
                self.Fin_loc_FC[my_key] = tmp_loc_FC
            if tot_key_found:
                self.Fin_tot_FC[my_key] = tmp_tot_FC
        self.has_FIN_FCDIC = True

    def Constr_SL(self, symmetric,ref_cell = 'cell_1'):
        if symmetric:
            zidir = self.SCMATS[str(0)][2, 2]
            zdir_1 = int(zidir/2)
            zdir_2 = int(zidir-zdir_1)
            # SCMAT_1 = self.SCMATS[str(0)]
            # SCMAT_1[2,2] = zdir_1
            # SCMAT_2 = self.SCMATS[str(0)]
            # SCMAT_2[2,2] = zdir_2
            SCMAT_2 = [[self.SCMATS[str(0)][0][0], 0, 0], [
                0, self.SCMATS[str(0)][0][0], 0], [0, 0, zdir_2]]
            SCMAT_1 = [[self.SCMATS[str(0)][0][0], 0, 0], [
                0, self.SCMATS[str(0)][0][0], 0], [0, 0, zdir_1]]

            a1 = make_supercell(self.uc_atoms[str(0)], SCMAT_1)
            self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell()
            temp_UC_ATOMS = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(
            ), scaled_positions=self.uc_atoms[str(1)].get_scaled_positions(), cell=self.uc_atoms[str(0)].get_cell())
            temp_UC_ATOMS.set_array(
                'BEC', self.uc_atoms[str(1)].get_array('BEC'))
            temp_UC_ATOMS.set_array(
                'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
            temp_UC_ATOMS.set_array(
                'str_ph', self.uc_atoms[str(1)].get_array('str_ph'))
            a2 = make_supercell(temp_UC_ATOMS, self.SCMATS[str(1)])
            a3 = make_supercell(self.uc_atoms[str(0)], SCMAT_2)
            temp_STRC = make_SL(a1, a2,ref_cell = ref_cell)
            STRC = make_SL(temp_STRC, a3)
            STRC.wrap()
            self.STRC = STRC
            self.get_ref_SL()

        else:
            a1 = make_supercell(self.uc_atoms[str(0)], self.SCMATS[str(0)])
            self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell()
            temp_UC_ATOMS = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(
            ), scaled_positions=self.uc_atoms[str(1)].get_scaled_positions(), cell=self.uc_atoms[str(0)].get_cell())
            temp_UC_ATOMS.set_array(
                'BEC', self.uc_atoms[str(1)].get_array('BEC'))
            temp_UC_ATOMS.set_array(
                'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
            temp_UC_ATOMS.set_array(
                'str_ph', self.uc_atoms[str(1)].get_array('str_ph'))
            a2 = make_supercell(temp_UC_ATOMS, self.SCMATS[str(1)])
            STRC = make_SL(a1, a2,ref_cell = ref_cell)
            STRC.wrap()
            self.STRC = STRC
            self.get_ref_SL()

    def write_xml(self, out_put='test.xml', asr=0, asr_chk=0):
        
        STRC = self.STRC  # xml1.atm_pos
        natm = len(STRC)
        self.cal_eps_inf()
        SC_FC = self.Fin_loc_FC
        if self.has_tot_FC:
            tSC_FC = self.Fin_tot_FC
            keys = tSC_FC.keys()
        else:
            keys = SC_FC.keys()
        
        self.get_ref_SL()
        lt_scll = self.ref_cell.get_cell()/Bohr    #STRC.get_cell()/Bohr
        SC_phon = np.zeros((natm, 3))
        SC_dmat = np.zeros((natm, natm))
        if asr:
            self.asr_impose()
            self.asr_chk()
        if asr_chk:
            self.asr_chk()
        str_ph = STRC.get_array('str_ph')
        masses = STRC.get_masses()
        self.SL_BEC_cal()
        BEC = self.SL_BEC   # STRC.get_array('BEC') # FIXME
        poss = self.ref_cell.get_positions()/Bohr
        SCL_elas = (np.linalg.det(self.SCMATS['0'])*(self.xmls_objs['0'].ela_cons)+np.linalg.det(
            self.SCMATS['1'])*(self.xmls_objs['1'].ela_cons))
        out_xml = open(out_put, 'w')
        out_xml.write('<?xml version="1.0" ?>\n')
        out_xml.write('<System_definition>\n')
        out_xml.write('  <energy>\n  {:.14E}\n  </energy>\n'.format(np.linalg.det(
            self.SCMATS['0'])*self.xmls_objs['0'].ref_eng+np.linalg.det(self.SCMATS['1'])*self.xmls_objs['1'].ref_eng))
        out_xml.write(
            '  <unit_cell units="bohrradius">\n {}  </unit_cell>\n'.format(to_text(lt_scll)))
        out_xml.write(
            '  <epsilon_inf units="epsilon0">\n  {}  </epsilon_inf>\n'.format(to_text((self.SL_eps_inf))))
        out_xml.write(
            '  <elastic units="hartree">\n  {}  </elastic>\n'.format(to_text(SCL_elas)))
        for i in range(natm):
            out_xml.write(
                f'  <atom mass="  {masses[i]}" massunits="atomicmassunit">\n    <position units="bohrradius">\n   {one_text(poss[i])}</position>\n    <borncharge units="abs(e)">\n   {to_text(BEC[i])}</borncharge>\n  </atom>\n')

        for key in keys:
            if key in (SC_FC.keys()):
                out_xml.write(
                    '  <local_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </local_force_constant>\n'.format(to_text((SC_FC[key])), key))
            if self.has_tot_FC:
                out_xml.write(
                    '  <total_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </total_force_constant>\n'.format(to_text((tSC_FC[key])), key))

       # xml.write('  <phonon>\n    <qpoint units="2pi*G0">  {}</qpoint>\n    <frequencies units="reciprocal cm">\n  {}    </frequencies>\n    <dynamical_matrix units="hartree/bohrradius**2">\n {}    </dynamical_matrix>\n   </phonon>\n'.format(self.xml.qpoint,to_text(SC_phon),to_text(SC_dmat)))
        for i in range(len(self.xmls_objs['0'].strain)):
            out_xml.write('  <strain_coupling voigt=" {}">\n    <strain>  {}    </strain>\n    <correction_force units="hartree/bohrradius">\n  {}    </correction_force>\n  </strain_coupling>\n'.format(
                i, (self.xmls_objs['0'].strain[i]), to_text(str_ph[:, i, :])))
        out_xml.write('</System_definition>')
        out_xml.close()

    def find_intrfc_atms(self):
        mdl_layer = []
        CPOS_SC1 = self.mySC1.get_positions()
        zmax = 0
        for i in range(len(CPOS_SC1)):
            if zmax <= CPOS_SC1[i, 2]:
                zmax = CPOS_SC1[i, 2]
        tmp_SC1 = make_supercell(self.UC_atoms1, self.SC_mat2)
        SL1 = make_SL(self.mySC1, tmp_SC1)
        CPOS_SL1 = SL1.get_positions()
        for i in range(len(CPOS_SL1)):
            if abs(CPOS_SL1[i, 2]-zmax) < 0.001:
                mdl_layer.append(i)
        finl_layer = []
        zmax = 0
        for i in range(len(CPOS_SL1)):
            if zmax <= CPOS_SL1[i, 2]:
                zmax = CPOS_SL1[i, 2]

        for i in range(len(CPOS_SL1)):
            if abs(CPOS_SL1[i, 2]-zmax) < 0.001:
                finl_layer.append(i)

        return(mdl_layer, finl_layer)

    def asr_impose(self):
        #print('ASR imposing')
        for atm_i in range(len(self.my_atm_list)):
            # asr_sr=np.zeros((3,3))
            asr_tot = np.zeros((3, 3))
            for atm_j in range(len(self.my_atm_list)):
                for key_j in self.Fin_tot_FC.keys():
                    # asr_sr+=self.Fin_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                    if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                        asr_tot += self.Fin_tot_FC[key_j][3 *
                                                          atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]
            if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                self.Fin_tot_FC['0 0 0'][3*atm_i:3 *
                                         atm_i+3, 3*atm_i:3*atm_i+3] -= asr_tot
            # else:
                #self.Fin_FC['0 0 0'][3*atm_i:3*atm_i+3,3*atm_i:3*atm_i+3]-=asr_sr

    def asr_chk(self):
        print('ASR chking')
        if 1:
            for atm_i in range(len(self.my_atm_list)):
                asr_sr = np.zeros((3, 3))
                asr_tot = np.zeros((3, 3))
                for atm_j in range(len(self.my_atm_list)):
                    for key_j in self.Fin_tot_FC.keys():
                        # asr_sr+=self.Fin_loc_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                        if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                            asr_tot += self.Fin_tot_FC[key_j][3 *
                                                              atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]
                print('Total')
                print(asr_tot)
                # print('SR')
                # print(asr_sr)

    def asr_intfce(self, a, b):
        #print('ASR imposing')
        for atm_i in range(len(self.my_atm_list)):
            asr_tot = np.zeros((3, 3))
            for atm_j in range(len(self.my_atm_list)):
                for key_j in self.Fin_FC.keys():
                    # asr_sr+=self.Fin_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                    if self.xml.has_tot_FC:
                        asr_tot += self.Fin_tot_FC[key_j][3 *
                                                          atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]
            if self.xml.has_tot_FC:
                self.Fin_tot_FC['0 0 0'][3*a:3*a+3, 3*b:3*b+3] -= asr_tot
            # else:
                #self.Fin_FC['0 0 0'][3*atm_i:3*atm_i+3,3*atm_i:3*atm_i+3]-=asr_sr

    def SL_BEC_cal(self):
        id_pars = {'0': '1', '1': '0'}
        if not self.has_weight:
            self.get_FC_weight(req_symbs=self.diff_elements)
        FC_weight = self.FC_weights
        self.cal_eps_inf()
        natom = len(self.STRC)
        eps_SL = self.SL_eps_inf
        eps = {}
        eps['0'] = self.xmls_objs['0'].eps_inf
        eps['1'] = self.xmls_objs['1'].eps_inf
        SCL = {}

        SCL['0'] = np.sqrt( np.sqrt(eps_SL[0][0]**2+eps_SL[1][1]**2+eps_SL[2][2]**2) / np.sqrt(eps['0'][0][0]**2+eps['0'][1][1]**2+eps['0'][2][2]**2) )        
        SCL['1'] = np.sqrt(np.sqrt(eps_SL[0][0]**2+eps_SL[1][1]**2+eps_SL[2][2]**2)/np.sqrt(eps['1'][0][0]**2+eps['1'][1][1]**2+eps['1'][2][2]**2))
              
        # SCL['0'] = [np.sqrt(eps_SL[0][0]/eps['0'][0][0]), np.sqrt(eps_SL[1]
        #                                                           [1]/eps['0'][1][1]), np.sqrt(eps_SL[2][2]/eps['0'][2][2])]
        # SCL['1'] = [np.sqrt(eps_SL[0][0]/eps['1'][0][0]), np.sqrt(eps_SL[1]
        #                                                           [1]/eps['1'][1][1]), np.sqrt(eps_SL[2][2]/eps['1'][2][2])]
        maped_strs = self.maped_strs
        dic_id_bec = {}
        tag_id_0 = self.uc_atoms['0'].get_array('tag_id')
        BEC_0 = self.uc_atoms['0'].get_array('BEC')
        dic_id_bec['0'] = {}
        for i in range(len(tag_id_0)):
            dic_id_bec['0'].update({tag_id_0[i][0]: BEC_0[i]})

        dic_id_bec['1'] = {}
        BEC_1 = self.uc_atoms['1'].get_array('BEC')
        tag_id_1 = self.uc_atoms['1'].get_array('tag_id')
        for i in range(len(tag_id_1)):
            dic_id_bec['1'].update({tag_id_0[i][0]: BEC_1[i]})

        SL_BEC = np.zeros((natom, 3, 3))
        tag_id = self.STRC.get_array('tag_id')
        for i in range(natom):
            tag_i = tag_id[i][0]
            id_i = tag_id[i][1]
            SL_BEC[i, :, :] = FC_weight[i][int(id_i)]*(SCL[id_i]*dic_id_bec[id_i][tag_i]) + FC_weight[i][int(
                id_pars[id_i])]*(SCL[id_pars[id_i]]*dic_id_bec[id_pars[id_i]][str(maped_strs[id_pars[id_i]][int(tag_i)])])
        self.SL_BEC = SL_BEC

    def cal_eps_inf(self):
        self.SL_eps_inf = np.zeros((3, 3))
        l_1 = self.SCMATS['0'][2][2] / \
            (self.SCMATS['0'][2][2]+self.SCMATS['1'][2][2])
        l_2 = self.SCMATS['1'][2][2] / \
            (self.SCMATS['0'][2][2]+self.SCMATS['1'][2][2])
        eps_xx = l_1*self.xmls_objs['0'].eps_inf[0][0] + \
            l_2*self.xmls_objs['1'].eps_inf[0][0]
        
        eps_yy = l_1*self.xmls_objs['0'].eps_inf[1][1] + \
            l_2*self.xmls_objs['1'].eps_inf[1][1]
        
        eps_zz = 1/(l_1*(1/self.xmls_objs['0'].eps_inf[2]
                    [2])+l_2*(1/self.xmls_objs['1'].eps_inf[2][2]))
        self.SL_eps_inf[0, 0] = eps_xx
        self.SL_eps_inf[1, 1] = eps_yy
        self.SL_eps_inf[2, 2] = eps_zz

    def Constr_ref_SL(self, symmetric,ref_cell = 'cell_1'):
        if symmetric:
            zidir = self.SCMATS[str(0)][2, 2]
            zdir_1 = int(zidir/2)
            zdir_2 = int(zidir-zdir_1)
            # SCMAT_1 = self.SCMATS[str(0)]
            # SCMAT_1[2,2] = zdir_1
            # SCMAT_2 = self.SCMATS[str(0)]
            # SCMAT_2[2,2] = zdir_2
            SCMAT_2 = [[self.SCMATS[str(0)][0][0], 0, 0], [
                0, self.SCMATS[str(0)][0][0], 0], [0, 0, zdir_2]]
            SCMAT_1 = [[self.SCMATS[str(0)][0][0], 0, 0], [
                0, self.SCMATS[str(0)][0][0], 0], [0, 0, zdir_1]]

            a1 = make_supercell(self.uc_atoms[str(0)], SCMAT_1)
            self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell()
            temp_UC_ATOMS = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(
            ), scaled_positions=self.uc_atoms[str(1)].get_scaled_positions(), cell=self.uc_atoms[str(0)].get_cell())
            temp_UC_ATOMS.set_array(
                'BEC', self.uc_atoms[str(1)].get_array('BEC'))
            temp_UC_ATOMS.set_array(
                'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
            temp_UC_ATOMS.set_array(
                'str_ph', self.uc_atoms[str(1)].get_array('str_ph'))
            a2 = make_supercell(temp_UC_ATOMS, self.SCMATS[str(1)])
            a3 = make_supercell(self.uc_atoms[str(0)], SCMAT_2)
            temp_STRC = make_SL(a1, a2,ref_cell = ref_cell)
            STRC = make_SL(temp_STRC, a3)
            STRC.wrap()
            self.STRC = STRC
            self.get_ref_SL()

        else:
            a1 = make_supercell(self.uc_atoms[str(0)], self.SCMATS[str(0)])
            self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell()

            cell_par1 = self.uc_atoms[str(0)].get_cell_lengths_and_angles()
            cell_par2 = self.uc_atoms[str(1)].get_cell_lengths_and_angles()
            cell_parr_diff = cell_par2[2]-cell_par1[2]
            # print(cell_parr_diff)
            temp_cell = [cell_par1[0],cell_par1[1],cell_par2[2]]
            temp_UC_ATOMS = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(
            ), scaled_positions=self.uc_atoms[str(1)].get_scaled_positions(), cell=temp_cell)
            temp_UC_ATOMS.set_array(
                'BEC', self.uc_atoms[str(1)].get_array('BEC'))
            temp_UC_ATOMS.set_array(
                'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
            temp_UC_ATOMS.set_array(
                'str_ph', self.uc_atoms[str(1)].get_array('str_ph'))
            a2 = make_supercell(temp_UC_ATOMS, self.SCMATS[str(1)])
            STRC = make_SL(a1, a2,ref_cell = ref_cell,cell_parr_diff = cell_parr_diff/2)
            STRC.wrap()
            write('temp_a2.cif',a2,format='cif')
            write('STRC.cif',STRC,format='cif')
            return( STRC)


    def get_ref_SL(self):  # ref structure according to thesis of Carlos***
        ELC1 = self.xmls_objs['0'].ela_cons
        ELC2 = self.xmls_objs['1'].ela_cons
        tmp_SC1 = make_supercell(self.uc_atoms['0'], self.SCMATS['1'])
        mySC1 = make_supercell(self.uc_atoms['0'], self.SCMATS['0'])
        SL1 = make_SL(mySC1, tmp_SC1)
        ABC_SL1 = SL1.cell.cellpar()[0:3]
        # ScPOS_SL1 = SL1.get_scaled_positions()
        SL1_cell = SL1.get_cell()
        
        tmp_SC2 = make_supercell(self.uc_atoms['1'], self.SCMATS['0'])
        mySC2 = make_supercell(self.uc_atoms['1'], self.SCMATS['1'])
        SL2 = make_SL(tmp_SC2, mySC2)
        # ABC_SL2 = SL2.cell.cellpar()[0:3]
        # ScPOS_SL2 = SL2.get_scaled_positions()
        SL2_cell = SL2.get_cell()
        cell_1 = self.uc_atoms['0'].get_cell()
        cell_2 = self.uc_atoms['1'].get_cell()
        p1 = cell_1[2][2]/(cell_1[2][2]+cell_2[2][2])
        p2 = cell_2[2][2]/(cell_1[2][2]+cell_2[2][2])
        m = np.zeros((3))
        for indx in range(3):
            m[indx] = p1*ELC1[indx][indx]/(p2*ELC2[indx][indx])
        a_avg = np.zeros((3))
        for a_indx in range(3):
            a_avg[a_indx] = cell_1[a_indx][a_indx]*cell_2[a_indx][a_indx] * \
                (m[a_indx]*cell_1[a_indx][a_indx]+cell_2[a_indx][a_indx]) / \
                (m[a_indx]*cell_1[a_indx][a_indx]**2+cell_2[a_indx][a_indx]**2)
        # numbers1 = mySC1.get_atomic_numbers()
        # numbers2 = mySC2.get_atomic_numbers()
        # numbers_SL = [*numbers1, *numbers2]
        a0 = self.SCMATS['0'][0][0]*a_avg[0]
        a1 = self.SCMATS['0'][1][1]*a_avg[1]
        a2 = self.SCMATS['0'][2][2]*a_avg[2]+self.SCMATS['1'][2][2]*a_avg[2]
 
        # cell = SL2_cell
        cell_par1 = SL1.get_cell_lengths_and_angles()
        # cell_par2 = SL2.get_cell_lengths_and_angles()

        # print(cell_par1)
        # print(cell_par2)
        # coeff_ab = 0.1
        # coeff_c = coeff_ab
        # cell =  [cell_par1[0], cell_par1[1]+coeff_ab*(cell_par2[1]-cell_par1[1]), cell_par1[2]+coeff_c*(cell_par2[2]-cell_par1[2])]  #SL2_cell    
        # print(cell)
        STRC = self.Constr_ref_SL(self.symmetric,ref_cell = 'cell_1')
        # cell = [SL1_cell[0,0], SL1_cell[1,1], STRC.get_cell()[2,2]] 
        alpha = 1
        cell = [SL1_cell[0,0],SL1_cell[1,1],alpha*STRC.get_cell()[2,2]]
        # print(cell)

        my_SL = Atoms(numbers=STRC.get_atomic_numbers(), scaled_positions=STRC.get_scaled_positions(),
                      cell=cell, pbc=True)
        
        write('my_SL_cif',my_SL,format='cif')
        my_SL.set_array(
            'BEC', self.STRC.get_array('BEC'))
        my_SL.set_array(
            'tag_id', self.STRC.get_array('tag_id'))
        my_SL.set_array(
            'str_ph', self.STRC.get_array('str_ph'))

        self.ref_cell = my_SL


###########################################################
# interface Anharmonic potential generation:

class Anh_intrface(Har_interface):
    def __init__(self, har_xml1, anh_xml1, SC_mat1, har_xml2, anh_xml2, SC_mat2, miss_fit_trms=False, symmetric=False,vigt_missfit=None):
        Har_interface.__init__(self, har_xml1, SC_mat1,
                               har_xml2, SC_mat2, symmetric=symmetric)
        self.coeff = {}
        self.terms = {}
        self.STRC_terms = {}
        self.STRC_coeffs = {}
        self.vigt_missfit = vigt_missfit
        self.coeff['0'], self.terms['0'] = self.xml_anha(
            anh_xml1, self.uc_atoms['0'])
        self.coeff['1'], self.terms['1'] = self.xml_anha(
            anh_xml2, self.uc_atoms['1'])
        self.has_weight = False
        self.miss_fit_trms = miss_fit_trms
        # self.Constr_SL(symmetric)
        self.get_match_pairs()
        if not self.has_weight:
            req_elemtsns = self.diff_elements
            self.get_FC_weight(req_symbs=req_elemtsns)

    def STRC_trms(self, id_in='0'):
        id_pars = {'0': '1', '1': '0'}
        tol_04 = 10**-4
        UC_STR = self.uc_atoms[str(0)]
        coeff, trms = self.coeff[id_in], self.terms[id_in]
        Xred_STRC = self.STRC.get_scaled_positions()
        STRC_cell = self.STRC.get_cell()
        inv_STRC_cell = np.linalg.inv(STRC_cell)
        cPOS = UC_STR.get_positions()
        tag_id = self.STRC.get_array('tag_id')
        uc_cell = UC_STR.get_cell()
        ###################################
        total_coefs = len(coeff)
        my_strain = 0.01*np.array([1,1,1]) #self.get_strain(id_in)
        temp_voits = []
        strain_flag = []
        for ii,i in enumerate(my_strain):
            if abs(i) >= tol_04:
                strain_flag.append(True)
                temp_voits.append(ii+1)
            else:
                strain_flag.append(False)
        if self.vigt_missfit is not None:
            temp_voits = self.vigt_missfit
        print(f' The missfit strains material {id_in} are in directions : ',10*'***',temp_voits)
        if self.miss_fit_trms and any(strain_flag):           
            my_tags = self.xmls_objs[id_in].tags
            new_coeffs, new_trms = self.get_final_coeffs(
                coeff, trms, my_tags, my_strain, voigts=temp_voits)
            for ntrm_cntr in range(len(new_coeffs)):
                trms.append(new_trms[ntrm_cntr])
                coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]
        ####################################
        wrapPos = ase.geometry.wrap_positions
        my_terms = []
        zdirec1 = range((self.SCMATS[id_in][2, 2]) +
                        (self.SCMATS[id_pars[id_in]][2, 2]))
        tmp_coeffs = {}
        for cc in range(len(coeff)):
            tmp_coeffs[cc] = coeff[cc]
            my_terms.append([])
            if cc > total_coefs:
                my_SAT_terms = self.get_SATs(
                    trms[cc][0], self.xmls_objs[id_in])
            else:
                my_SAT_terms = trms[cc]
            for tc in range(len(my_SAT_terms)):
                for prd1 in range(self.SCMATS[id_in][0, 0]):
                    for prd2 in range(self.SCMATS[id_in][1, 1]):
                        for prd3 in zdirec1:
                            my_term = []
                            disp_cnt = 0
                            prd_dis = np.dot(uc_cell, [prd1, prd2, prd3])
                            temp_weight = 0
                            for disp in range(int(my_SAT_terms[tc][-1]['dips'])):
                                atm_a = int(my_SAT_terms[tc][disp]['atom_a'])
                                atm_b = int(my_SAT_terms[tc][disp]['atom_b'])
                                #cell_a0 = [int(x) for x in  my_SAT_terms[tc][disp]['cell_a'].split()]
                                cell_b0 = [
                                    int(x) for x in my_SAT_terms[tc][disp]['cell_b'].split()]
                                #id_a = my_SAT_terms[tc][disp]['tag_id_a'][1]
                                #id_b = my_SAT_terms[tc][disp]['tag_id_b'][1]
                                catm_a0 = cPOS[atm_a]
                                catm_b0 = np.dot(uc_cell, cell_b0)+cPOS[atm_b]
                                dst0 = catm_a0-catm_b0
                                catm_an = prd_dis + catm_a0
                                catm_bn = prd_dis + catm_b0
                                red_an = np.dot(inv_STRC_cell, catm_an)
                                red_bn = np.dot(inv_STRC_cell, catm_bn)
                                ind_an = find_index_xred(Xred_STRC, red_an)
                                ind_bn = find_index_xred(Xred_STRC, red_bn)
                                if ind_an == -1:
                                    wrp_a = wrapPos([catm_an], STRC_cell)[0]
                                    red_ann = np.dot(inv_STRC_cell, wrp_a)
                                    ind_an = find_index_xred(
                                        Xred_STRC, red_ann)
                                if ind_bn == -1:
                                    wrp_b = wrapPos([catm_bn], STRC_cell)[0]
                                    red_bnn = np.dot(inv_STRC_cell, wrp_b)
                                    ind_bn = find_index_xred(
                                        Xred_STRC, red_bnn)
                                cell_a = red_an-Xred_STRC[ind_an]
                                cell_b = list(
                                    map(int, red_bn-Xred_STRC[ind_bn]-np.array(cell_a)))
                                tag_an, id_an = tag_id[ind_an]
                                # tag_bn,id_bn = tag_id[ind_bn]
                                dst = [catm_an[i]-catm_bn[i] for i in range(3)]
                                dif_ds = [
                                    False if (abs(dst[i]-dst0[i]) > tol_04) else True for i in range(3)]
                                if all(dif_ds):
                                    # if id_an==id_in: #abs(self.FC_weights[ind_an][int(id_in)]) > tol_04 :
                                    trm_weight = self.FC_weights[ind_an][int(
                                        id_in)]+self.FC_weights[ind_bn][int(id_in)]
                                    # else:
                                    #   trm_weight = 0
                                    #   break
                                    temp_weight += trm_weight
                                    cell_b_Str = f'{cell_b[0]} {cell_b[1]} {cell_b[2]}'
                                    new_dis = {'atom_a': ind_an, 'cell_a': f'0 0 0', 'atom_b': ind_bn, 'cell_b': cell_b_Str, 'direction': my_SAT_terms[tc][disp]['direction'],
                                               'power': my_SAT_terms[tc][disp]['power'], 'weight': trm_weight/2,
                                               }
                                    my_term.append(new_dis)
                                else:
                                    break
                                disp_cnt += 1
                            if disp_cnt > 0:
                                temp_weight /= (2*disp_cnt)
                            else:
                                temp_weight = 1
                            if (int(my_SAT_terms[tc][-1]['dips']) == 0 or (disp_cnt == int(my_SAT_terms[tc][-1]['dips']) and (len(my_term) != 0))):
                                tmp_d = 0
                                if disp_cnt == 0:
                                    tmp_d = 1
                                for str_cnt in range(int(my_SAT_terms[tc][-1]['strain'])):
                                    my_term.append(
                                        {'power': my_SAT_terms[tc][disp_cnt+tmp_d+str_cnt]['power'], 'voigt': my_SAT_terms[tc][disp_cnt+tmp_d+str_cnt]['voigt']})
                            if len(my_term) == int(my_SAT_terms[tc][-1]['dips'])+int(my_SAT_terms[tc][-1]['strain']) and abs(temp_weight) > tol_04:
                                if (int(my_SAT_terms[tc][-1]['dips']) == 0 and int(my_SAT_terms[tc][-1]['strain']) != 0):
                                    my_term.append(
                                        {'weight': float(my_SAT_terms[tc][0]['weight'])*temp_weight})
                                my_term.append(
                                    {'dips': my_SAT_terms[tc][-1]['dips'], 'strain': my_SAT_terms[tc][-1]['strain']})
                                my_term[-1]['weight'] = float(
                                    my_SAT_terms[tc][0]['weight'])*temp_weight
                                if my_term not in (my_terms[cc]):
                                    my_terms[cc].append(my_term)

        self.STRC_terms[id_in] = my_terms
        self.STRC_coeffs[id_in] = tmp_coeffs

    def get_SATs(self, my_term, hxml):
        hxml.get_ase_atoms()
        hxml.get_str_cp()
        my_strain = hxml.strain
        my_atms = hxml.ase_atoms
        sds = spg.get_symmetry_dataset(my_atms)
        #print(f"space group number = {sds['number']}  INT = {sds['international']} ")
        rot = sds['rotations']
        trans = sds['translations']
        vgt_mat = np.zeros((7, 3, 3))
        vgt_mat[1, :, :] = np.reshape([float(v)
                                      for v in my_strain[0].split()], (3, 3))
        vgt_mat[2, :, :] = np.reshape([float(v)
                                      for v in my_strain[1].split()], (3, 3))
        vgt_mat[3, :, :] = np.reshape([float(v)
                                      for v in my_strain[2].split()], (3, 3))
        vgt_mat[4, :, :] = np.reshape([float(v)
                                      for v in my_strain[3].split()], (3, 3))
        vgt_mat[5, :, :] = np.reshape([float(v)
                                      for v in my_strain[4].split()], (3, 3))
        vgt_mat[6, :, :] = np.reshape([float(v)
                                      for v in my_strain[5].split()], (3, 3))
        dir_dic = {'x': 1, 'y': 2, 'z': 3}
        pos_prm = my_atms.get_positions()
        wrapPos = ase.geometry.wrap_positions
        cell_prm = my_atms.get_cell()
        my_trms = []
        for my_rot in range(len(rot)):
            disp_lst = []
            dis_tmp_weight = 1
            wrong_acell = False
            this_term = False
            for disp in range(my_term[-1]['dips']):
                atm_a = int(my_term[disp]['atom_a'])
                cell_a = [int(cll)
                          for cll in (my_term[disp]['cell_a'].split())]
                atm_b = int(my_term[disp]['atom_b'])
                cell_b = [int(cll)
                          for cll in (my_term[disp]['cell_b'].split())]
                power = my_term[disp]['power']
                vec_dir = dir_dic[my_term[disp]['direction']]
                rot_vec_dir = vgt_mat[vec_dir, :, :]
                rot_vec_dir = np.dot(
                    np.dot(rot[my_rot], rot_vec_dir), (np.linalg.inv(rot[my_rot])))

                for vgt_cnt in range(1, len(vgt_mat)):
                    diff = np.matrix.flatten(
                        abs(rot_vec_dir) - vgt_mat[vgt_cnt, :, :])
                    if not any(diff):
                        rot_voit_key = vgt_cnt

                pos_a = pos_prm[atm_a] + np.dot(cell_a, cell_prm)
                pos_b = pos_prm[atm_b] + np.dot(cell_b, cell_prm)

                rot_pos_a = np.dot(rot[my_rot], pos_a) + \
                    np.dot(trans[my_rot], cell_prm)
                rot_pos_b = np.dot(rot[my_rot], pos_b) + \
                    np.dot(trans[my_rot], cell_prm)

                wrp_a = wrapPos([rot_pos_a], cell_prm)[0]
                wrp_b = wrapPos([rot_pos_b], cell_prm)[0]

                ncell_a0 = rot_pos_a - wrp_a
                ncell_b = rot_pos_b - wrp_b - ncell_a0
                ncell_a = [0, 0, 0]  # ncell_a

                atm_a_ind = find_index(pos_prm, wrp_a, tol=0.0001)
                atm_b_ind = find_index(pos_prm, wrp_b, tol=0.0001)

                dst0 = [pos_a[0]-pos_b[0], pos_a[1] -
                        pos_b[1], pos_a[2]-pos_b[2]]
                dst = [rot_pos_a[0]-rot_pos_b[0], rot_pos_a[1] -
                       rot_pos_b[1], rot_pos_a[2]-rot_pos_b[2]]

                if abs(np.linalg.norm(dst0)-np.linalg.norm(dst)) > 0.00001:
                    wrong_acell = True
                    break

                dis_tmp_weight *= (rot[my_rot]
                                   [rot_voit_key-1, vec_dir-1])**int(power)

                ndis = {
                    'atom_a': atm_a_ind,
                    'cell_a': f"{round(ncell_a[0]/cell_prm[0,0])} {round(ncell_a[1]/cell_prm[1,1])} {round(ncell_a[2]/cell_prm[2,2])}",
                    'atom_b': atm_b_ind,
                    'cell_b': f"{round(ncell_b[0]/cell_prm[0,0])} {round(ncell_b[1]/cell_prm[1,1])} {round(ncell_b[2]/cell_prm[2,2])}",
                    'direction': get_key(dir_dic, rot_voit_key),
                    'power': power
                }
                disp_lst.append(ndis)
            tem_vogt_wgt = 1
            if not(wrong_acell):
                if my_term[-1]['dips'] != 0 and my_term[-1]['strain'] != 0 and len(disp_lst) > 0:
                    for strain in range(my_term[-1]['strain']):
                        voigt = int(my_term[disp+strain+1]['voigt'])
                        pwr_str = int(my_term[disp+strain+1]['power'])
                        vgt_trans = np.dot(
                            np.dot(rot[my_rot], vgt_mat[voigt, :, :]), (np.linalg.inv(rot[my_rot])))
                        for vgt_cnt in range(1, len(vgt_mat)):
                            diff = np.matrix.flatten(-(vgt_trans) -
                                                     vgt_mat[vgt_cnt, :, :])
                            if not any(diff):
                                rot_voit_key = vgt_cnt
                                vogt_wgt = -1
                            diff = np.matrix.flatten(
                                (vgt_trans) - vgt_mat[vgt_cnt, :, :])
                            if not any(diff):
                                rot_voit_key = vgt_cnt
                                vogt_wgt = 1
                        tem_vogt_wgt *= vogt_wgt**int(pwr_str)
                        disp_lst.append(
                            {'power': f' {pwr_str}', 'voigt': f' {rot_voit_key}'})

                for i in range(my_term[-1]['dips']):
                    disp_lst[i]['weight'] = " %2.6f" % (
                        dis_tmp_weight*tem_vogt_wgt)
                if len(disp_lst) > 0:
                    disp_lst.append(my_term[-1])
                    found = False
                    prm_lst = list(permutations(disp_lst[:]))
                    for prm_mem in prm_lst:
                        my_temp = [prm for prm in prm_mem]
                        if my_temp in my_trms:
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
                    vgt_trans = np.dot(
                        np.dot(rot[my_rot], vgt_mat[voigt, :, :]), (np.linalg.inv(rot[my_rot])))
                    for vgt_cnt in range(1, len(vgt_mat)):
                        diff = np.matrix.flatten(-(vgt_trans) -
                                                 vgt_mat[vgt_cnt, :, :])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = -1
                        diff = np.matrix.flatten(
                            (vgt_trans) - vgt_mat[vgt_cnt, :, :])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = 1
                    disp_lst.append(
                        {'power': f' {pwr_str}', 'voigt': f' {rot_voit_key}'})
                tem_vogt_wgt *= vogt_wgt**int(pwr_str)
                disp_lst[0]['weight'] = " %2.6f" % tem_vogt_wgt

                disp_lst.append(my_term[-1])
                prm_lst = list(permutations(disp_lst[:]))
                found = False
                for prm_mem in prm_lst:
                    my_temp = [prm for prm in prm_mem]
                    if my_temp in my_trms:
                        found = True

                if disp_lst not in my_trms and len(disp_lst) > 0 and not found:
                    my_trms.append(disp_lst)

        return(my_trms)

    def xml_anha(self, fname, atoms):
        tree = ET.parse(fname)
        root = tree.getroot()
        coeff = {}
        lst_trm = []
        car_pos = atoms.get_positions()
        cell = atoms.get_cell()
        tag_id = atoms.get_array('tag_id')
        abc = atoms.cell.cellpar()[0:3]
        for cc, inf in enumerate(root.iter('coefficient')):
            coeff[cc] = inf.attrib
            lst_trm.append([])
            for tt, trm in enumerate(inf.iter('term')):
                lst_trm[cc].append([])
                # print(trm.attrib)
                disp_cnt = 0
                strcnt = 0
                for dd, disp in enumerate(trm.iter('displacement_diff')):
                    disp_cnt += 1
                    dta = disp.attrib
                    atma = int(disp.attrib['atom_a'])
                    atmb = int(disp.attrib['atom_b'])
                    cella = (disp.find('cell_a').text)
                    cellb = (disp.find('cell_b').text)
                    cell_a = [float(x) for x in cella.split()]
                    cell_b = [float(x) for x in cellb.split()]
                    # print(abc[0])
                    # pos_a= [cell_a[0]*abc[0]+car_pos[atma,0],cell_a[1]*abc[1]+car_pos[atma,1],cell_a[2]*abc[2]+car_pos[atma,2]]
                    pos_a = np.dot(cell, cell_a)+car_pos[atma]
                    # pos_b=[cell_b[0]*abc[0]+car_pos[atmb,0],cell_b[1]*abc[1]+car_pos[atmb,1],cell_b[2]*abc[2]+car_pos[atmb,2]]
                    pos_b = np.dot(cell, cell_b)+car_pos[atmb]
                    dist = ((pos_a[0]-pos_b[0])**2+(pos_a[1] -
                            pos_b[1])**2+(pos_a[2]-pos_b[2])**2)**0.5
                    dta['cell_a'] = cella
                    dta['cell_b'] = cellb
                    dta['weight'] = trm.attrib['weight']
                    dta['tag_id_a'] = tag_id[atma]
                    dta['tag_id_b'] = tag_id[atmb]
                    # include tag_a_id a and b
                    lst_trm[cc][tt].append(dta)
                if disp_cnt == 0:
                    # dta={}
                    dist = 0
                    dta = trm.attrib
                    lst_trm[cc][tt].append(dta)
                for ss, strain in enumerate(trm.iter('strain')):
                    dta = strain.find('strain')
                    lst_trm[cc][tt].append(strain.attrib)
                    strcnt += 1

                lst_trm[cc][tt].append(
                    {'dips': disp_cnt, 'strain': strcnt, 'distance': dist})
        return(coeff, lst_trm)

    def wrt_anxml(self, fout, str_str=0):
        self.STRC_trms(id_in='0')
        self.STRC_trms(id_in='1')
        coeff1 = self.STRC_coeffs['0']  # self.coeff['0']
        trms1 = self.STRC_terms['0']
        coeff2 = self.STRC_coeffs['1']  # self.coeff['1']
        trms2 = self.STRC_terms['1']
        output = open(fout, 'w')
        output.write('<?xml version="1.0" ?>\n')
        output.write('<Heff_definition>\n')
        #print(f'terms_1 =  {len(trms1)}   terms_2 = {len(trms1)}    intr_trms_1 = {len(intrface_terms1)}   inter_terms_2 = {len(intrface_terms2)}')
        str_coeff_1 = []
        str_coeff_2 = []
        coef_cntr = 1

        l1 = self.SCMATS['0'][2][2]
        l2 = self.SCMATS['1'][2][2]

        for i in range(len(coeff1)):
            k = 0
            for j in range(len(self.STRC_terms['0'][i])):
                for k in range(int(trms1[i][j][-1]['dips'])):
                    if int(trms1[i][j][-1]['dips']) != 0 and k == 0 and j == 0:
                        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                            coef_cntr, coeff1[i]['value'], coeff1[i]['text'],))
                    if (k == 0) and int(trms1[i][j][-1]['dips']) != 0:
                        output.write('    <term weight="{}">\n'.format(
                            trms1[i][j][-1]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(
                        trms1[i][j][k]['atom_a'], trms1[i][j][k]['atom_b'], trms1[i][j][k]['direction'], trms1[i][j][k]['power']))
                    output.write(
                        '        <cell_a>{}</cell_a>\n'.format(trms1[i][j][k]['cell_a']))
                    output.write(
                        '        <cell_b>{}</cell_b>\n'.format(trms1[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')

                if int(trms1[i][j][-1]['dips']) != 0:
                    for l in range(int(trms1[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                            trms1[i][j][k+l+1]['power'], trms1[i][j][k+l+1]['voigt']))
                    output.write('    </term>\n')

            if len(trms1[i]) != 0:
                if int(trms1[i][0][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1
                else:
                    str_coeff_1.append(i)

        for i in range(len(coeff2)):
            k = 0
            for j in range(len(self.STRC_terms['1'][i])):
                for k in range(int(trms2[i][j][-1]['dips'])):
                    if int(trms2[i][j][-1]['dips']) != 0 and k == 0 and j == 0:
                        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                            coef_cntr, coeff2[i]['value'], coeff2[i]['text'],))
                    if (k == 0) and int(trms2[i][j][-1]['dips']) != 0:
                        output.write('    <term weight="{}">\n'.format(
                            trms2[i][j][-1]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(
                        trms2[i][j][k]['atom_a'], trms2[i][j][k]['atom_b'], trms2[i][j][k]['direction'], trms2[i][j][k]['power']))
                    output.write(
                        '        <cell_a>{}</cell_a>\n'.format(trms2[i][j][k]['cell_a']))
                    output.write(
                        '        <cell_b>{}</cell_b>\n'.format(trms2[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')

                if int(trms2[i][j][-1]['dips']) != 0:
                    for l in range(int(trms2[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                            trms2[i][j][k+l+1]['power'], trms2[i][j][k+l+1]['voigt']))
                    output.write('    </term>\n')

            if len(trms2[i]) != 0:
                if int(trms2[i][0][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1
                else:
                    str_coeff_2.append(i)

        if (str_str == 0 or str_str == 1):
            for i in str_coeff_1:
                output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                    coef_cntr, (l1/(l1+l2))*float(coeff1[i]['value']), coeff1[i]['text'],))
                for j in range(len(trms1[i])):
                    output.write('    <term weight="{}">\n'.format(
                        trms1[i][j][-2]['weight']))
                    for l in range(int(trms1[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                            trms1[i][j][l]['power'], trms1[i][j][l]['voigt']))
                    output.write('    </term>\n')
                output.write('  </coefficient>\n')
                coef_cntr += 1

        if (str_str == 0 or str_str == 2):
            for i in str_coeff_2:
                output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                    coef_cntr, (l2/(l1+l2))*float(coeff2[i]['value']), coeff2[i]['text'],))
                for j in range(len(trms2[i])):
                    output.write('    <term weight="{}">\n'.format(
                        trms2[i][j][-2]['weight']))
                    for l in range(int(trms2[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                            trms2[i][j][l]['power'], trms2[i][j][l]['voigt']))
                    output.write('    </term>\n')
                output.write('  </coefficient>\n')
                coef_cntr += 1
        output.write('</Heff_definition>\n')
        output.close()

    def find_str_phon(self, my_term):
        self.str_phonon_coeff = []
        nterm = 0
        for i in range(len(my_term)):
            # for j in range(len(trms[i])):
            print(my_term[-1])
            nstrain = int(my_term[-1]['strain'])
            ndis = int(my_term[-1]['dips'])
            if nstrain != 0 and ndis != 0:
                self.str_phonon_coeff.append(i)

    def get_str_phonon_voigt(self, my_term, voigts=[1, 2, 3]):
        self.find_str_phon(my_term)
        nterm = 0
        my_terms = []
        for i in self.str_phonon_coeff:
            nstrain = int(my_term[-1]['strain'])
            ndis = int(my_term[-1]['dips'])
            for l in range(nstrain):
                my_voigt = int(my_term[ndis+l]['voigt'])
                if my_voigt in voigts:
                    my_terms.append(i)
                    break
        return(my_terms)

    def get_new_str_terms(self, my_term, voigts=[1, 2, 3]):
        vogt_terms = []
        my_vogt_dic = {1: 'x', 2: 'y', 3: 'z', 4: 'xx', 5: 'yy', 6: 'zz'}
        my_vogt_str = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', }
        my_voigt_temrs = self.get_str_phonon_voigt(my_term, voigts)
        # print()
        # i = my_coeff
        nstrain = int(my_term[-1]['strain'])
        ndis = int(my_term[-1]['dips'])
        my_lst = []
        for l in range(nstrain):
            my_voigt = int(my_term[ndis+l]['voigt'])
            my_power = int(my_term[ndis+l]['power'])
            if int(my_voigt) in voigts:
                my_str = (
                    {my_vogt_dic[my_voigt]: 1, my_vogt_str[my_voigt]: 1}, my_power)
                my_lst.append(my_str)
        vogt_terms.append(my_lst)
        print(10*'===')
        print(vogt_terms)
        print(10*'===')
        return(vogt_terms)

    def get_org_terms(self, my_term, voigts=[1, 2, 3]):
        org_terms = []
        my_vogt_dic = {1: 'x', 2: 'y', 3: 'z'}
        my_voigt_temrs = self.get_str_phonon_voigt(my_term, voigts)

        # if 1:
        # i = my_coeff
        nstrain = int(my_term[-1]['strain'])
        ndis = int(my_term[-1]['dips'])
        my_lst = []
        for l in range(nstrain):
            my_voigt = int(my_term[ndis+l]['voigt'])
            my_power = int(my_term[ndis+l]['power'])
            if int(my_voigt) in voigts:
                my_str = ({my_vogt_dic[my_voigt]: 1}, my_power)
                my_lst.append(my_str)
        org_terms.append(my_lst)
        return(org_terms)

    def get_mult_coeffs(self, my_str_trms):
        mult_terms = []
        for i in range(len(my_str_trms)):
            tem_dic = {}
            for j in range(len(my_str_trms[i])):
                if j == 0:
                    tem_dic = my_functions.get_pwr_N(
                        my_str_trms[i][j][0], my_str_trms[i][j][1])
                else:
                    tem_dic = my_functions.terms_mult(tem_dic, my_functions.get_pwr_N(
                        my_str_trms[i][j][0], my_str_trms[i][j][1]))
            mult_terms.append(tem_dic)
        return(mult_terms)

    def get_shifted_terms(self,my_term, my_strain=[0, 0, 0], voigts=[1, 2, 3]):
        not_shift = ['x', 'y', 'z']
        a, b, c = my_strain[0], my_strain[1], my_strain[2]
        my_mul_terms = self.get_mult_coeffs(
            self.get_new_str_terms(my_term, voigts))
        org_terms = self.get_mult_coeffs(
            self.get_org_terms(my_term, voigts))

        for i in range(len(org_terms)):
            for my_key in org_terms[i].keys():
                del my_mul_terms[i][my_key]
        shift_mult_terms = []
        for i in range(len(my_mul_terms)):
            new_dict = {}
            for my_key in my_mul_terms[i].keys():
                my_trms = my_key.split()
                my_val = my_mul_terms[i][my_key]
                new_key = ' '
                for tt in my_trms:
                    if tt in not_shift:
                        new_key = new_key + ' ' + tt
                    else:
                        if tt == 'a':
                            my_val *= a
                        elif tt == 'b':
                            my_val *= b
                        elif tt == 'c':
                            my_val *= c
                new_dict[new_key] = my_val
            shift_mult_terms.append(new_dict)
        return(shift_mult_terms)

    def get_missfit_term(self, coeff, trms, my_tags, my_strain, voigts=[1, 2, 3]):

        for my_term in trms:
            str_coeffs = self.get_str_phonon_voigt(my_term, voigts)
            print(str_coeffs)
            tot_nterms = len(trms)
            new_coeffs = []
            new_temrs = []
            for i, ii in enumerate(str_coeffs):
                # print(tot_nterms)
                my_str_phon_term = []
                for my_term in trms[ii]: 
                    # print(my_term)           
                    nstrain = int(my_term[-1]['strain'])
                    ndis = int(my_term[-1]['dips'])
                    voits_found = False
                    for l in range(nstrain):
                        my_voigt = int(my_term[ndis+l]['voigt'])
                        if int(my_voigt) in voigts:
                            voits_found = True

                    if voits_found:
                        my_terms = self.get_shifted_terms(
                            my_term, my_strain, voigts)
                        disp_text = ''
                        ndisp = int(my_term[-1]['dips'])
                        for disp in range(ndisp):
                            atm_a = int(my_term[disp]['atom_a'])
                            atm_b = int(my_term[disp]['atom_b'])
                            cell_a = [int(x) for x in my_term
                                    [disp]['cell_a'].split()]
                            cell_b = [int(x) for x in my_term
                                    [disp]['cell_b'].split()]
                            direction = my_term[disp]['direction']
                            power = my_term[disp]['power']
                            if not any(cell_b):
                                disp_text += (
                                    f'({my_tags[atm_a]}_{direction}-{my_tags[atm_b]}_{direction})^{power}')
                            else:
                                disp_text += (
                                    f'({my_tags[atm_a]}_{direction}-{my_tags[atm_b]}_{direction}[{cell_b[0]} {cell_b[1]} {cell_b[2]}])^{power}')
                        
                        # ad the term here for disp

                        term_cnre = 0
                        # print(my_term)
                        for tmp_key in my_terms[0].keys():
                            my_dis_term = copy.deepcopy(my_term[0:ndisp])
                            if len(my_str_phon_term) < len(my_terms[0].keys()):
                                my_str_phon_term.append([])
                            num_str_temrs = 0
                            str_terms = []
                            # find x
                            pwer_x = tmp_key.count('x')
                            if pwer_x != 0:
                                str_terms.append(
                                    {'power': f' {pwer_x}', 'voigt': ' 1'})
                                num_str_temrs += 1
                            # find y
                            pwer_y = tmp_key.count('y')
                            if pwer_y != 0:
                                str_terms.append(
                                    {'power': f' {pwer_y}', 'voigt': ' 2'})
                                num_str_temrs += 1
                            # find z
                            pwer_z = tmp_key.count('z')
                            if pwer_z != 0:
                                str_terms.append(
                                    {'power': f' {pwer_z}', 'voigt': ' 3'})
                                num_str_temrs += 1

                            for str_cntr in range(int(my_term[-1]['strain'])):
                                if int(my_term[ndisp+str_cntr]['voigt']) not in (voigts):
                                    str_terms.append(my_term[ndisp+str_cntr])
                                    num_str_temrs += 1


                            # for disp in range(ndisp):
                            #     my_dis_term[disp]['weight'] = float(my_dis_term[disp]['weight']) * my_terms[0][tmp_key]


                            my_str_phon_term[term_cnre].append(
                                [*my_dis_term, *str_terms, {'dips': ndisp, 'strain': num_str_temrs, 'distance': 0.0}])
                            term_cnre += 1
                        # print(my_str_phon_term[term_cnre-1])

                        if my_term == 0:
                            temp_trms = my_functions.re_order_terms(my_terms[0])
                            key_cntr = 0
                            for my_key in temp_trms.keys():
                                tot_nterms += 1
                                my_value = float(coeff[ii]['value'])*temp_trms[my_key]
                                my_key = my_key.replace('x', '(eta_1)')
                                my_key = my_key.replace('y', '(eta_2)')
                                my_key = my_key.replace('z', '(eta_3)')
                                my_text = disp_text+my_key
                                new_coeff = {'number': str(tot_nterms), 'value': str(
                                    my_value), 'text': my_text, 'org_coeff': ii}
                                new_coeffs.append(new_coeff)
                                key_cntr += 1
                
                for temp_cntr in range(len(my_str_phon_term)):
                    new_temrs.append(my_str_phon_term[temp_cntr])

        return(new_coeffs, new_temrs)

    def get_missfit_terms(self, coeff, trms, my_tags, my_strain, voigts=[1, 2, 3]):
        final_ceffs = []
        final_terms = []
        for ii in range(len(coeff)):
            Coeff_new,term_new = self.get_missfit_term(coeff[ii], trms[ii], my_tags, my_strain, voigts=voigts)
            final_ceffs.append(Coeff_new)
            final_terms.append(term_new)


    def get_strain(self, id_in):
        id_pairs = {'0':'1','1':'0'}
        tmp_SC2 = make_supercell(self.uc_atoms[id_in], self.SCMATS[id_pairs[id_in]])
        mySC2 = make_supercell(self.uc_atoms[id_in], self.SCMATS[id_in])
        SL_ref = make_SL(tmp_SC2, mySC2)
        ref_cell_pure = SL_ref.get_cell()
        int_cell = self.ref_cell.get_cell()
        inv_ref_cell = np.linalg.inv(ref_cell_pure)
        strain = (np.dot(int_cell,np.transpose(inv_ref_cell))-np.eye(3))
        voigt_str = [strain[0,0],strain[1,1],strain[2,2],(strain[1,2]+strain[2,1])/2,(strain[0,2]+strain[2,0])/2,(strain[0,1]+strain[1,0])/2]
        return(np.array(voigt_str))

class Multipol():
    def str_mult(self, a, b):
        my_list = [*a.split(), *b.split()]
        my_list.sort()
        return(' '.join(my_list))

    def terms_mult(self, T_1, T_2):
        T1T2 = {}
        for i in T_1.keys():
            for j in T_2.keys():
                my_key = self.str_mult(i, j)
                if my_key in T1T2.keys():
                    T1T2[my_key] = T1T2[my_key] + T_1[i]*T_2[j]
                else:
                    T1T2[my_key] = T_1[i]*T_2[j]
        return(T1T2)

    def get_pwr_N(self, T1, n):
        if n-1 != 0:
            return(self.terms_mult(self.get_pwr_N(T1, n-1), T1))
        else:
            return(T1)

    def re_order_terms(self, T1):
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
            New_key = [tmp_key+'^'+str(my_pwr_list[tmp_key])
                       for tmp_key in my_pwr_list.keys()]
            New_key = ' '.join(New_key)
            fin_dict[New_key] = T1[key]
        return(fin_dict)


def model_maker(xmlf1, anh_file1, scmat1, xmlf2, anh_file2, scmat2, symmetric=False, miss_fit_trms=False, har_file='int_harmoni.xml', Anhar_file='int_harmoni.xml',negelect_A_SITE=False,negelect_Tot_FCs=False,vigt_missfit=None):
    # Harmonic_term generation
    har_xml = Har_interface(xmlf1, scmat1, xmlf2, scmat2, symmetric=symmetric,negelect_A_SITE=negelect_A_SITE,negelect_Tot_FCs=negelect_Tot_FCs)
    har_xml.get_STR_FCDIC()
    har_xml.reshape_FCDIC()
    STRC = har_xml.ref_cell
    har_xml.write_xml(har_file)

    # Anhrmonic_term generation
    intf = Anh_intrface(xmlf1, anh_file1, scmat1, xmlf2, anh_file2,
                        scmat2, miss_fit_trms=miss_fit_trms, symmetric=symmetric,vigt_missfit=vigt_missfit)
    intf.wrt_anxml(Anhar_file)
    # print(intf.FC_weights)
    return(STRC)

# some simple functions:


def anh_terms_mani(har_xml, anh_xml, output='test_mani.xml', terms_to_write=None):
    xmlsys = SC_xml_potential.xml_sys(har_xml)
    xmlsys.get_ase_atoms()
    atoms = xmlsys.ase_atoms
    anhXml = SC_xml_potential.anh_scl(har_xml, anh_xml)
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


def get_mapped_strcs(str_to_be_map, str_to_map_to, Ret_index=False):
    natom = len(str_to_map_to.get_scaled_positions())
    natom2 = len(str_to_be_map.get_scaled_positions())
    if natom != natom2:
        print('wrong structures')
        return(0)
    str_cell = np.array(str_to_be_map.get_cell())
    map_index = np.zeros(natom, dtype=int)
    xred_maped = np.zeros((natom, 3))
    for ia, xred_a in enumerate(str_to_map_to.get_scaled_positions()):
        diff_xred = np.zeros((natom, 3))
        shift = np.zeros((natom, 3))
        list_dist = np.zeros(natom)
        list_absdist = np.zeros((natom, 3))
        diff_xred = str_to_be_map.get_scaled_positions()-xred_a
        for ib, b in enumerate(str_to_be_map.get_scaled_positions()):
            if diff_xred[ib, 0] > 0.5:
                diff_xred[ib, 0] = 1 - diff_xred[ib, 0]
                shift[ib, 0] = -1
            if diff_xred[ib, 1] > 0.5:
                diff_xred[ib, 1] = 1 - diff_xred[ib, 1]
                shift[ib, 1] = -1
            if diff_xred[ib, 2] > 0.5:
                diff_xred[ib, 2] = 1 - diff_xred[ib, 2]
                shift[ib, 2] = -1
            if diff_xred[ib, 0] < -0.5:
                diff_xred[ib, 0] = -1-diff_xred[ib, 0]
                shift[ib, 0] = 1
            if diff_xred[ib, 1] < -0.5:
                diff_xred[ib, 1] = -1-diff_xred[ib, 1]
                shift[ib, 1] = 1
            if diff_xred[ib, 2] < -0.5:
                diff_xred[ib, 2] = -1-diff_xred[ib, 2]
                shift[ib, 2] = 1
            list_absdist[ib, :] = np.dot(str_cell, diff_xred[ib, :])
            list_dist[ib] = np.sqrt(
                np.dot(list_absdist[ib, :], np.transpose(list_absdist[ib, :])))

        map_index[ia] = np.where(list_dist == min(list_dist))[0][0]
        xred_maped[ia, :] = str_to_be_map.get_scaled_positions(
        )[map_index[ia], :] + shift[map_index[ia], :]

    if Ret_index:
        return(map_index)

    maped_str = Atoms(numbers=str_to_map_to.get_atomic_numbers(
    ), scaled_positions=xred_maped, cell=str_to_be_map.get_cell())
    return(maped_str)


def terms_comp(trms1, trms2):
    disp = []
    strains = []
    for i in range(len(trms1)):
        for j in range(len(trms2)):
            for k in (trms2[j]):
                temp_term = trms1[i][0].copy()
                kp = k.copy()
                list_elemet = temp_term.pop(-1)
                kp.pop(-1)
                if temp_term == kp:
                    if int(list_elemet['dips']) != 0:
                        disp.append({'term_1': i, 'term_2': j})
                    else:
                        strains.append({'term_1': i, 'term_2': j})
    return(disp, strains)

# This function is used to get atomic numbber for the mass of an atoms


def get_atom_num(atomic_mass):
    for i in range(len(ase.data.atomic_masses)):
        if abs(ase.data.atomic_masses[i]-atomic_mass) < 0.1:
            mynum = i
    return(mynum)

# This functions is used when writing the xml file and convert a 2D array to text format

def to_text(arr):
    mytxt = ''
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            if i == 0 and j == 0:
                mytxt = ' {:.14E}'.format(arr[i][j])
            else:
                mytxt += '  {:.14E}'.format(arr[i][j])
        mytxt = mytxt+'\n'
    return(mytxt)

# This functions is used when writing the xml file and convert a 1D array to text format

def one_text(arr):
    mytxt = ''
    a = len(arr)
    for i in range(a):
        if i == 0:
            mytxt = ' {:.14E}'.format(arr[i])
        else:
            mytxt += '  {:.14E}'.format(arr[i])
    mytxt = mytxt+'\n'
    return(mytxt)

# This function is used to find the position and index of an atom in a structure (MCOR = positins of the atoms in the structure, Vec = position of the atoms to find its index )

def find_index_xred(Mcor, Vec, tol=0.001):
    index = -1
    for m in range(len(Mcor)):
        # print(m)
        diff0 = np.array(Mcor[m])-np.array(Vec)
        # print('0',diff)
        diff = [(1 - i) if i > 0.5 else i for i in diff0]
        # print('1',diff)
        diff = [(-1 - i) if i < -0.5 else i for i in diff]
        # print('2',diff)
        bol_flgs = [True if abs(i) < tol else False for i in diff]
        if all(bol_flgs):
            index = m
    # if index==-1:
    #     index = find_index_xred(Mcor,np.array(Vec)-(np.array(diff)-np.array(diff0)))
    return(index)

# This function is used to make superlattice of two structures as Atoms objec

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

def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def find_index(Mcor, Vec, tol=0.001):
    index = -1
    for m in range(len(Mcor)):
        flg = []
        for v in range(len(Mcor[m])):
            diff = Mcor[m, v]-Vec[v]
            if abs(diff) < tol:
                flg.append(True)
            else:
                flg.append(False)
        if all(flg):
            index = m
    return(index)

###########################################################
if __name__ == '__main__':
    import time
    from interface_xmls import *
    import P_interface_xmls
    path_0 = os.getcwd()

    xmlf1 = f'{path_0}/src_xmls/Har_xml_8812'
    anh_file10 = f'{path_0}/src_xmls/AnHar_xml_8812'

    # xmlf2 = f'{path_0}/src_xmls/Har_xml_1048'
    # anh_file20 = f'{path_0}/src_xmls/AnHar_xml_1048'

    xmlf2 = f'{path_0}/src_xmls/Har_xml_8812'
    anh_file20 = f'{path_0}/src_xmls/AnHar_xml_8812'

    # xmlf2 = f'{path_0}/src_xmls/trans_Har_xml_1048'
    # anh_file20 = f'{path_0}/src_xmls/trans_AnHar_xml_1048'

    #term = 5
    symmetric = False
    Temp = 10
    for term in [0]:  # range(10):
        # anh_file1 = f'{path_0}/xmls_files/mani1_{term}.xml'
        # anh_terms_mani(xmlf1,anh_file10,output=anh_file1,terms_to_write=[term])

        # anh_file2 = f'{path_0}/xmls_files/mani2_{term}.xml'
        # anh_terms_mani(xmlf2,anh_file20,output=anh_file2,terms_to_write=[term])

        anh_file1 = anh_file10
        anh_file2 = anh_file20

        dim_a = 1
        dim_b = 1
        scmat = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        scmat2 = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        dim_z = scmat[2][2]+scmat2[2][2]

        # ts = time.perf_counter()
        har_xml = Har_interface(xmlf2, scmat, xmlf2,
                                scmat2, symmetric=symmetric)
        # har_xml.get_STR_FCDIC()
        # har_xml.reshape_FCDIC()
        # har_xml.write_xml(SL_xml)
        # tf = time.perf_counter()
        # print('TIME = ',tf-ts)
        UC_1 = har_xml.uc_atoms['0']

        str_ten = [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]]
        phon_scll = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        SIM_cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        distance = 0.01
        phonon_disp = my_functions.get_phonon(f'{xmlf1}', anh_file1, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                              8, 8, 8], UC=UC_1, my_EXEC='MB_16Jun', SIM_cell=SIM_cell, path=f'{path_0}/ref', disp_amp=distance)
        os.chdir(path_0)
        phonon_disp.run_random_displacements(Temp)
        rnd_disp = phonon_disp.random_displacements.u
        new_scld_pos = UC_1.get_scaled_positions()+rnd_disp
        new_UC = Atoms(numbers=UC_1.get_atomic_numbers(),
                       scaled_positions=new_scld_pos[0], cell=UC_1.get_cell())

        new_UC = UC_1

        SL_xml = f'{path_0}/xmls_files/N_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        anh_int = f'{path_0}/xmls_files/{term}_test_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        model_maker(xmlf1, anh_file1, scmat, xmlf2, anh_file2, scmat2,
                    symmetric=symmetric, har_file=SL_xml, Anhar_file=anh_int)

        str_ten = [[0.10, 0, 0], [0, 0, 0], [0, 0, 0]]
        distance = 0.11
        phon_scll = [[2, 0, 0], [0, 2, 0], [0, 0, dim_z]]
        SIM_cell = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        # xml_old or anh_int

        phonon = my_functions.get_phonon(SL_xml, anh_int, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                         8, 8, 8], UC=new_UC, my_EXEC='MB_16Jun', SIM_cell=SIM_cell, path=f'{path_0}/intf', disp_amp=distance)
        os.chdir(path_0)
        my_functions.plot_phonon(
            phonon, name=f'0_{term}_N_plt_{dim_z}', cpath='./')

        pordc_str_trms = False
        SL_xml_old = f'{path_0}/xmls_files/Old_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        anh_int_old = f'{path_0}/xmls_files/Old_{term}_test_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'

        mdl = P_interface_xmls.har_interface(xmlf1, scmat, xmlf2, scmat2,)
        mdl.reshape_FCDIC()
        mdl.write_xml(SL_xml_old)
        my_intr = P_interface_xmls.anh_intrface(
            xmlf1, anh_file1, scmat, xmlf2, anh_file2, scmat2, pordc_str_trms)
        my_intr.wrt_anxml(anh_int_old)

        #anh_int_old = 'no'

        phonon_n = my_functions.get_phonon(SL_xml_old, anh_int_old, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                           8, 8, 8], UC=new_UC, my_EXEC='MB_16Jun', SIM_cell=SIM_cell, path=f'{path_0}/intf', disp_amp=distance)
        os.chdir(path_0)
        my_functions.plot_phonon(
            phonon_n, name=f'0_{term}_old_plt_{dim_z}', cpath='./')

        # phon_scll = [[2,0,0],[0,2,0],[0,0,dim_z]]
        # SIM_cell =  [[2,0,0],[0,2,0],[0,0,dim_z]]
        # phonon_ref = my_functions.get_phonon(f'{xmlf2}', f'{anh_file}', phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[8,8,8], UC=new_UC ,my_EXEC='MB_Mar20', SIM_cell=SIM_cell,path=f'{path_0}/ref',distance=distance)
        # os.chdir(path_0)
        # my_functions.plot_phonon(phonon_ref,name=f'0_{term}_ref_plt_{dim_z}',cpath='./')

#     anh_file = "AnHar_xml_888.xml"
#     oloc = old_har_xml.loc_SC_FCDIC
#     loc = har_xml.loc_SL_FCDIC
#     cntr = 0
#     for i,j in zip(oloc.keys(),loc.keys()):
#         cntr += 1
#         #print(i,'  ',j)
#         diffs = oloc[i]-loc[j]
#         if any(if_all_zeros(diffs)):
#             print(i,j)
#             print(10*'--')
#             print(oloc[i])
#             print(10*'--')
#             print(loc[j])
#             print(10*'--')
#         if cntr==10:
#             break

# def if_all_zeros(matr):
#     d1 = len(matr)
#     d2 = len(matr[0])
#     log_m = []
#     for i in range(d1):
#         #log_m.append([])
#         for j in range(d2):
#             flg = False
#             if matr[i,j]>1.0*10**-20:
#                 flg = True
#             log_m.append(flg)
#     return(log_m)

#     from interface_xmls import *

#     xmlf1 = 'PTO_Har_xml_998.xml'
#     dim_a = 1
#     dim_b = 2
#     scmat = [[dim_a,0,0],[0,dim_b,0],[0,0,1]]

#     #anh_file = "AnHar_xml_888.xml"
#     anh_file = 'S_trm.xml'

#     intf = Anh_intrface(xmlf1,anh_file,scmat,xmlf1,anh_file,scmat,pordc_str_trms=False,symmetric=False)
#     intf.SC_trms(id_in='0')
#     intf.SC_trms(id_in='1')
#     intf.wrt_anxml(f'test_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml')
#     write('POSCAR',intf.STRC,format='vasp')
