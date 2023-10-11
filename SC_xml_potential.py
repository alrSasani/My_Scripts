import numpy as np
import xml.etree.ElementTree as ET
from math import ceil
from ase import Atoms
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses
from ase.build import make_supercell
from ase.io import write
import netCDF4 as nc
import ase
import time
from ase.io import write, read
import spglib as spg
import numpy.linalg as alg
from itertools import permutations
import copy

###############################################################################

# Harmonic xml reader:


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


################################################################################

class my_sc_maker():
    def __init__(self, xml_file, SC_mat,strain=np.zeros(3)):
        self.xml = xml_sys(xml_file)
        self.xml.get_ase_atoms()
        self.my_atoms = self.xml.ase_atoms
        self.SC_mat = SC_mat
        self.set_SC(self.my_atoms,strain=strain)
        self.mySC = make_supercell(self.my_atoms, self.SC_mat)
        self.SC_natom = self.mySC.get_global_number_of_atoms()
        self.SC_num_Uclls = np.linalg.det(SC_mat)
        self.has_SC_FCDIC = False
        self.has_FIN_FCDIC = False
        self.xml.get_loc_FC_dic()
        self.xml.get_tot_FC_dic()
        self.xml.get_Per_clls()
        self.has_tot_FC = self.xml.has_tot_FC
        self.tot_FC_dic = self.xml.tot_FC_dic
        self.loc_FC_dic = self.xml.loc_FC_dic
        self.xml.get_atoms()
        self.atm_pos = self.xml.atm_pos
        self.UC_natom = 5

    def set_tot_UCFC(self, temp_tot_FCdic, flag=True):
        if flag:
            self.tot_FC_dic = temp_tot_FCdic
            self.has_tot_FC = flag
        else:
            self.has_tot_FC = flag

    def set_loc_UCFC(self, temp_loc_FCdic):
        self.loc_FC_dic = temp_loc_FCdic

    def set_SC(self, tmp_atoms,strain=np.zeros(3)):
        mySC = make_supercell(tmp_atoms, self.SC_mat)
        cell = mySC.get_cell()+np.dor(strain,mySC.get_cell())
        self.mySC = Atoms(numbers= mySC.get_atomic_numbers,scaled_positions=mySC.get_scaled_positions,cell = cell)

    def get_SC_FCDIC(self):
        CPOS = self.mySC.get_positions()
        abc = self.my_atoms.cell.cellpar()[0:3]
        ABC = self.mySC.cell.cellpar()[0:3]
        ncll_loc = self.xml.ncll
        if self.has_tot_FC:
            self.xml.get_tot_Per_clls()
            ncll_tot = self.xml.tot_ncll
        else:
            ncll_tot = [1, 1, 1]
        px, py, pz = max((ncll_loc[0]-1)/2, (ncll_tot[0]-1)/2), max(
            (ncll_loc[1]-1)/2, (ncll_tot[1]-1)/2), max((ncll_loc[2]-1)/2, (ncll_tot[2]-1)/2)
        xperiod = ceil(px/self.SC_mat[0][0])
        yperiod = ceil((py)/self.SC_mat[1][1])
        zperiod = ceil((pz)/self.SC_mat[2][2])
        UC_natm = self.my_atoms.get_global_number_of_atoms()
        self.loc_SC_FCDIC = {}
        self.tot_SC_FCDIC = {}
        self.tot_mykeys = []
        self.loc_mykeys = []
        for prd1 in range(-int(xperiod), int(xperiod)+1):
            for prd2 in range(-int(yperiod), int(yperiod)+1):
                for prd3 in range(-int(zperiod), int(zperiod)+1):
                    SC_cell = '{} {} {}'.format(prd1, prd2, prd3)
                    for atm_i in range(self.SC_natom):
                        for atm_j in range(self.SC_natom):
                            dist = (prd1*ABC[0]+CPOS[int(atm_j/UC_natm)*self.UC_natom][0]-CPOS[int(atm_i/UC_natm)*self.UC_natom][0],
                                    prd2*ABC[1]+CPOS[int(atm_j/UC_natm)*self.UC_natom][1] -
                                    CPOS[int(atm_i/UC_natm)*self.UC_natom][1],
                                    prd3*ABC[2]+CPOS[int(atm_j/UC_natm)*self.UC_natom][2]-CPOS[int(atm_i/UC_natm)*self.UC_natom][2])
                            cell_b = (int(
                                dist[0]/(abc[0]*0.95)), int(dist[1]/(abc[1]*0.95)), int(dist[2]/(abc[2]*0.95)))
                            UC_key = '{}_{}_{}_{}_{}'.format(
                                atm_i % UC_natm, atm_j % UC_natm, cell_b[0], cell_b[1], cell_b[2])
                            SC_key = '{}_{}_{}_{}_{}'.format(
                                atm_i, atm_j, prd1, prd2, prd3)

                            if UC_key in self.loc_FC_dic.keys():
                                self.loc_SC_FCDIC[SC_key] = self.loc_FC_dic[UC_key]
                                if SC_cell not in (self.loc_mykeys):
                                    self.loc_mykeys.append(SC_cell)

                            if self.has_tot_FC and UC_key in self.tot_FC_dic.keys():
                                self.tot_SC_FCDIC[SC_key] = self.tot_FC_dic[UC_key]
                                if SC_cell not in (self.tot_mykeys):
                                    self.tot_mykeys.append(SC_cell)
        self.has_SC_FCDIC = True

    def reshape_FCDIC(self, tmp_sc=0):
        if not self.has_SC_FCDIC:
            self.get_SC_FCDIC()
        self.Fin_loc_FC = {}
        if self.has_tot_FC:
            self.Fin_tot_FC = {}
            my_keys = self.tot_mykeys
        else:
            my_keys = self.loc_mykeys
        if tmp_sc:
            self.my_atm_list = self.mapping(tmp_sc)
            # self.mySC=make_supercell(tmp_sc,self.SC_mat)
        else:
            self.my_atm_list = range(self.SC_natom)
        for my_key in (my_keys):
            my_cell = [int(x) for x in my_key.split()]
            loc_key_found = False
            tot_key_found = False
            tmp_loc_FC = np.zeros((3*self.SC_natom, 3*self.SC_natom))
            tmp_tot_FC = np.zeros((3*self.SC_natom, 3*self.SC_natom))
            cnt_a = 0
            for atm_a in self.my_atm_list:
                cnt_b = 0
                for atm_b in self.my_atm_list:
                    my_index = '{}_{}_{}_{}_{}'.format(
                        atm_a, atm_b, my_cell[0], my_cell[1], my_cell[2])
                    if my_index in self.loc_SC_FCDIC.keys():
                        loc_key_found = True
                        tmp_loc_FC[cnt_a*3:cnt_a*3+3, cnt_b *
                                   3:cnt_b*3+3] = self.loc_SC_FCDIC[my_index]
                    if my_index in self.tot_SC_FCDIC.keys() and self.has_tot_FC:
                        tot_key_found = True
                        tmp_tot_FC[cnt_a*3:cnt_a*3+3, cnt_b *
                                   3:cnt_b*3+3] = self.tot_SC_FCDIC[my_index]
                    cnt_b += 1
                cnt_a += 1
            if loc_key_found:
                self.Fin_loc_FC[my_key] = tmp_loc_FC
            if tot_key_found:
                self.Fin_tot_FC[my_key] = tmp_tot_FC

        self.has_FIN_FCDIC = True

    def asr_impose(self):
        #print('ASR imposing')
        if self.has_tot_FC:
            for atm_i in range(len(self.my_atm_list)):
                # asr_sr=np.zeros((3,3))
                asr_tot = np.zeros((3, 3))
                for atm_j in range(len(self.my_atm_list)):
                    for key_j in self.Fin_tot_FC.keys():
                        # asr_sr+=self.Fin_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                        # if self.has_tot_FC:
                        asr_tot += self.Fin_tot_FC[key_j][3 *
                                                          atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]

                self.Fin_tot_FC['0 0 0'][3*atm_i:3 *
                                         atm_i+3, 3*atm_i:3*atm_i+3] -= asr_tot

    def asr_chk(self):
        print('ASR chking')
        if self.has_tot_FC:
            for atm_i in range(len(self.my_atm_list)):
                asr_tot = np.zeros((3, 3))
                for atm_j in range(len(self.my_atm_list)):
                    for key_j in self.Fin_tot_FC.keys():
                        asr_tot += self.Fin_tot_FC[key_j][3 *
                                                          atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]
                asr_tot += self.Fin_tot_FC[key_j][3 *
                                                  atm_i:3*atm_i+3, 3*atm_j:3*atm_j+3]
                print(asr_tot)

    def mapping(self, tmp_scll):
        mscp = self.mySC.get_scaled_positions()
        tmp_scp = tmp_scll.get_scaled_positions()
        # for i in range(len(tmp_scp)):
        mstm_lst = []
        for i in range(len(tmp_scp)):
            ind = self.find_index(mscp, tmp_scp[i])
            if ind != -1:
                mstm_lst.append(ind)
        return(mstm_lst)

    def mapping_new(self, str_to_be_map, Ret_index=True):
        str_to_map_to = self.mySC.get_scaled_positions()
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

    def find_index(self, Mcor, Vec, tol=0.001):
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

    def set_UC_BEC(self, tem_BEC):
        tmp_UC_BEC = {}
        for i in range(len(tem_BEC)):
            brn_tmp = [float(j) for j in tem_BEC[i]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            tmp_UC_BEC[i] = brn_tmp
        self.UC_BEC = tmp_UC_BEC

    def get_UC_BEC(self):
        if len(self.my_atm_list) > self.UC_natom:
            my_list = range(len(self.atm_pos))
        else:
            my_list = self.my_atm_list

        self.UC_BEC = {}
        for i, ii in enumerate(my_list):
            brn_tmp = [float(j) for j in self.atm_pos[ii][2].split()[:]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            self.UC_BEC[i] = brn_tmp

    def get_SC_BEC(self):
        self.get_UC_BEC()
        self.SC_BEC = {}
        for i, ii in enumerate(self.my_atm_list):

            brn_tmp = self.UC_BEC[ii % self.UC_natom]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            self.SC_BEC[i] = brn_tmp

    def get_SC_corr_forc(self):
        self.xml.get_str_cp()
        self.SC_corr_forc = {}
        natm = self.SC_natom
        for k in range(len(self.xml.corr_forc)):
            lst = []
            for j in self.my_atm_list:
                lst.append(self.xml.corr_forc[k][j % self.UC_natom, :])
            np.reshape(lst, (natm, 3))
            self.SC_corr_forc[k] = np.array(lst)

    def write_xml(self, out_put, asr=0):
        if not self.has_FIN_FCDIC:
            self.reshape_FCDIC()
        self.xml.get_str_cp()
        ncll = self.xml.ncll
        self.xml.get_eps_inf()
        self.xml.get_ela_cons()
        self.xml.get_ref_energy()
        SC_FC = self.Fin_loc_FC
        self.get_SC_BEC()
        self.xml.get_tot_forces()
        if self.has_tot_FC:
            tSC_FC = self.Fin_tot_FC
            keys = tSC_FC.keys()
        else:
            keys = SC_FC.keys()
        lt_scll = self.mySC.get_cell()/Bohr
        atm_pos_scll = self.mySC.get_positions()/Bohr
        natm = self.SC_natom
        self.get_SC_corr_forc()
        SC_CorForc = self.SC_corr_forc
        if asr:
            self.asr_impose()
            self.asr_chk()
        SCL_elas = ((self.xml.ela_cons)*self.SC_num_Uclls)
        out_xml = open(out_put, 'w')
        out_xml.write('<?xml version="1.0" ?>\n')
        out_xml.write('<System_definition>\n')
        out_xml.write('  <energy>\n  {:.14E}\n  </energy>\n'.format(
            self.SC_num_Uclls*self.xml.ref_eng))  # multiply
        out_xml.write(
            '  <unit_cell units="bohrradius">\n {}  </unit_cell>\n'.format(to_text(lt_scll)))  # multiply
        out_xml.write(
            '  <epsilon_inf units="epsilon0">\n  {}  </epsilon_inf>\n'.format(to_text(self.xml.eps_inf)))
        out_xml.write(
            '  <elastic units="hartree">\n  {}  </elastic>\n'.format(to_text(SCL_elas)))
        for i, ii in enumerate(self.my_atm_list):
            out_xml.write('  <atom mass="  {}" massunits="atomicmassunit">\n    <position units="bohrradius">\n   {}</position>\n    <borncharge units="abs(e)">\n  {}</borncharge>\n  </atom>\n'.format(
                self.atm_pos[ii % self.UC_natom][0], one_text(atm_pos_scll[ii, :]), to_text(self.SC_BEC[ii])))

        for key in keys:
            if key in (SC_FC.keys()):
                out_xml.write(
                    '  <local_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </local_force_constant>\n'.format(to_text((SC_FC[key])), key))
            if self.has_tot_FC:
                out_xml.write(
                    '  <total_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </total_force_constant>\n'.format(to_text((tSC_FC[key])), key))

       # xml.write('  <phonon>\n    <qpoint units="2pi*G0">  {}</qpoint>\n    <frequencies units="reciprocal cm">\n  {}    </frequencies>\n    <dynamical_matrix units="hartree/bohrradius**2">\n {}    </dynamical_matrix>\n   </phonon>\n'.format(self.xml.qpoint,to_text(SC_phon),to_text(SC_dmat)))
        for i in range(len(self.xml.strain)):
            out_xml.write('  <strain_coupling voigt=" {}">\n    <strain>  {}    </strain>\n    <correction_force units="hartree/bohrradius">\n  {}    </correction_force>\n  </strain_coupling>\n'.format(
                i, (self.xml.strain[i]), to_text(SC_CorForc[i])))
        out_xml.write('</System_definition>')
        out_xml.close()

################################################################################

# Anharmonic Potential Creation::


def xml_anha(fname, atoms):
    tree = ET.parse(fname)
    root = tree.getroot()
    coeff = {}
    lst_trm = []
    car_pos = atoms.get_positions()
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
                cell_a_flg = [True if l != 0 else False for l in cell_a]
                # print(abc[0])
                if any(cell_a_flg):
                    cell_a -= cell_a
                    cell_b = cell_b-cell_a
                    cella = f'{cell_a[0]} {cell_a[1]} {cell_a[2]}'
                    cellb = f'{cell_b[0]} {cell_b[1]} {cell_b[2]}'

                pos_a = [cell_a[0]*abc[0]+car_pos[atma, 0], cell_a[1] *
                         abc[1]+car_pos[atma, 1], cell_a[2]*abc[2]+car_pos[atma, 2]]
                pos_b = [cell_b[0]*abc[0]+car_pos[atmb, 0], cell_b[1] *
                         abc[1]+car_pos[atmb, 1], cell_b[2]*abc[2]+car_pos[atmb, 2]]
                dist = ((pos_a[0]-pos_b[0])**2+(pos_a[1]-pos_b[1])
                        ** 2+(pos_a[2]-pos_b[2])**2)**0.5
                dta['cell_a'] = cella
                dta['cell_b'] = cellb
                dta['weight'] = trm.attrib['weight']
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

################################################################################


class anh_scl():

    def __init__(self, har_xml, anh_xml,strain_in=np.zeros(3)):
        self.xml = har_xml
        self.ahxml = anh_xml
        strain_vogt = self.get_strain(strain=strain_in)
        if any(strain_vogt)>0.0001:
            self.has_strain = True
            self.voigt_strain = strain_vogt
        else:
            self.has_strain = False
            self.voigt_strain = [0,0,0,0,0,0]
            
    def SC_trms(self, MySC, SC_mat):
        myxml_clss = xml_sys(self.xml)
        myxml_clss.get_ase_atoms()
        
        my_atoms = myxml_clss.ase_atoms
        coeff, trms = xml_anha(self.ahxml, my_atoms)
        self.SC_mat = SC_mat
        mySC = MySC
###########################        
        total_coefs = len(coeff)
        tol_04 = 0.0001
        if self.miss_fit_trms:

            print(f'The strain for material id  = ',self.voigt_strain)
            temp_voits = []
            strain_flag = []
            stain_flag_inp = []
            for ii,i in enumerate(self.voigt_strain):
                if abs(i) >= tol_04:
                    strain_flag.append(True)
                    temp_voits.append(ii+1)
                    if self.voigt_missfit is not None and ii+1 in self.voigt_missfit:
                        stain_flag_inp.append(True)
                else:
                    strain_flag.append(False)
                    if self.voigt_missfit is not None and ii+1 not in self.voigt_missfit:
                        stain_flag_inp.append(False)                               
            if self.voigt_missfit is not None:
                temp_voits = self.voigt_missfit
                strain_flag = stain_flag_inp
                
            # print(f' The missfit strains material {id_in} are in directions : ',10*'***',temp_voits, 'in direction ', stain_flag_inp)
            if any(strain_flag):           
                my_tags = myxml_clss.tags
                new_coeffs, new_trms = self.get_missfit_terms(
                    coeff, trms, my_tags, self.voigt_strain, voigts=temp_voits)
                for ntrm_cntr in range(len(new_coeffs)):
                    trms.append(new_trms[ntrm_cntr])
                    coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]
                print(f'number of Missfit Coeffiecinets for material is {len(new_coeffs)}')

                # total_coefs = len(coeff)
                # new_coeffs, new_trms = self.get_elas_missfit(self.voigt_strain,voigts=temp_voits)
                # for ntrm_cntr in range(len(new_coeffs)):
                #     trms.append(new_trms[ntrm_cntr])
                #     coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]

####################################                        
        CPOS = mySC.get_positions()
        ncell = np.linalg.det(self.SC_mat)        
        # XPOS=mySC.get_scaled_positions()
        ABC = mySC.cell.cellpar()[0:3]
        cPOS = my_atoms.get_positions()
        abc = my_atoms.cell.cellpar()[0:3]
        # myHa_SC=my_sc_maker(self.xml,self.SC_mat)
        my_terms = []
        for cc in range(len(coeff)):
            my_terms.append([])
            for tc in range(len(trms[cc])):
                if 1:
                    for prd1 in range((self.SC_mat[0][0])):
                        for prd2 in range((self.SC_mat[1][1])):
                            for prd3 in range((self.SC_mat[2][2])):
                                # for prd1p in range(self.SC_mat[0][0]):
                                # for prd2p in range(self.SC_mat[1][1]):
                                # for prd3p in range(self.SC_mat[2][2]):
                                my_term = []
                                disp_cnt = 0
                                for disp in range(int(trms[cc][tc][-1]['dips'])):
                                    atm_a = int(trms[cc][tc][disp]['atom_a'])
                                    atm_b = int(trms[cc][tc][disp]['atom_b'])
                                    cell_a0 = [
                                        int(x) for x in trms[cc][tc][disp]['cell_a'].split()]
                                    cell_b0 = [
                                        int(x) for x in trms[cc][tc][disp]['cell_b'].split()]
                                    catm_a0 = cell_a0[0]*abc[0]+cPOS[atm_a][0], cell_a0[1] * \
                                        abc[1]+cPOS[atm_a][1], cell_a0[2] * \
                                        abc[2]+cPOS[atm_a][2]
                                    catm_b0 = cell_b0[0]*abc[0]+cPOS[atm_b][0], cell_b0[1] * \
                                        abc[1]+cPOS[atm_b][1], cell_b0[2] * \
                                        abc[2]+cPOS[atm_b][2]
                                    #dst0 = distance.euclidean(catm_a0, catm_b0)
                                    dst0 = [catm_a0[0]-catm_b0[0], catm_a0[1] -
                                            catm_b0[1], catm_a0[2]-catm_b0[2]]
                                    catm_an = prd1 * \
                                        abc[0]+catm_a0[0], prd2*abc[1] + \
                                        catm_a0[1], prd3*abc[2]+catm_a0[2]
                                    catm_bn = prd1 * \
                                        abc[0]+catm_b0[0], prd2*abc[1] + \
                                        catm_b0[1], prd3*abc[2]+catm_b0[2]
                                    ind_an = find_index(CPOS, catm_an)
                                    ind_bn = find_index(CPOS, catm_bn)
                                    #dst = distance.euclidean(catm_an, catm_bn)
                                    dst = [catm_an[0]-catm_bn[0], catm_an[1] -
                                           catm_bn[1], catm_an[2]-catm_bn[2]]

                                    dif_ds = np.zeros((3))

                                    # dif_ds=[abs(dst[i]-dst0[i] for i in range(3)]
                                    for i in range(3):
                                        dif_ds[i] = abs(dst[i]-dst0[i])
                                    if (ind_an != -1) and all(dif_ds < 0.001):
                                        if ind_bn == -1:
                                            #cell_b='{} {} {}'.format(cell_b0[0],cell_b0[1],cell_b0[2])
                                            # disp_pos=catm_bn[0]-(cell_b0[0])*ABC[0],catm_bn[1]-(cell_b0[1])*ABC[1],catm_bn[2]-(cell_b0[2])*ABC[2]
                                            red_pos = catm_bn[0]/ABC[0], catm_bn[1] / \
                                                ABC[1], catm_bn[2]/ABC[2]
                                            tmp_par = np.zeros((3))
                                            for i, ii in enumerate(red_pos):
                                                if ii < 0:
                                                    tmp_par[i] = 1
                                            cell_b = '{} {} {}'.format(int(int(red_pos[0])-tmp_par[0]), int(
                                                int(red_pos[1])-tmp_par[1]), int(int(red_pos[2])-tmp_par[2]))
                                            disp_pos = (red_pos[0]-(int(red_pos[0])-tmp_par[0]))*ABC[0], (red_pos[1]-(int(
                                                red_pos[1])-tmp_par[1]))*ABC[1], (red_pos[2]-(int(red_pos[2])-tmp_par[2]))*ABC[2]
                                            ind_bn = find_index(CPOS, disp_pos)
                                        else:
                                            cell_b = '0 0 0'
                                        new_dis = {'atom_a': ind_an, 'cell_a': '0 0 0', 'atom_b': ind_bn, 'cell_b': cell_b, 'direction': trms[cc][
                                            tc][disp]['direction'], 'power': trms[cc][tc][disp]['power'], 'weight': trms[cc][tc][disp]['weight']}
                                        my_term.append(new_dis)
                                    disp_cnt += 1
                                if (int(trms[cc][tc][-1]['dips']) == 0 or (disp_cnt == int(trms[cc][tc][-1]['dips']) and (len(my_term) != 0))):
                                    tmp_d = 0
                                    if disp_cnt == 0:
                                        tmp_d = 1

                                    for str_cnt in range(int(trms[cc][tc][-1]['strain'])):
                                        my_term.append(
                                            {'power': trms[cc][tc][disp_cnt+tmp_d+str_cnt]['power'], 'voigt': trms[cc][tc][disp_cnt+tmp_d+str_cnt]['voigt']})

                                if len(my_term) == int(trms[cc][tc][-1]['dips'])+int(trms[cc][tc][-1]['strain']):

                                    if (int(trms[cc][tc][-1]['dips']) == 0 and int(trms[cc][tc][-1]['strain']) != 0):
                                        my_term.append(
                                            {'weight': trms[cc][tc][0]['weight']})

                                    my_term.append(trms[cc][tc][-1])
                                    if my_term not in (my_terms[cc]):
                                        my_terms[cc].append(my_term)

            if len(trms[cc])>=1:
                if (int(trms[cc][0][-1]['dips']) == 0) and (int(trms[cc][0][-1]['strain'])!=0):
                    coeff[cc]['value'] = ncell*float(coeff[cc]['value'])

        self.SC_terms = my_terms
        self.SC_coeff = coeff

    def wrt_anxml(self, fout):
        coeff = self.SC_coeff
        trms = self.SC_terms
        output = open(fout, 'w')
        output.write('<?xml version="1.0" ?>\n')
        output.write('<Heff_definition>\n')
        for i, key in enumerate(coeff.keys()):
            output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                coeff[key]['number'], coeff[key]['value'], coeff[key]['text'],))
            for j in range(len(trms[i])):
                for k in range(int(trms[i][j][-1]['dips'])):
                    if (k == 0):
                        output.write('    <term weight="{}">\n'.format(
                            trms[i][j][k]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(
                        trms[i][j][k]['atom_a'], trms[i][j][k]['atom_b'], trms[i][j][k]['direction'], trms[i][j][k]['power']))
                    output.write(
                        '        <cell_a>{}</cell_a>\n'.format(trms[i][j][k]['cell_a']))
                    output.write(
                        '        <cell_b>{}</cell_b>\n'.format(trms[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')
                if int(trms[i][j][-1]['dips']) == 0:
                    k = int(trms[i][j][-1]['strain'])
                    output.write('    <term weight="{}">\n'.format(
                        trms[i][j][k]['weight']))
                    k = -1
                for l in range(int(trms[i][j][-1]['strain'])):
                    output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                        trms[i][j][k+l+1]['power'], trms[i][j][k+l+1]['voigt']))
                output.write('    </term>\n')
            output.write('  </coefficient>\n')
        output.write('</Heff_definition>\n')

    def find_str_phonon_coeffs(self, trms):
        '''this function returns all the terms with strain phonon couplings'''
        str_phonon_coeffs = []
        nterm = 0
        for i in range(len(trms)):
            # for j in range(len(trms[i])):
            nstrain = int(trms[i][nterm][-1]['strain'])
            ndis = int(trms[i][nterm][-1]['dips'])
            if nstrain != 0 and ndis == 0:
                str_phonon_coeffs.append(i)
        return(str_phonon_coeffs)

    def get_str_phonon_voigt(self, trms, voigts=[1, 2, 3]):
        '''This function returns the number of coefficients that have particulat strain phonon coupling'''
        str_phonon_coeffs = self.find_str_phonon_coeffs(trms)
        str_phonon_voigt = []

        for i in str_phonon_coeffs:
            voigt_found = False
            for nterm in range(len(trms[i])):
                if not voigt_found:
                    nstrain = int(trms[i][nterm][-1]['strain'])
                    ndis = int(trms[i][nterm][-1]['dips'])
                    if ndis == 0:
                        ndis = 1
                    for l in range(nstrain):
                        # print(trms[i][nterm][ndis+l])
                        my_voigt = int(trms[i][nterm][ndis+l]['voigt'])
                        if my_voigt in voigts:
                            str_phonon_voigt.append(i)
                            voigt_found = True
                            break
        # print(10*'*******')
        return(str_phonon_voigt)

    def get_new_str_terms(self, term,get_org=False):
        '''This function changes the term to a dictionary so that we can expand and multiply like polynomials
        it returns a list like : [[({'z': 1, 'c': 1}, 2)]] where we have in the dictionary different voigt strains 
        (x,y,z,xy ..) and their values as a,b,c,d ... and the power of the strain as the last element of the list '''
        vogt_terms = []
        my_vogt_dic = {1: 'x', 2: 'y', 3: 'z', 4: 'yz', 5: 'xz', 6: 'xy'}
        my_vogt_str = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', }

        nstrain = int(term[-1]['strain'])
        ndis = int(term[-1]['dips'])
        if ndis == 0:
            ndis = 1
        my_lst = []
        for l in range(nstrain):
            my_voigt = int(term[ndis+l]['voigt'])
            my_power = int(term[ndis+l]['power'])
            if get_org:
                my_str = ({my_vogt_dic[my_voigt]: 1}, my_power)
            else:
                my_str = (
                    {my_vogt_dic[my_voigt]: 1, my_vogt_str[my_voigt]: 1}, my_power)
            my_lst.append(my_str)
        vogt_terms.append(my_lst)
        return(vogt_terms)

    def get_mult_coeffs(self, my_str_trms):
        mult_terms = []
        for i in range(len(my_str_trms)):
            tem_dic = {}
            for j in range(len(my_str_trms[i])):
                if j == 0:
                    tem_dic = get_pwr_N(
                        my_str_trms[i][j][0], my_str_trms[i][j][1])
                else:
                    tem_dic = terms_mult(tem_dic, get_pwr_N(
                        my_str_trms[i][j][0], my_str_trms[i][j][1]))
            mult_terms.append(tem_dic)
        return(mult_terms)

    def get_shifted_terms(self, term, my_strain=[0, 0, 0]):
        not_shift = ['x', 'y', 'z']
        a, b, c = my_strain[0], my_strain[1], my_strain[2]
        my_mul_terms = self.get_mult_coeffs(
            self.get_new_str_terms(term))
        org_terms = self.get_mult_coeffs(
            self.get_new_str_terms(term,get_org=True))
        # print(10*'---')
        # print(org_terms)
        # print(my_mul_terms)
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
        tot_nterms = 0 
        new_coeffs = []
        new_temrs = []
        my_str_phon_term = []
        for i_term,my_term in enumerate(trms): 
            no_disp = False
            nstrain = int(my_term[-1]['strain'])
            ndisp = int(my_term[-1]['dips'])
            if ndisp == 0:
                ndisp = 1
            voits_found = False
            for l in range(nstrain):
                my_voigt = int(my_term[ndisp+l]['voigt'])
                if int(my_voigt) in voigts:
                    voits_found = True
            if voits_found:
                my_terms = self.get_shifted_terms(
                    my_term, my_strain)
                ndisp = int(my_term[-1]['dips'])
                # print(my_terms)
                if ndisp>0 :
                    disp_text = self.get_disp_text(my_term,my_tags)
                else:
                    disp_text = ''
                    no_disp = True
                    ndisp = 1                
                term_cnre = 0
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
                    for disp in range(ndisp):
                            my_dis_term[disp]['weight'] = float(my_dis_term[disp]['weight']) * my_terms[0][tmp_key]
                    if no_disp==False or num_str_temrs > 0:
                        if no_disp:
                            temp_ndisp = 0
                        else:
                            temp_ndisp = ndisp
                        my_str_phon_term[term_cnre].append(
                            [*my_dis_term, *str_terms, {'dips': temp_ndisp, 'strain': num_str_temrs, 'distance': 0.0}])
                    term_cnre += 1
                if i_term == 0 : #and (no_disp==False or num_str_temrs) > 0:
                    temp_trms = re_order_terms(my_terms[0])
                    key_cntr = 0
                    for my_key in temp_trms.keys():
                        tot_nterms += 1
                        my_value = float(coeff['value']) 
                        my_key = my_key.replace('x', '(eta_1)')
                        my_key = my_key.replace('y', '(eta_2)')
                        my_key = my_key.replace('z', '(eta_3)')
                        my_text = disp_text+my_key
                        if my_text != '':
                            new_coeff = {'number': str(tot_nterms), 'value': str(
                                       my_value), 'text': my_text}
                            new_coeffs.append(new_coeff)
                        key_cntr += 1

        for temp_cntr in range(len(my_str_phon_term)):
            # print(my_str_phon_term[temp_cntr])
            new_temrs.append(my_str_phon_term[temp_cntr])
        
        return(new_coeffs, new_temrs)

    def get_missfit_terms(self, coeff, terms, my_tags, my_strain, voigts=[1, 2, 3]):
        str_phonon_voigt = self.get_str_phonon_voigt(terms, voigts)
        new_coeffs = []
        new_terms = []
        # print(str_phonon_voigt)
        for icoeff in str_phonon_voigt:
            # print(10*'****---- ORG ')
            # # print(terms[icoeff][0])
            # print(coeff[icoeff])
            # print(terms[icoeff])
            temp_coeffs,temp_terms = self.get_missfit_term(coeff[icoeff], terms[icoeff], my_tags, my_strain, voigts=[1, 2, 3])
            for cntr in range(len(temp_coeffs)):
                new_coeffs.append(temp_coeffs[cntr])
                new_terms.append(temp_terms[cntr])
                # print(10*'---------')
                # print(temp_coeffs[cntr])
                # print(temp_terms[cntr])
        
        return(new_coeffs,new_terms)

    def get_disp_text(self,my_term,my_tags):
        disp_text = ''
        ndisp = int(my_term[-1]['dips'])
        for disp in range(ndisp):
            atm_a = int(my_term[disp]['atom_a'])
            atm_b = int(my_term[disp]['atom_b'])
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
        return(disp_text)     

    def get_strain(self, strain=np.zeros(0)):
        voigt_str = [strain[0,0],strain[1,1],strain[2,2],(strain[1,2]+strain[2,1])/2,(strain[0,2]+strain[2,0])/2,(strain[0,1]+strain[1,0])/2]
        return(np.array(voigt_str))

    def get_elas_missfit(self,id_in,my_strain,voigts=[]):
        new_coeffs = []
        new_terms = []
        my_vogt_dic = {1: 'eta_1', 2: 'eta_2', 3: 'eta_3', 4: 'eta_4', 5: 'eta_5', 6: 'eta_6'}
        ela_cnst = (self.xmls_objs[id_in].ela_cons)   #np.linalg.det(self.SCMATS[id_in])*
        tot_nterms = 1
        for alpha in voigts:
            for beta in voigts:
                if my_strain[alpha-1] !=0 or my_strain[beta-1] != 0:
                    if alpha == beta:
                        new_coeffs.append({'number': str(tot_nterms), 'value': str(ela_cnst[alpha,beta]*my_strain[alpha-1]), 'text': my_vogt_dic[alpha]})
                        new_term = [[{'weight': ' 1.000000'},{'power': ' 1', 'voigt': str(alpha)}, {'dips': 0, 'strain': 1, 'distance': 0}]]
                        new_terms.append(new_term)
                        tot_nterms += 1
                    else:
                        if my_strain[alpha-1] > 0.0001:
                            new_coeffs.append({'number': str(tot_nterms), 'value': str(0.5*ela_cnst[alpha,beta]*my_strain[alpha-1]), 'text': str(my_vogt_dic[beta])})
                            new_term = [[{'weight': ' 1.000000'},{'power': ' 1', 'voigt': str(beta)}, {'dips': 0, 'strain': 1, 'distance': 0}]]
                            new_terms.append(new_term)
                            tot_nterms += 1

                        if my_strain[beta-1] > 0.0001:
                            new_coeffs.append({'number': str(tot_nterms), 'value': str(0.5*ela_cnst[alpha,beta]*my_strain[beta-1]), 'text': str(my_vogt_dic[alpha])})
                            new_term = [[{'weight': ' 1.000000'},{'power': ' 1', 'voigt': str(alpha)}, {'dips': 0, 'strain': 1, 'distance': 0}]]
                            new_terms.append(new_term)
                            tot_nterms += 1
        return(new_coeffs,new_terms)

#################################################################################

class new_anh_scl():
    def __init__(self, har_xml, anh_xml):
        self.xml = har_xml
        self.ahxml = anh_xml

    def SC_trms(self, MySC, scll_mat):
        hxml = xml_sys(self.xml)
        hxml.get_ase_atoms()
        hxml.get_str_cp()
        my_strain = hxml.strain
        my_atms = hxml.ase_atoms
        coeff, trms = xml_anha(self.ahxml, my_atms)
        sds = spg.get_symmetry_dataset(my_atms)
        print(
            f"space group number = {sds['number']}  INT = {sds['international']} ")
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
        scld_pos_prm = my_atms.get_scaled_positions()
        wrapPos = ase.geometry.wrap_positions
        cell_prm = my_atms.get_cell()
        my_trms = []
        SC_pos = MySC.get_positions()
        cell_SC = MySC.get_cell()
        for cc in range(len(coeff)):
            my_trms.append([])
            my_int_dict = {}
            for prd_3 in range(scll_mat[0][0]):
                for prd_1 in range(scll_mat[1][1]):
                    for prd_2 in range(scll_mat[2][2]):
                        for my_rot in range(len(rot)):
                            disp_lst = []
                            dis_tmp_weight = 1
                            wrong_acell = False
                            my_new_key = ''
                            for disp in range(trms[cc][0][-1]['dips']):
                                atm_a = int(trms[cc][0][disp]['atom_a'])
                                cell_a = [int(cll) for cll in (
                                    trms[cc][0][disp]['cell_a'].split())]
                                atm_b = int(trms[cc][0][disp]['atom_b'])
                                cell_b = [int(cll) for cll in (
                                    trms[cc][0][disp]['cell_b'].split())]
                                power = trms[cc][0][disp]['power']
                                vec_dir = dir_dic[trms[cc]
                                                  [0][disp]['direction']]
                                rot_vec_dir = vgt_mat[vec_dir, :, :]
                                rot_vec_dir = np.dot(
                                    np.dot(rot[my_rot], rot_vec_dir), (alg.inv(rot[my_rot])))
                                for vgt_cnt in range(1, len(vgt_mat)):
                                    diff = np.matrix.flatten(
                                        abs(rot_vec_dir) - vgt_mat[vgt_cnt, :, :])
                                    if not any(diff):
                                        rot_voit_key = vgt_cnt
                                pos_a = pos_prm[atm_a] + \
                                    np.dot(cell_a, cell_prm)
                                pos_b = pos_prm[atm_b] + \
                                    np.dot(cell_b, cell_prm)
                                #rot_pos_a = np.dot(rot[my_rot],pos_a)+np.dot([prd_1,prd_2,prd_3],cell_prm)
                                #rot_pos_b = np.dot(rot[my_rot],pos_b)+np.dot([prd_1,prd_2,prd_3],cell_prm )
                                rot_pos_a = np.dot(rot[my_rot], pos_a)+np.dot(
                                    trans[my_rot], cell_prm)+np.dot([prd_1, prd_2, prd_3], cell_prm)
                                rot_pos_b = np.dot(rot[my_rot], pos_b)+np.dot(
                                    trans[my_rot], cell_prm)+np.dot([prd_1, prd_2, prd_3], cell_prm)
                                # finding index of atom a and atom b in cell
                                wrp_a = wrapPos([rot_pos_a], cell_SC)[0]
                                wrp_b = wrapPos([rot_pos_b], cell_SC)[0]
                                ncell_a = rot_pos_a - wrp_a
                                ncell_b = rot_pos_b - wrp_b - ncell_a
                                atm_a_ind = find_index(SC_pos, wrp_a, tol=0.01)
                                atm_b_ind = find_index(SC_pos, wrp_b, tol=0.01)
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
                                    'cell_a': '0 0 0',
                                    'atom_b': atm_b_ind,
                                    'cell_b': f"{round(ncell_b[0]/cell_SC[0,0])} {round(ncell_b[1]/cell_SC[1,1])} {round(ncell_b[2]/cell_SC[2,2])}",
                                    'direction': get_key(dir_dic, rot_voit_key),
                                    'power': power
                                }
                                disp_lst.append(ndis)
                                my_new_key = my_new_key + \
                                    f'{atm_a_ind}_{atm_b_ind}_{round(ncell_b[0]/cell_SC[0,0])}_{round(ncell_b[1]/cell_SC[1,1])}_{round(ncell_b[2]/cell_SC[2,2])}'
                            tem_vogt_wgt = 1
                            if not(wrong_acell):
                                if trms[cc][0][-1]['dips'] != 0 and trms[cc][0][-1]['strain'] != 0 and len(disp_lst) > 0:
                                    for strain in range(trms[cc][0][-1]['strain']):
                                        voigt = int(
                                            trms[cc][0][disp+strain+1]['voigt'])
                                        pwr_str = int(
                                            trms[cc][0][disp+strain+1]['power'])
                                        vgt_trans = np.dot(
                                            np.dot(rot[my_rot], vgt_mat[voigt, :, :]), (alg.inv(rot[my_rot])))
                                        for vgt_cnt in range(1, len(vgt_mat)):
                                            diff = np.matrix.flatten(
                                                -(vgt_trans) - vgt_mat[vgt_cnt, :, :])
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
                                        my_new_key = my_new_key + \
                                            f'_Vogt_{rot_voit_key}_P_{pwr_str}'

                                for i in range(len(disp_lst)):
                                    disp_lst[i]['weight'] = " %2.6f" % (
                                        dis_tmp_weight*tem_vogt_wgt)
                                if len(disp_lst) > 0:
                                    disp_lst.append(trms[cc][0][-1])
                                    found = False
                                    prm_lst = list(permutations(disp_lst[:]))
                                    for prm_mem in prm_lst:
                                        my_temp = [prm for prm in prm_mem]
                                        if my_temp in my_trms[cc]:
                                            found = True
                                    if disp_lst not in my_trms[cc] and not found:
                                        my_trms[cc].append(disp_lst)

            for my_rot in range(len(rot)):
                tem_vogt_wgt = 1
                if trms[cc][0][-1]['dips'] == 0 and trms[cc][0][-1]['strain'] != 0:
                    disp_lst = []
                    for strain in range(trms[cc][0][-1]['strain']):
                        voigt = int(trms[cc][0][strain+1]['voigt'])
                        pwr_str = int(trms[cc][0][strain+1]['power'])
                        vgt_trans = np.dot(
                            np.dot(rot[my_rot], vgt_mat[voigt, :, :]), (alg.inv(rot[my_rot])))
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
                    disp_lst.append(trms[cc][0][0])
                    disp_lst.append(trms[cc][0][-1])
                    prm_lst = list(permutations(disp_lst[:]))
                    found = False
                    for prm_mem in prm_lst:
                        my_temp = [prm for prm in prm_mem]
                        if my_temp in my_trms[cc]:
                            found = True

                    if disp_lst not in my_trms[cc] and len(disp_lst) > 0 and not found:
                        my_trms[cc].append(disp_lst)

        self.SC_terms = my_trms
        self.SC_coeff = coeff

    def wrt_anxml(self, fout):
        coeff = self.SC_coeff
        trms = self.SC_terms
        output = open(fout, 'w')
        output.write('<?xml version="1.0" ?>\n')
        output.write('<Heff_definition>\n')
        for i in range(len(coeff)):
            output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
                coeff[i]['number'], coeff[i]['value'], coeff[i]['text'],))
            for j in range(len(trms[i])):
                for k in range(int(trms[i][j][-1]['dips'])):
                    if (k == 0):
                        output.write('    <term weight="{}">\n'.format(
                            trms[i][j][k]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(
                        trms[i][j][k]['atom_a'], trms[i][j][k]['atom_b'], trms[i][j][k]['direction'], trms[i][j][k]['power']))
                    output.write(
                        '        <cell_a>{}</cell_a>\n'.format(trms[i][j][k]['cell_a']))
                    output.write(
                        '        <cell_b>{}</cell_b>\n'.format(trms[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')
                if int(trms[i][j][-1]['dips']) == 0:
                    k = int(trms[i][j][-1]['strain'])
                    output.write('    <term weight="{}">\n'.format(
                        trms[i][j][k]['weight']))
                    k = -1
                for l in range(int(trms[i][j][-1]['strain'])):
                    output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                        trms[i][j][k+l+1]['power'], trms[i][j][k+l+1]['voigt']))
                output.write('    </term>\n')
            output.write('  </coefficient>\n')
        output.write('</Heff_definition>\n')

#####################################################


def str_mult(a,b):
    '''this function returns multiplication of two strings as like :
        a   >  x      b   >>    a result    >   a x'''
    my_list = [*a.split(),*b.split()]
    my_list.sort() 
    return(' '.join(my_list))

def terms_mult(T_1,T_2):
    '''This function returns multiplication of two terms T_1 and T_2 
    T1  >   {'z': 1, 'c': 1}  T2  >  {'z': 1, 'c': 1}  ===>   T1T2  > {'z z': 1, 'c z': 2, 'c c': 1}'''
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
    '''This function return term T1 to the power of n'''
    if n-1!=0:
        return(terms_mult(get_pwr_N(T1,n-1),T1))
    else:
        return(T1)

def re_order_terms(T1):
    '''This function changes a dictionary written as {' x x x : 1} to {x^3 : 1}
    T1  >>   {'  x': 0.2, ' ': 0.010000000000000002} Fin_dict >>   {'x^1': 0.2, '': 0.010000000000000002}'''
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

def Xu_write_vasp(filename, atoms, label='', direct=False, sort=True, symbol_count=None, long_format=True, vasp5=True):
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
    else:  # Assume it's a 'file-like object'
        f = filename

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError("Don't know how to save more than " +
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
        ind = np.argsort(atoms.get_chemical_symbols(), kind='mergesort')
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

def to_text(arr):
    mytxt = ''
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            if i == 0 and j == 0:
                mytxt = ' {:.14E}'.format(arr[i][j])
            else:
                mytxt = mytxt+'  {:.14E}'.format(arr[i][j])
        mytxt = mytxt+'\n'
    return(mytxt)


def one_text(arr):
    mytxt = ''
    a = len(arr)
    for i in range(a):
        if i == 0:
            mytxt = ' {:.14E}'.format(arr[i])
        else:
            mytxt = mytxt+'  {:.14E}'.format(arr[i])
    mytxt = mytxt+'\n'
    return(mytxt)


def get_atom_num(atomic_mass, tol=0.1):
    if abs(atomic_mass-208) < 1:
        tol = 0.001
    for i in range(len(atomic_masses)):
        if abs(atomic_masses[i]-atomic_mass) < tol:
            mynum = i
    return(mynum)


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


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


if __name__ == '__main__':
    xmlf = 'Har_xml_141414'
    dim = 3
    sys = xml_sys(xmlf)
    sys.get_ase_atoms()
    atms = sys.ase_atoms
    scll = np.eye(3, dtype=int)*dim
    my_sc = make_supercell(atms, scll)
    # Xu_write_vasp('POSCAR_reorders',my_sc)
    t0 = time.perf_counter()
    #msc = my_sc_maker(xmlf,scll)
    #NEW_SCLL =read('POSCAR_reorders')
    # msc.reshape_FCDIC(NEW_SCLL)
    # msc.write_xml(f'tst_{dim}.xml')
    # te=time.perf_counter()
    #print('time old  =>  {}'.format(te-t0))
    anh_file = "AnHar_xml_141414"
    anxml = anh_scl(xmlf, anh_file)
    anxml.SC_trms(my_sc, scll)
    anxml.wrt_anxml(f'{dim}_anxml.xml')
    te = time.perf_counter()
    print('time old  =>  {}'.format(te-t0))
