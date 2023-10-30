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
from ase.build import stack
import xml_io  # xml_io.xml_anha_reader
import tools

###########################################################

###########################################################

class Har_interface:
    ''' interface harmonic potential generation: This class is given two xml file for two materilas for the interface two 3x3 matirixes(SC_mat1 and SC_mat2) which give the matirx for
     one side of the SL (these should have SC_mat1[0,0]=SC_mat2[0,0] and SC_mat1[1,1]=SC_mat2[1,1] while SC_mat1[2,2] and SC_mat2[2,2] define the thickness of the each material
     in two sides of the SL) '''
    
    def __init__(self, xml_file1, SCMAT_1, xml_file2, SCMAT_2, symmetric=False,negelect_A_SITE=False,negelect_Tot_FCs=False,NW_Strc = False):
        self.negelect_A_SITE = negelect_A_SITE
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
        self.NW_Strc = NW_Strc
        if self.NW_Strc:
            
            self.Constr_NW()
        else:
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
        my_xml_obj = xml_io.Xml_sys_reader(xml_file, mat_id=str(
            self.__Curnt_id), extract_dta=True)
        self.SCMATS[str(self.__Curnt_id)] = np.array(SCMAT)
        self.xmls_objs[str(self.__Curnt_id)] = my_xml_obj
        self.uc_atoms[str(self.__Curnt_id)] = my_xml_obj.ase_atoms
        self.loc_FCs[str(self.__Curnt_id)] = my_xml_obj.loc_FC_tgs
        self.tot_FCs[str(self.__Curnt_id)] = my_xml_obj.tot_FC_tgs
        self.has_tot_FC *= my_xml_obj.has_tot_FC
        self.__Curnt_id += 1

    def get_cells(self): 
        SC_mat1 = self.SCMATS[str(0)]
        temp_loc_keys1 = self.xmls_objs[str(0)].loc_cells
        temp_loc_keys2 = self.xmls_objs[str(1)].loc_cells
        temp_tot_keys1 = self.xmls_objs[str(0)].tot_cells
        temp_tot_keys2 = self.xmls_objs[str(1)].tot_cells
        SC_mat2 = self.SCMATS[str(1)]
        ##############################
        minxl = min(ceil(temp_loc_keys1[0][0]/SC_mat1[0][0]), ceil(temp_loc_keys2[0][0]/SC_mat2[0][0]), ceil(
            temp_tot_keys1[0][0]/SC_mat1[0][0]), ceil(temp_tot_keys2[0][0]/SC_mat2[0][0]))
        
        maxxl = max(ceil(temp_loc_keys1[0][1]/SC_mat1[0][0]), ceil(temp_loc_keys2[0][1]/SC_mat2[0][0]), ceil(
            temp_tot_keys1[0][1]/SC_mat1[0][0]), ceil(temp_tot_keys2[0][1]/SC_mat2[0][0]))
        
        minyl = min(ceil(temp_loc_keys1[1][0]/SC_mat1[1][1]), ceil(temp_loc_keys2[1][0]/SC_mat1[1][1]), ceil(
            temp_tot_keys1[1][0]/SC_mat1[1][1]), ceil(temp_tot_keys2[1][0]/SC_mat1[1][1]))
        
        maxyl = max(ceil(temp_loc_keys1[1][1]/SC_mat1[1][1]), ceil(temp_loc_keys2[1][1]/SC_mat2[1][1]), ceil(
            temp_tot_keys1[1][1]/SC_mat1[1][1]), ceil(temp_tot_keys2[1][1]/SC_mat2[1][1]))
        if self.NW_Strc :
            minzl = min(ceil(temp_loc_keys1[2][0]/SC_mat1[2][2]), ceil(temp_loc_keys2[2][0]/SC_mat2[2][2]), ceil(
                temp_tot_keys1[2][0]/SC_mat1[2][2]), ceil(temp_tot_keys2[2][0]/SC_mat2[2][2]))
            
            maxzl = max(ceil(temp_loc_keys1[2][1]/SC_mat1[2][2]), ceil(temp_loc_keys2[2][1]/SC_mat2[2][2]), ceil(
                temp_tot_keys1[2][1]/SC_mat1[2][2]), ceil(temp_tot_keys2[2][1]/SC_mat2[2][2]))
        else:
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

    def get_match_pairs(self):  
        self.maped_strs = {}
        self.maped_strs['0'] = tools.get_mapped_strcs(
            self.uc_atoms['0'], self.uc_atoms['1'], Ret_index=True)
        self.maped_strs['1'] = tools.get_mapped_strcs(
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
        Cells = self.get_cells()    ### FIXME FOR NWs
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

    def reshape_FCDIC(self):
        tol_10 = 10**-10
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
            if loc_key_found and any(abs(np.array(tmp_loc_FC.flatten()))>tol_10):   # chek in all not zero FIXME
                self.Fin_loc_FC[my_key] = tmp_loc_FC
            if tot_key_found and any(abs(np.array(tmp_tot_FC.flatten()))>tol_10):
                self.Fin_tot_FC[my_key] = tmp_tot_FC
        self.has_FIN_FCDIC = True

    def write_xml(self, out_put='test.xml'):
        xml_dta = {}
        ref_cell = self.ref_cell
        self.cal_eps_inf()
        SCL_elas = (np.linalg.det(self.SCMATS['0'])*(self.xmls_objs['0'].ela_cons)+np.linalg.det(
            self.SCMATS['1'])*(self.xmls_objs['1'].ela_cons))
        ref_eng =  np.linalg.det(self.SCMATS['0'])*self.xmls_objs['0'].ref_eng+np.linalg.det(self.SCMATS['1'])*self.xmls_objs['1'].ref_eng 

        str_ph = ref_cell.get_array('str_ph')
        self.SL_BEC_cal()        
        if self.has_tot_FC:
            keys = self.Fin_tot_FC.keys()
            xml_dta['SC_total_FC'] = self.Fin_tot_FC
            xml_dta['has_tot_FC'] = self.has_tot_FC            
        else:
            xml_dta['has_tot_FC'] = self.has_tot_FC 
            keys = self.Fin_loc_FC.keys()
        xml_dta['keys'] = keys
        xml_dta['SCL_elas'] = SCL_elas
        xml_dta['SCL_ref_energy'] =ref_eng
        xml_dta['SCL_lat'] = ref_cell.get_cell()/Bohr 
        xml_dta['eps_inf'] = self.SL_eps_inf 
        xml_dta['atoms_mass'] = ref_cell.get_masses() 
        xml_dta['SC_BEC'] = self.SL_BEC 
        xml_dta['SC_atoms_pos'] = ref_cell.get_positions()/Bohr 
        xml_dta['SC_local_FC'] = self.Fin_loc_FC
        xml_dta['SC_corr_forc'] = str_ph
        xml_dta['strain'] = self.xmls_objs['0'].strain
        xml_dta['my_atm_list'] = range(len(ref_cell.get_masses()))
        xml_io.write_sys_xml(xml_dta,out_put)

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
        
        SCL['0'] = [np.sqrt(eps_SL[0][0]/eps['0'][0][0]), np.sqrt(eps_SL[1]
                                                                  [1]/eps['0'][1][1]), np.sqrt(eps_SL[2][2]/eps['0'][2][2])]
        SCL['1'] = [np.sqrt(eps_SL[0][0]/eps['1'][0][0]), np.sqrt(eps_SL[1]
                                                                  [1]/eps['1'][1][1]), np.sqrt(eps_SL[2][2]/eps['1'][2][2])]

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
        self.ref_cell.set_array(
            'BEC', SL_BEC)  
        self.STRC.set_array(
            'BEC', SL_BEC)        

    def cal_eps_inf(self):
        self.SL_eps_inf = np.zeros((3, 3))
        if self.NW_Strc:
            l_1 = np.linalg.det(self.SCMATS['0']) /(np.linalg.det(self.SCMATS['0'])+np.linalg.det(self.SCMATS['1']))

            l_2 = np.linalg.det(self.SCMATS['1']) /(np.linalg.det(self.SCMATS['0'])+np.linalg.det(self.SCMATS['1']))
            
            eps_xx = l_1*self.xmls_objs['0'].eps_inf[0][0] + \
                    l_2*self.xmls_objs['1'].eps_inf[0][0]
            
            eps_yy = l_1*self.xmls_objs['0'].eps_inf[1][1] + \
                   l_2*self.xmls_objs['1'].eps_inf[1][1]
            
            eps_zz = l_1*self.xmls_objs['0'].eps_inf[1][1] + \
                    l_2*self.xmls_objs['1'].eps_inf[1][1]           
        else:

            l_1 = self.SCMATS['0'][2][2] / (self.SCMATS['0'][2][2]+self.SCMATS['1'][2][2])
            l_2 = self.SCMATS['1'][2][2] / (self.SCMATS['0'][2][2]+self.SCMATS['1'][2][2])
            
            eps_xx = l_1*self.xmls_objs['0'].eps_inf[0][0] + \
                l_2*self.xmls_objs['1'].eps_inf[0][0]
            
            eps_yy = l_1*self.xmls_objs['0'].eps_inf[1][1] + \
                l_2*self.xmls_objs['1'].eps_inf[1][1]
            
            eps_zz = 1/(l_1*(1/self.xmls_objs['0'].eps_inf[2]
                        [2])+l_2*(1/self.xmls_objs['1'].eps_inf[2][2]))
            
        self.SL_eps_inf[0, 0] = eps_xx
        self.SL_eps_inf[1, 1] = eps_yy
        self.SL_eps_inf[2, 2] = eps_zz

    def Constr_SL(self, symmetric):
        cell_par1 = self.uc_atoms[str(0)].get_cell_lengths_and_angles()
        cell_par2 = self.uc_atoms[str(1)].get_cell_lengths_and_angles()
        cell_parr_diff = [0,0,(cell_par2[2]-cell_par1[2])/2]
        dim_a = self.SCMATS[str(0)][0][0]
        dim_b = self.SCMATS[str(0)][1][1]
        zdir_L1 = self.SCMATS[str(0)][2, 2]
        zdir_L2 = self.SCMATS[str(1)][2, 2]
        STRC_atom_2 = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(), scaled_positions = self.uc_atoms[str(1)].get_scaled_positions(), cell = self.uc_atoms[str(0)].get_cell())
        STRC_atom_2.set_array(
            'BEC', self.uc_atoms[str(1)].get_array('BEC'))
        STRC_atom_2.set_array(
            'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
        STRC_atom_2.set_array(
            'str_ph', self.uc_atoms[str(1)].get_array('str_ph')) 
        
        ref_atom_2 = Atoms(numbers=self.uc_atoms[str(1)].get_atomic_numbers(), positions = self.uc_atoms[str(1)].get_positions()+cell_parr_diff, cell = self.uc_atoms[str(1)].get_cell())
        ref_atom_2.set_array(
            'BEC', self.uc_atoms[str(1)].get_array('BEC'))
        ref_atom_2.set_array(
            'tag_id', self.uc_atoms[str(1)].get_array('tag_id'))
        ref_atom_2.set_array(
            'str_ph', self.uc_atoms[str(1)].get_array('str_ph'))   

        self.STRC_uc = self.uc_atoms[str(0)]      
        self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell() 
        if symmetric:
            zdir_1 = int(zdir_L1/2)
            zdir_3 = int(zdir_L1-zdir_1)
            L1 = self.uc_atoms[str(0)].repeat([dim_a,dim_b,zdir_1])
            L2_ref = ref_atom_2.repeat([dim_a,dim_b,zdir_L2])    
            L2_Strc = STRC_atom_2.repeat([dim_a,dim_b,zdir_L2])
            L3 = self.uc_atoms[str(0)].repeat([dim_a,dim_b,zdir_3])

            res_L1_L2 = stack(L1,L2_ref,axis=2,fix=0.5,distance=None)
            Strc_L1_L2 = stack(L1,L2_Strc,axis=2,fix=0,distance=None)
            self.ref_cell = stack(res_L1_L2,L3,axis=2,fix=0.5,distance=None)
            self.STRC = stack(Strc_L1_L2,L3,axis=2,fix=0,distance=None)  

              
            write('Ref_cell.cif',self.ref_cell,format='cif')
            # write('STRC.cif',self.STRC,format='cif')                      
        else:

            L1 = self.uc_atoms[str(0)].repeat([dim_a,dim_b,zdir_L1])
            L2 = ref_atom_2.repeat([dim_a,dim_b,zdir_L2])   
            L2_Strc = STRC_atom_2.repeat([dim_a,dim_b,zdir_L2])
            # to be written in XML
            self.ref_cell = stack(L1,L2,axis=2,fix=0,distance=None)
            # To be used in creating AFCs
            self.STRC = stack(L1,L2_Strc,axis=2,fix=0,distance=None)   
            # self.STRC_uc_cell = self.uc_atoms[str(0)].get_cell()  

            write('Ref_cell.cif',self.ref_cell,format='cif')
            # write('STRC.cif',self.STRC,format='cif')            

    def Constr_NW(self):
        atoms_1 = self.uc_atoms['0']
        atoms_2 = self.uc_atoms['1']  
        dim_a1 = self.SCMATS[str(0)][0][0]
        dim_b1 = self.SCMATS[str(0)][1][1]

        dim_a2 = self.SCMATS[str(1)][0][0]
        dim_b2 = self.SCMATS[str(1)][1][1]

        zdir = self.SCMATS[str(0)][2, 2]

        STRC_atom_2 = Atoms(numbers=atoms_2.get_atomic_numbers(), scaled_positions = atoms_2.get_scaled_positions(), cell = atoms_1.get_cell())
        STRC_atom_2.set_array(
            'BEC', atoms_2.get_array('BEC'))
        STRC_atom_2.set_array(
            'tag_id', atoms_2.get_array('tag_id'))
        STRC_atom_2.set_array(
            'str_ph', atoms_2.get_array('str_ph')) 

        # cell_par1 = atoms_1.get_cell_lengths_and_angles()
        # cell_par2 = atoms_2.get_cell_lengths_and_angles()
        # cell_parr_diff = [0,0,(cell_par2[2]-cell_par1[2])/2]        
        # ref_atom_2 = Atoms(numbers=atoms_2.get_atomic_numbers(), positions = atoms_2.get_positions()+cell_parr_diff, cell = atoms_2.get_cell())
        # ref_atom_2.set_array(
        #     'BEC', atoms_2.get_array('BEC'))
        # ref_atom_2.set_array(
        #     'tag_id', atoms_2.get_array('tag_id'))
        # ref_atom_2.set_array(
        #     'str_ph', atoms_2.get_array('str_ph')) 
         
        dim_b_temp = int((dim_b1-dim_b2)/2)

        dim_a_temp = int((dim_a1-dim_a2)/2)

        s1 = atoms_1.repeat([dim_a1,dim_b_temp,zdir])

        s2 = atoms_1.repeat([dim_a_temp,dim_b2,zdir])

        s3 = STRC_atom_2.repeat([dim_a2,dim_b2,zdir])

        s4 = stack(s2,s3,axis=0)
        s5 = stack(s4,s2,axis=0)
        s6 = stack(s1,s5,axis=1)
        s7 = stack(s6,s1,axis=1)
        
        write('POSCAR_ref',s7,vasp5=True,sort=True)
        self.STRC_uc_cell = atoms_1.get_cell() 
        self.STRC_uc = atoms_1
        self.STRC = s7
        self.ref_cell = s7

###########################################################
# interface Anharmonic potential generation:

class Anh_intrface(Har_interface):
    def __init__(self, har_xml1, anh_xml1, SC_mat1, har_xml2, anh_xml2, SC_mat2, symmetric=False,NW_Strc = False):
        Har_interface.__init__(self, har_xml1, SC_mat1,
                               har_xml2, SC_mat2, symmetric=symmetric,NW_Strc = NW_Strc)
        self.coeff = {}
        self.terms = {}
        self.STRC_terms = {}
        self.STRC_coeffs = {}
        self.coeff['0'], self.terms['0'] = xml_io.xml_anha_reader(
            anh_xml1, self.uc_atoms['0'])
        self.coeff['1'], self.terms['1'] = xml_io.xml_anha_reader(
            anh_xml2, self.uc_atoms['1'])
            
        self.has_weight = False
        self.get_match_pairs()
        if not self.has_weight:
            req_elemtsns = self.diff_elements
            self.get_FC_weight(req_symbs=req_elemtsns)

    def STRC_trms(self, id_in='0'):
        id_pars = {'0': '1', '1': '0'}
        tol_04 = 10**-4
        UC_STR = self.STRC_uc 
        coeff, trms = self.coeff[id_in], self.terms[id_in]
        Xred_STRC = self.STRC.get_scaled_positions()
        STRC_cell = self.STRC.get_cell()
        inv_STRC_cell = np.linalg.inv(STRC_cell)
        cPOS = UC_STR.get_positions()
        tag_id = self.STRC.get_array('tag_id')
        uc_cell = UC_STR.get_cell()
        ###################################
        tot_scmat = self.SCMATS[id_in].copy()
        tot_scmat[2,2] += self.SCMATS[id_pars[id_in]][2, 2]
        ncell = np.linalg.det(tot_scmat)

        wrapPos = ase.geometry.wrap_positions
        my_terms = []

        if self.NW_Strc:
            SCMATS_temp = np.zeros((3,3),dtype=int)
            for idir in range(3):
                SCMATS_temp[idir,idir]=int(max((self.SCMATS[id_in][idir,idir]),self.SCMATS[id_pars[id_in]][idir,idir]))

            prop_l1l2 = np.linalg.det(self.SCMATS[id_in])/(np.linalg.det(self.SCMATS[id_in])+np.linalg.det(self.SCMATS[id_pars[id_in]])) 
            zdirec1 = range(2*SCMATS_temp[2, 2])


        else:
            SCMATS_temp = self.SCMATS[id_in]
            zdirec1 = range((self.SCMATS[id_in][2, 2]) +
                        (self.SCMATS[id_pars[id_in]][2, 2]))  
            prop_l1l2 = self.SCMATS[id_in][2][2]/(self.SCMATS[id_in][2][2]+self.SCMATS[id_pars[id_in]][2][2])                                  

        tmp_coeffs = {}
        for cc in range(len(coeff)):
            tmp_coeffs[cc] = coeff[cc]
            my_terms.append([])
            my_SAT_terms = trms[cc]  
            for tc in range(len(my_SAT_terms)):
                for prd1 in range(SCMATS_temp[0, 0]):
                    for prd2 in range(SCMATS_temp[1, 1]):
                        for prd3 in zdirec1:

                            my_term = []
                            disp_cnt = 0
                            prd_dis = np.dot(uc_cell, [prd1, prd2, prd3])
                            temp_weight = 0
                            for disp in range(int(my_SAT_terms[tc][-1]['dips'])):
                                atm_a = int(my_SAT_terms[tc][disp]['atom_a'])
                                atm_b = int(my_SAT_terms[tc][disp]['atom_b'])
                                cell_b0 = [
                                    int(x) for x in my_SAT_terms[tc][disp]['cell_b'].split()]
                                catm_a0 = cPOS[atm_a]
                                catm_b0 = np.dot(uc_cell, cell_b0)+cPOS[atm_b]
                                dst0 = catm_a0-catm_b0
                                catm_an = prd_dis + catm_a0
                                catm_bn = prd_dis + catm_b0
                                red_an = np.dot(inv_STRC_cell, catm_an)
                                red_bn = np.dot(inv_STRC_cell, catm_bn)
                                ind_an = tools.find_index_xred(Xred_STRC, red_an)
                                ind_bn = tools.find_index_xred(Xred_STRC, red_bn)
                                if ind_an == -1:
                                    wrp_a = wrapPos([catm_an], STRC_cell)[0]
                                    red_ann = np.dot(inv_STRC_cell, wrp_a)
                                    ind_an = tools.find_index_xred(
                                        Xred_STRC, red_ann)
                                if ind_bn == -1:
                                    wrp_b = wrapPos([catm_bn], STRC_cell)[0]
                                    red_bnn = np.dot(inv_STRC_cell, wrp_b)
                                    ind_bn = tools.find_index_xred(
                                        Xred_STRC, red_bnn)
                                cell_a = red_an-Xred_STRC[ind_an]
                                cell_b = list(
                                    map(int, red_bn-Xred_STRC[ind_bn]-np.array(cell_a)))
                                tag_an, id_an = tag_id[ind_an]
                                dst = [catm_an[i]-catm_bn[i] for i in range(3)]
                                dif_ds = [
                                    False if (abs(dst[i]-dst0[i]) > tol_04) else True for i in range(3)]
                                if all(dif_ds):
                                    trm_weight = self.FC_weights[ind_an][int(
                                        id_in)]+self.FC_weights[ind_bn][int(id_in)]
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
                                for ii_disp in range(int(my_SAT_terms[tc][-1]['dips'])):
                                    my_term[ii_disp]['weight'] = float(
                                        my_SAT_terms[tc][ii_disp]['weight'])*temp_weight
                                if my_term not in (my_terms[cc]):
                                    my_terms[cc].append(my_term)

            if len(my_SAT_terms)>=1:
                if (int(my_SAT_terms[0][-1]['dips']) == 0) and (int(my_SAT_terms[0][-1]['strain'])!=0):
                    tmp_coeffs[cc]['value'] = ncell*prop_l1l2*float(tmp_coeffs[cc]['value'])
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

                atm_a_ind = tools.find_index(pos_prm, wrp_a, tol=0.0001)
                atm_b_ind = tools.find_index(pos_prm, wrp_b, tol=0.0001)

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
                    'direction': tools.get_key(dir_dic, rot_voit_key),
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

    def wrt_anxml(self, fout):
        self.STRC_trms(id_in='0')
        self.STRC_trms(id_in='1')
        coeff1 = self.STRC_coeffs['0'] 
        trms1 = self.STRC_terms['0']
        coeff2 = self.STRC_coeffs['1'] 
        trms2 = self.STRC_terms['1']
        str_coeff = {}
        str_terms = []        
        total_coeffs = {}
        total_tems = []
        coef_cntr = 1
        str_cntr = 0
        for i in range(len(coeff1)):
            print(20*'-----')
            print(coeff1[i])
            print(len(trms1[i]))                   
            if trms1[i][0][-1]['dips'] != 0 :
                total_coeffs[coef_cntr] = coeff1[i]
                total_tems.append(trms1[i])
                coef_cntr +=1                
            else:
                str_coeff[str_cntr] = coeff1[i]
                str_terms.append(trms1[i])
                str_cntr += 1                 
        # print('coeffs anf terms lengst are :',len(coeff2),'  and ',len(trms2))
        for i in range(len(coeff2)):
            if trms2[i][0][-1]['dips'] != 0 :
                total_coeffs[coef_cntr] = coeff2[i]
                total_tems.append(trms2[i])
                coef_cntr +=1                
            else:
                str_coeff[str_cntr] = coeff2[i]
                str_terms.append(trms2[i])
                str_cntr += 1     
         
        for i in range(str_cntr):
            total_coeffs[coef_cntr] = str_coeff[i]
            total_tems.append(str_terms[i])
            coef_cntr +=1 

        xml_io.write_anh_xml(total_coeffs,total_tems,fout)


###########################################################
if __name__ == '__main__':
    pass
    