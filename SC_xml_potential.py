import numpy as np
from math import ceil
from ase import Atoms
from ase.units import Bohr
from ase.build import make_supercell,sort
import xml_io
import missfit_terms
import  tools 

###############################################################################

class Har_sc_maker():
    def __init__(self, xml_file, SC_mat,strain_in=np.zeros((3,3)),neglect_tot_FC=False):
        self.__Curnt_id = 0        
        self.add_material(xml_file, SC_mat,strain_in)  
        self.mySC = make_supercell(self.my_atoms, self.SC_mat) 
        self.SC_natom = self.mySC.get_global_number_of_atoms()
        self.has_SC_FCDIC = False
        self.has_FIN_FCDIC = False
        self.neglect_tot_FC = neglect_tot_FC

    def add_material(self, xml_file, SCMAT,strain_in):
        my_xml_obj = xml_io.Xml_sys_reader(xml_file, mat_id=str(
            self.__Curnt_id), extract_dta=True)        
        self.SC_mat = np.array(SCMAT)
        self.xml = my_xml_obj

        temp_atoms = my_xml_obj.ase_atoms
        cell = np.dot((np.eye(3)+strain_in),temp_atoms.get_cell())
        self.my_atoms = Atoms(numbers= temp_atoms.get_atomic_numbers(),scaled_positions=temp_atoms.get_scaled_positions(),cell = cell)  
        self.my_atoms.set_array(
            'BEC', temp_atoms.get_array('BEC'))
        self.my_atoms.set_array(
            'tag_id', temp_atoms.get_array('tag_id'))
        self.my_atoms.set_array(
            'str_ph', temp_atoms.get_array('str_ph'))
        
        self.STRC_uc_cell = self.my_atoms.get_cell()
        self.loc_FC_dic = my_xml_obj.loc_FC_tgs
        self.tot_FC_dic = my_xml_obj.tot_FC_tgs
        self.has_tot_FC = my_xml_obj.has_tot_FC*(not self.neglect_tot_FC)
                  
    def get_period_cells(self):
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
        return(xperiod,yperiod,zperiod) 
                           
    def get_SC_FCDIC(self):
        STRC_inv_uc_cell = np.linalg.inv(self.STRC_uc_cell) 
        tag_id = self.mySC.get_array('tag_id')
        STRC_Cell =  self.mySC.get_cell()      
        xperiod,yperiod,zperiod = self.get_period_cells()
        self.loc_SC_FCDIC = {}
        self.tot_SC_FCDIC = {}
        self.tot_mykeys = []
        self.loc_mykeys = []
        tmp_atm = {'0':self.my_atoms}
        Ucells_vecs_in_STRC = tools.get_Uclls_in_STRC(self.mySC,tmp_atm)        
        for prd1 in range(-int(xperiod), int(xperiod)+1):
            for prd2 in range(-int(yperiod), int(yperiod)+1):
                for prd3 in range(-int(zperiod), int(zperiod)+1):
                    SC_cell = f'{prd1} {prd2} {prd3}'  #.format(prd1, prd2, prd3)
                    per_dist = np.dot(np.array([prd1, prd2, prd3]), STRC_Cell)
                    Per_cells = np.dot((1/0.98)*STRC_inv_uc_cell, per_dist)
                    for atm_i in range(self.SC_natom):
                        tag_i = tag_id[atm_i][0]
                        id_i = tag_id[atm_i][1]
                        for atm_j in range(self.SC_natom):
                            tag_j = tag_id[atm_j][0]
                            id_j = tag_id[atm_j][1]
                            cell_b = Ucells_vecs_in_STRC[atm_j,atm_i] + Per_cells
                            cell_b = list(map(int, cell_b))  
                            UC_key = f'{id_i}{tag_i}_{id_j}{tag_j}_{cell_b[0]}_{cell_b[1]}_{cell_b[2]}'  
                            SC_key = f'{atm_i}_{atm_j}_{prd1}_{prd2}_{prd3}'
                            if UC_key in self.loc_FC_dic.keys():
                                self.loc_SC_FCDIC[SC_key] = self.loc_FC_dic[UC_key]
                                if SC_cell not in (self.loc_mykeys):
                                    self.loc_mykeys.append(SC_cell)
                            if self.has_tot_FC and UC_key in self.tot_FC_dic.keys():
                                self.tot_SC_FCDIC[SC_key] = self.tot_FC_dic[UC_key]
                                if SC_cell not in (self.tot_mykeys):
                                    self.tot_mykeys.append(SC_cell)
        self.has_SC_FCDIC = True

    def reshape_FCDIC(self, tmp_sc=None):
        if not self.has_SC_FCDIC:
            self.get_SC_FCDIC()
        self.Fin_loc_FC = {}
        if self.has_tot_FC:
            self.Fin_tot_FC = {}
            my_keys = self.tot_mykeys
        else:
            my_keys = self.loc_mykeys
        if tmp_sc is not None:
            my_atm_list = tools.mapping(tmp_sc,self.mySC)
            self.mySC = sort(self.mySC,tags = my_atm_list)
            # self.mySC=make_supercell(tmp_sc,self.SC_mat)
        else:
            my_atm_list = range(self.SC_natom)
            
        for my_key in (my_keys):
            my_cell = [int(x) for x in my_key.split()]
            loc_key_found = False
            tot_key_found = False
            tmp_loc_FC = np.zeros((3*self.SC_natom, 3*self.SC_natom))
            tmp_tot_FC = np.zeros((3*self.SC_natom, 3*self.SC_natom))
            cnt_a = 0
            for atm_a in my_atm_list:
                cnt_b = 0
                for atm_b in my_atm_list:
                    my_index = f'{atm_a}_{atm_b}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
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

    def write_xml(self, out_put):
        SC_num_Uclls = np.linalg.det(self.SC_mat)
        xml_dta = {}
        if not self.has_FIN_FCDIC:
            self.reshape_FCDIC()
        SC_FC = self.Fin_loc_FC
        self.xml.get_tot_forces()
        xml_dta['has_tot_FC'] = self.has_tot_FC 
        if self.has_tot_FC:
            keys = self.Fin_tot_FC.keys()
            xml_dta['SC_total_FC'] = self.Fin_tot_FC                      
        else:
            keys = SC_FC.keys()

        SCL_elas = ((self.xml.ela_cons)*SC_num_Uclls)
        xml_dta['keys'] = keys
        xml_dta['SCL_elas'] = SCL_elas
        xml_dta['SCL_ref_energy'] = SC_num_Uclls*self.xml.ref_eng
        xml_dta['SCL_lat'] = self.mySC.get_cell()/Bohr
        xml_dta['eps_inf'] = self.xml.eps_inf
        xml_dta['atoms_mass'] = self.mySC.get_masses()
        xml_dta['SC_BEC'] = self.mySC.get_array('BEC') 
        xml_dta['SC_atoms_pos'] = self.mySC.get_positions()/Bohr
        xml_dta['SC_local_FC'] = self.Fin_loc_FC

        xml_dta['SC_corr_forc'] = self.mySC.get_array('str_ph')   
        xml_dta['strain'] = self.xml.strain
        xml_dta['my_atm_list'] = range(len(self.mySC))

        xml_io.write_sys_xml(xml_dta,out_put)

################################################################################
# Anharmonic Potential Creation::

class Anh_sc_maker():
    def __init__(self, har_xml, anh_xml,strain_in=np.zeros((3,3)),missfit_strain=True,Higher_order_strain=False):
        self.xml = har_xml
        self.ahxml = anh_xml
        strain_vogt = missfit_terms.get_strain(strain=strain_in)
        self.missfit_strain = missfit_strain
        self.Higher_order_strain = Higher_order_strain
        if any(strain_vogt)>0.0001:
            self.has_strain = True
            self.voigt_strain = strain_vogt
        else:
            self.has_strain = False
            self.voigt_strain = [0,0,0,0,0,0]
            
    def SC_trms(self, MySC, SC_mat):
        myxml_clss = xml_io.Xml_sys_reader(self.xml)
        myxml_clss.get_ase_atoms()
        
        my_atoms = myxml_clss.ase_atoms
        coeff, trms = xml_io.xml_anha_reader(self.ahxml, my_atoms)
        self.SC_mat = SC_mat
        mySC = MySC
###########################        
        total_coefs = len(coeff)
        tol_04 = 0.0001
        if self.missfit_strain:
            
            print(f'The strain for {my_atoms.get_chemical_formula()}  is ',self.voigt_strain)
            temp_voits = []
            strain_flag = []
            stain_flag_inp = []
            for ii,i in enumerate(self.voigt_strain):
                if abs(i) >= tol_04:
                    strain_flag.append(True)
                    temp_voits.append(ii+1)
                else:
                    strain_flag.append(False)
                strain_flag = stain_flag_inp
                
            if any(strain_flag): 
                myxml_clss.set_tags()
                my_tags = myxml_clss.tags
                new_coeffs, new_trms = missfit_terms.get_missfit_terms(
                    coeff, trms, my_tags, self.voigt_strain,  Higher_order_strain=self.Higher_order_strain,voigts=temp_voits)
                for ntrm_cntr in range(len(new_coeffs)):
                    trms.append(new_trms[ntrm_cntr])
                    coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]
                print(f'number of Missfit Coeffiecinets for this  {my_atoms.get_chemical_formula()}  is {len(new_coeffs)}')

                total_coefs = len(coeff)
                myxml_clss.get_ela_cons()
                new_coeffs, new_trms = missfit_terms.get_elas_missfit(myxml_clss.ela_cons,self.voigt_strain)
                print(f'Creating elastic terms for missfit strain for  {my_atoms.get_chemical_formula()}  : # of terms is {len(new_coeffs)}')
                for ntrm_cntr in range(len(new_coeffs)):
                    trms.append(new_trms[ntrm_cntr])
                    coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]

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
                                    ind_an = tools.find_index(CPOS, catm_an)
                                    ind_bn = tools.find_index(CPOS, catm_bn)
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
                                            red_pos = catm_bn[0]/ABC[0], catm_bn[1]/ABC[1], catm_bn[2]/ABC[2]
                                            tmp_par = np.zeros((3))
                                            for i, ii in enumerate(red_pos):
                                                if ii < 0:
                                                    tmp_par[i] = 1
     
                                            cell_b = '{} {} {}'.format(int(int(red_pos[0])-tmp_par[0]), int(
                                                int(red_pos[1])-tmp_par[1]), int(int(red_pos[2])-tmp_par[2]))
                                            
                                            disp_pos = (red_pos[0]-(int(red_pos[0])-tmp_par[0]))*ABC[0], (red_pos[1]-(int(
                                                red_pos[1])-tmp_par[1]))*ABC[1], (red_pos[2]-(int(red_pos[2])-tmp_par[2]))*ABC[2]
                                            
                                            ind_bn = tools.find_index(CPOS, disp_pos)
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
        xml_io.write_anh_xml(self.SC_coeff,self.SC_terms,fout)
 
if __name__ == '__main__':
    pass

