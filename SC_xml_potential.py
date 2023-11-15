import numpy as np
from math import ceil
from ase import Atoms
from ase.units import Bohr
from ase.build import make_supercell,sort
import xml_io
import missfit_terms
from tools import find_index, mapping

###############################################################################

class Har_sc_maker():
    def __init__(self, xml_file, SC_mat,strain_in=np.zeros((3,3))):
        self.xml = xml_io.Xml_sys_reader(xml_file)
        self.xml.get_ase_atoms()
        self.my_atoms = self.xml.ase_atoms
        self.SC_mat = SC_mat
        self.set_SC(self.my_atoms,strain=strain_in)
        # self.mySC = make_supercell(self.my_atoms, self.SC_mat)
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

    def set_SC(self, tmp_atoms,strain=np.zeros((3,3))):
        mySC = make_supercell(tmp_atoms, self.SC_mat)
        cell = mySC.get_cell()+np.dot(strain,mySC.get_cell())
        self.mySC = Atoms(numbers= mySC.get_atomic_numbers(),scaled_positions=mySC.get_scaled_positions(),cell = cell)        
        self.mySC .set_array(
            'BEC', mySC.get_array('BEC'))
        self.mySC .set_array(
            'tag_id', mySC.get_array('tag_id'))
        self.mySC .set_array(
            'str_ph', mySC.get_array('str_ph'))
        
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
            my_atm_list = mapping(tmp_sc,self.mySC)
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

    def write_xml(self, out_put):
        xml_dta = {}
        if not self.has_FIN_FCDIC:
            self.reshape_FCDIC()
        self.xml.get_str_cp()
        self.xml.get_eps_inf()
        self.xml.get_ela_cons()
        self.xml.get_ref_energy()
        SC_FC = self.Fin_loc_FC
        # self.get_SC_BEC()
        self.xml.get_tot_forces()
        if self.has_tot_FC:
            keys = self.Fin_tot_FC.keys()
            xml_dta['SC_total_FC'] = self.Fin_tot_FC
            xml_dta['has_tot_FC'] = self.has_tot_FC            
        else:
            xml_dta['has_tot_FC'] = self.has_tot_FC 
            keys = SC_FC.keys()
        # self.get_SC_corr_forc()
        SCL_elas = ((self.xml.ela_cons)*self.SC_num_Uclls)

        xml_dta['keys'] = keys
        xml_dta['SCL_elas'] = SCL_elas
        xml_dta['SCL_ref_energy'] = self.SC_num_Uclls*self.xml.ref_eng
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
        xml_io.write_anh_xml(self.SC_coeff,self.SC_terms,fout)
 
if __name__ == '__main__':
    pass

