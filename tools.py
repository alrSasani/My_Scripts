import numpy as np
from ase import Atoms
from ase.data import atomic_masses


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

def map_ordered_strctures(str_1,str_2,tol=0.5):  # the ordering is correct
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

def get_atom_num(atomic_mass, tol=0.1):
    if abs(atomic_mass-208) < 1:
        tol = 0.001
    for i in range(len(atomic_masses)):
        if abs(atomic_masses[i]-atomic_mass) < tol:
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

def sng_domain(atom,dim=[1,1,1],atm_to_move={'Ti':[0,0,0.02]}):
     red_corr = atom.get_scaled_positions()
     chem_sym = atom.get_chemical_symbols()
     nw_red = []
     for i_rcor,rcord in enumerate(red_corr):
          if chem_sym[i_rcor] in atm_to_move.keys():
              nw_red.append(rcord+atm_to_move[chem_sym[i_rcor]]) 
          else:
              nw_red.append(rcord)
     print(atom.get_cell())
     tmp_atms = Atoms(numbers=atom.get_atomic_numbers(), scaled_positions=nw_red, cell=atom.get_cell()) 
     new_SC = tmp_atms.repeat(dim)

     return(new_SC)

def find_tag_index(tags, tag):
    for i, ii in enumerate(tags):
        if tag[0] == ii[0] and tag[1] == ii[1]:
            return(i)

def get_atm_ij_diff_in_UC(STRC,uc_strcs,STRC_uc_cell):
    # STRC = self.mySC
    natom = len(STRC)
    # STRC_uc_cell = uc_strc.get_cell()
    tag_id = STRC.get_array('tag_id')
    indx_tag = []
    for i in range(natom):
        indx_tag.append(find_tag_index(
            uc_strcs[tag_id[i][1]].get_array('tag_id'), tag_id[i]))
                    
    atm_ij_diff_in_mat = np.zeros((natom, natom, 3))
    for i in range(natom):
        for j in range(natom):
            atm_ij_diff_in_mat[j, i] = np.dot(STRC_uc_cell, (uc_strcs[tag_id[i][1]].get_scaled_positions()[
                                                indx_tag[i]]-uc_strcs[tag_id[j][1]].get_scaled_positions()[indx_tag[j]]))            
    return(atm_ij_diff_in_mat)

def get_Uclls_in_STRC(STRC,uc_strcs,STRC_uc_cell_in=None):
    # STRC = self.mySC
    if STRC_uc_cell_in is not None:
        STRC_uc_cell = STRC_uc_cell_in
    else:
        STRC_uc_cell = uc_strcs['0'].get_cell()
    natom = len(STRC)
    STR_POSs = STRC.get_positions()
    atm_ij_diff_in_mat = get_atm_ij_diff_in_UC(STRC,uc_strcs,STRC_uc_cell)
    STRC_inv_uc_cell = np.linalg.inv(STRC_uc_cell)
    cells_vecs = np.zeros((natom, natom, 3))
    for atm_i in range(natom):
        for atm_j in range(natom):
            dists = STR_POSs[atm_i]-STR_POSs[atm_j] - \
                atm_ij_diff_in_mat[atm_j, atm_i]
            cells_vecs[atm_i, atm_j, :] = np.dot(
                (1/0.98)*STRC_inv_uc_cell, dists)
    return(cells_vecs)