import numpy as np
from ase import Atoms

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

def mapping_old(tmp_scll,atoms_ref):
   tmp_scp = tmp_scll.get_scaled_positions()
   mscp = atoms_ref.get_scaled_positions()
   # for i in range(len(tmp_scp)):
   mstm_lst = []
   for i in range(len(tmp_scp)):
      ind = find_index(mscp, tmp_scp[i])
      if ind != -1:
            mstm_lst.append(ind)
      else:
            raise "Err: Could not map the structures! "
   return(mstm_lst)

def mapping(str_to_be_map, str_to_map_to,Ret_index=True):
   # str_to_map_to = self.mySC.get_scaled_positions()
   natom = len(str_to_map_to.get_scaled_positions())
   natom2 = len(str_to_be_map.get_scaled_positions())
   if natom != natom2:
      raise 'wrong structures'
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