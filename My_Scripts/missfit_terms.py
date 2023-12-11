"""
Functions to calculate the missfit terms.
"""
import copy
import numpy as np


def find_str_phonon_coeffs( trms,Higher_order_strain=False):
   '''This function returns the list of ID of coefficients that have strain phonon coupling
   params:
   trms: list of all terms
   Higher_order_strain: if True, it will return the coefficients that have strain-strain coupling.
   '''
   str_phonon_coeffs = []
   nterm = 0
   for i in range(len(trms)):
      nstrain = int(trms[i][nterm][-1]['strain'])
      ndis = int(trms[i][nterm][-1]['dips'])
      if Higher_order_strain:
         if nstrain != 0:
               str_phonon_coeffs.append(i)
      else:
         if nstrain != 0 and ndis != 0:
               str_phonon_coeffs.append(i)
   return(str_phonon_coeffs)

def get_str_phonon_voigt( trms,  Higher_order_strain=False ,voigts=[1, 2, 3]):
   '''This function returns the number of coefficients that have strain phonon coupling
      same as find_str_phonon_coeffs but only for the voigts that are in the list voigts.
   '''
   str_phonon_coeffs = find_str_phonon_coeffs(trms,Higher_order_strain)
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

def get_new_str_terms( term,get_org=False):
   '''This function changes the term to a dictionary so that we can expand and multiply like polynomials
   it returns a list like : [[({'z': 1, 'c': 1}, 2)]] where we have in the dictionary different voigt strains 
   (x,y,z,xy ..) and their values as a,b,c,d ... and the power of the strain as the last element of the list 

   example:
   term=[[({'z': 1, 'c': 1}, 2)]]    i.e. (1*z+1*c)^2. where z is the strain in z direction and c is misfit strain in z direction.
   '''

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

def get_mult_coeffs( my_str_trms):
   """
   This function returns the multiplication of the terms in the list my_str_trms
   """
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

def get_shifted_terms( term, my_strain=[0, 0, 0]):
   """
   This function returns the terms by removing the terms already inside the model ."""
   not_shift = ['x', 'y', 'z']
   a, b, c = my_strain[0], my_strain[1], my_strain[2]
   my_mul_terms = get_mult_coeffs(
      get_new_str_terms(term))
   org_terms = get_mult_coeffs(
      get_new_str_terms(term,get_org=True))
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

def get_missfit_term( coeff, trms, my_tags, my_strain, voigts=[1, 2, 3]):
   """
   For one coefficient. 
   For each term, this function returns the missfit terms (not already in the model).
   """
   tol_wght = 10**-10
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
            my_terms = get_shifted_terms(
               my_term, my_strain)
            ndisp = int(my_term[-1]['dips'])
            # print(my_terms)
            if ndisp>0 :
               disp_text = get_disp_text(my_term,my_tags)
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

                  tmp_term = [*my_dis_term, *str_terms, {'dips': temp_ndisp, 'strain': num_str_temrs, 'distance': 0.0}]
                  if tmp_term not in my_str_phon_term[term_cnre] and abs((float(my_dis_term[0]['weight']) * my_terms[0][tmp_key]) > tol_wght):
                        my_str_phon_term[term_cnre].append(tmp_term)
                  elif  abs((float(my_dis_term[0]['weight']) * my_terms[0][tmp_key]) > tol_wght):
                        indx = my_str_phon_term[term_cnre].index(tmp_term)
                        ndisp_tmp = my_str_phon_term[term_cnre][indx][-1]['dips']
                        nstr_tmp = my_str_phon_term[term_cnre][indx][-1]['strain']
                        if ndisp_tmp ==0:
                           ndisp_tmp = 1
                        for i in range(ndisp_tmp):
                                 my_str_phon_term[term_cnre][indx][i]['weight'] += tmp_term[i]['weight']
                                    
               term_cnre += 1
            if i_term == 0 : #and (no_disp==False or num_str_temrs) > 0:
               temp_trms = re_order_terms(my_terms[0])
               key_cntr = 0
               for my_key in temp_trms.keys():
                  tot_nterms += 1
                  my_value = float(coeff['value']) 
                  # to follow the convention of the multibinit. 
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

def get_missfit_terms( coeff, terms, my_tags, my_strain, Higher_order_strain=False,voigts=[1, 2, 3]):
   """
   Loop over the coefficients and return the missfit terms for each coefficient.   
   """
   str_phonon_voigt = get_str_phonon_voigt(terms, Higher_order_strain=Higher_order_strain ,voigts=voigts)
   new_coeffs = []
   new_terms = []
   for icoeff in str_phonon_voigt:
      temp_coeffs,temp_terms = get_missfit_term(coeff[icoeff], terms[icoeff], my_tags, my_strain, voigts=[1, 2, 3])
      for cntr in range(len(temp_coeffs)):
            if len(temp_terms[cntr])>0:
               new_coeffs.append(temp_coeffs[cntr])
               new_terms.append(temp_terms[cntr])  
   return(new_coeffs,new_terms)

def get_disp_text(my_term,my_tags):
   """
   This function returns the text of the displacement term.
   """
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

def get_strain( strain=np.zeros((0,0))):
   """
   This function returns the voigt strain from the strain tensor.
   """
   voigt_str = [strain[0,0],strain[1,1],strain[2,2],(strain[1,2]+strain[2,1])/2,(strain[0,2]+strain[2,0])/2,(strain[0,1]+strain[1,0])/2]
   return(np.array(voigt_str))

def get_elas_missfit(ela_cnst,my_strain):
   """
   This function returns the elastic missfit terms.
   """
   tol_str = 10**-6
   # myxml_clss = xml_sys(self.xml)
   # myxml_clss.get_ela_cons()
   strten = np.zeros(6)
   new_coeffs = []
   new_terms = []
   my_vogt_dic = {1: 'eta_1', 2: 'eta_2', 3: 'eta_3', 4: 'eta_4', 5: 'eta_5', 6: 'eta_6'}
   # ela_cnst = (myxml_clss.ela_cons)   #np.linalg.det(self.SCMATS[])*
   tot_nterms = 1
   for alpha in my_vogt_dic.keys():
      for beta in my_vogt_dic.keys():

            strten[alpha-1] +=  ela_cnst[alpha-1,beta-1]*my_strain[beta-1] 

      # if strten[alpha-1] > tol_str:
      new_coeffs.append({'number': str(tot_nterms), 'value': str(strten[alpha-1]), 'text': my_vogt_dic[alpha]})
      new_term = [[{'weight': ' 1.000000'},{'power': ' 1', 'voigt': str(alpha)}, {'dips': 0, 'strain': 1, 'distance': 0}]]
      new_terms.append(new_term)
      tot_nterms += 1

   return(new_coeffs,new_terms)

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