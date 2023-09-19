import numpy as np
import xml.etree.ElementTree as ET
from math import ceil
from ase import Atoms
import ase
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses
from ase.build import make_supercell
from ase.io import write
from my_functions import *

###########################################################
##### some simple functions:

def str_mult(a,b):
    my_list = [*a.split(),*b.split()]
    my_list.sort()
    return(' '.join(my_list))


def terms_mult(T_1,T_2):
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
    if n-1!=0:
        return(terms_mult(get_pwr_N(T1,n-1),T1))
    else:
        return(T1)


def re_order_terms(T1):
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


def terms_comp(trms1,trms2):
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
                        disp.append({'term_1':i , 'term_2':j})
                    else:
                        strains.append({'term_1':i , 'term_2':j})
    return(disp,strains)

# This function is used to get atomic numbber for the mass of an atoms
def get_atom_num(atomic_mass):
    for i in range(len(ase.data.atomic_masses)):
        if abs(ase.data.atomic_masses[i]-atomic_mass)<0.1:
            mynum=i
    return(mynum)

# This functions is used when writing the xml file and convert a 2D array to text format
def to_text(arr):
    mytxt=''
    a=arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            if i==0 and j==0:
                mytxt=' {:.14E}'.format(arr[i][j])
            else:
                mytxt=mytxt+'  {:.14E}'.format(arr[i][j])
        mytxt=mytxt+'\n'
    return(mytxt)

# This functions is used when writing the xml file and convert a 1D array to text format
def one_text(arr):
    mytxt=''
    a=len(arr)
    for i in range(a):
        if i==0:
            mytxt=' {:.14E}'.format(arr[i])
        else:
            mytxt=mytxt+'  {:.14E}'.format(arr[i])
    mytxt=mytxt+'\n'
    return(mytxt)

# This function is used to find the position and index of an atom in a structure (MCOR = positins of the atoms in the structure, Vec = position of the atoms to find its index )
def find_index(Mcor,Vec,tol=0.001):
    index=-1
    for m in range(len(Mcor)):
        flg=[]
        for v in range(len(Mcor[m])):
            diff=Mcor[m,v]-Vec[v]
            if abs(diff) < tol:
                flg.append(True)
            else:
                flg.append(False)
        if all (flg) :
            index=m
    return(index)

# This function is used to make superlattice of two structures as Atoms objec
def make_SL(a1,a2):
    cell_1=a1.get_cell()
    cell_2=a2.get_cell()
    cell_SL=[cell_1[0][0],cell_1[1][1],cell_1[2][2]+cell_2[2][2]]
    pos1=a1.get_positions()
    pos2=a2.get_positions()
    car_SL=[]
    for i in pos1:
        car_SL.append(i)
    for i in pos2:
        car_SL.append([i[0],i[1],i[2]+cell_1[2][2]])
    numbers1=a1.get_atomic_numbers()
    numbers2=a2.get_atomic_numbers()
    numbers_SL= [*numbers1, *numbers2]
    my_SL=Atoms(numbers=numbers_SL,positions=car_SL, cell=cell_SL, pbc=True)
    return(my_SL)

#####Harmonic xml reader:
#This class is used to reas harmonic xml file  and return the data needed for construction of the SL

class xml_sys():
    def __init__(self,xml_file):
        self.xml_file=xml_file
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()

    #atomic mass and Positions        >>>>>>>>             self.atm_pos          self.natm
    def get_atoms(self):
        self.natm=0
        self.atm_pos=[]
        for Inf in self.root.iter('atom'):
            mass=Inf.attrib['mass']
            #print(Inf.attrib['massunits'])
            pos = (Inf.find('position').text)
            chrgs = (Inf.find('borncharge').text)
            self.atm_pos.append([mass,pos,chrgs])
            self.natm+=1
        self.atm_pos=np.array(self.atm_pos)

    def get_BEC(self):
        self.get_atoms()
        atm_pos=self.atm_pos
        self.BEC={}
        for i in range(len(self.atm_pos)):
            brn_tmp=[float(j) for j in atm_pos[i][2].split()[:]]
            brn_tmp=np.reshape(brn_tmp,(3,3))
            self.BEC[i]=brn_tmp

    #number of cells in the super cell and local forces    >>>>  self.ncll
    def get_Per_clls(self):
        self.ncll=[0,0,0]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)

            if cll.split()[1]=='0' and cll.split()[2]=='0' :
                self.ncll[0]+=1
            if cll.split()[0]=='0' and cll.split()[2]=='0' :
                #print(cll)
                self.ncll[1]+=1
            if cll.split()[0]=='0' and cll.split()[1]=='0' :
                self.ncll[2]+=1

    #number of cells in the super cell and local forces    >>>>  self.ncll
    def get_tot_Per_clls(self):
        self.tot_ncll=[0,0,0]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            if cll.split()[1]=='0' and cll.split()[2]=='0' :
                self.tot_ncll[0]+=1
            if cll.split()[0]=='0' and cll.split()[2]=='0' :
                #print(cll)
                self.tot_ncll[1]+=1
            if cll.split()[0]=='0' and cll.split()[1]=='0' :
                self.tot_ncll[2]+=1

    # getting total forces    add a condition if exists   >>  self.tot_fc
    def get_tot_forces(self):
        self.has_tot_FC=0
        self.tot_fc={}
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt='{} {} {}'.format(cll.split()[0],cll.split()[1],cll.split()[2])
            self.tot_fc[cll_txt]=np.array([float(i) for i in data])
        if len(self.tot_fc)>2:
            self.has_tot_FC=1

    # getting local forces        add a condition if exists  >>>> self.loc_fc
    def get_loc_forces(self):
        self.loc_fc={}
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt='{} {} {}'.format(cll.split()[0],cll.split()[1],cll.split()[2])
            self.loc_fc[cll_txt]=np.array([float(i) for i in data])

    #refrence energy        >>>>     self.ref_eng
    def get_ref_energy(self):
        self.ref_eng=0
        for Inf in self.root.iter('energy'):
            self.ref_eng=float(Inf.text)

    #Unit cell        >>>>    self.prim_vects
    def get_prim_vects(self):
        self.prim_vects=np.zeros((3,3))
        for Inf in self.root.iter('unit_cell'):
            #print(Inf.attrib['units'])
            for i in range(3):
                self.prim_vects[i,:]=float((Inf.text).split()[3*i+0]),float((Inf.text).split()[3*i+1]),float((Inf.text).split()[3*i+2])
        self.prim_vects=np.array(self.prim_vects)

    #epsilon infinity        >>>>     self.eps_inf
    def get_eps_inf(self):
        self.eps_inf=np.zeros((3,3))
        for Inf in self.root.iter('epsilon_inf'):
            #print(Inf.attrib['units'])
            for i in range(3):
                self.eps_inf[i,:]=(Inf.text).split()[3*i+0],(Inf.text).split()[3*i+1],(Inf.text).split()[3*i+2]
        self.eps_inf=np.array(self.eps_inf)

    def comp_tot_and_loc_FC(self):
        self.get_loc_forces()
        self.get_tot_forces()
        if self.has_tot_FC:
            if self.loc_sc.keys() == tot_fc.keys():
                self.similar_cells=True
            else:
                self.similar_cells=False

    #elastic constants        >>>>   self.ela_cons
    def get_ela_cons(self):
        ndim=6
        self.ela_cons=np.zeros((6,6))
        for Inf in self.root.iter('elastic'):
            #print(Inf.attrib['units'])
            for i in range(6):
                self.ela_cons[i,:]=(Inf.text).split()[6*i+0],(Inf.text).split()[6*i+1],(Inf.text).split()[6*i+2],(Inf.text).split()[6*i+3],(Inf.text).split()[6*i+4],(Inf.text).split()[6*i+5]
        self.ela_cons=np.array(self.ela_cons)

    ##reading phonons data        >>>>     self.dymat     self.freq     self.qpoint
    def get_phonos(self):
        self.has_phonons=0
        self.freq=[]
        self.dymat=[]
        for Inf in self.root.iter('phonon'):
            data = (Inf.find('qpoint').text)
            self.qpoint=(data)
            data = (Inf.find('frequencies').text).split()
            self.freq.append([float(i) for i in data])
            data = (Inf.find('dynamical_matrix').text).split()
            self.dymat.append([float(i) for i in data])
        self.freq=np.reshape(self.freq,(self.natm,3))
        self.dymat=np.reshape(self.dymat,(3*self.natm,3*self.natm))
        if len(self.freq)>2:
            self.has_phonons=1

    # reading strain phonon data        >>>>      self.corr_forc       self.strain
    def get_str_cp(self):
        self.corr_forc={}
        self.strain={}
        for Inf in self.root.iter('strain_coupling'):
            voigt=float(Inf.attrib['voigt'])
            data = (Inf.find('strain').text)
            self.strain[voigt]=data           #np.reshape(([float(i) for i in data]),(1,9))
            data = (Inf.find('correction_force').text).split()
            self.corr_forc[voigt]=np.reshape(([float(i) for i in data]),(self.natm,3))

    # making ase_atoms object from the structure of the xml file
    def get_ase_atoms(self):
        self.get_prim_vects()
        self.get_atoms()
        atm_pos=self.atm_pos
        natm=self.natm
        car_pos=[]
        amu=[]
        for i in range(natm):
            car_pos.append([float(xx)*Bohr for xx in atm_pos[i][1].split()])
            amu.append(float(atm_pos[i][0]))
        atom_num=[get_atom_num(x) for x in amu ]
        #print(Bohr*self.prim_vects)
        car_pos=np.array(car_pos)
        self.ase_atoms = Atoms(numbers=atom_num,positions=car_pos, cell=Bohr*self.prim_vects, pbc=True)
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
                cntr_list[f'{my_symbols[i]}_cntr'] +=1
            else:
                cntr_list[f'{my_symbols[i]}_cntr'] = 1
            if cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i]>1:
                my_char = 1
            elif cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i]==1:
                my_char = ''
            else:
                my_char = cntr_list[f'{my_symbols[i]}_cntr']
            my_tags.append(f'{my_symbols[i]}{my_char}')
        self.tags = my_tags

    # this function return the local force constants in the xml file as dictionary
    def get_loc_FC_dic(self):
        self.loc_FC_dic={}
        self.get_ase_atoms()
        natm=self.ase_atoms.get_global_number_of_atoms()
        self.get_loc_forces()
        #lc_fc=self.loc_fc
        for key in self.loc_fc.keys():
            my_cell=[int(x) for x in key.split()]
            my_fc=np.reshape(self.loc_fc[key],(3*natm,3*natm))
            for atm_a in range(natm):
                for atm_b in range(natm):
                    FC_mat=np.zeros((3,3))
                    int_key='{}_{}_{}_{}_{}'.format(atm_a,atm_b,my_cell[0],my_cell[1],my_cell[2])
                    FC_mat=my_fc[atm_a*3:atm_a*3+3,atm_b*3:atm_b*3+3]
                    self.loc_FC_dic[int_key]=FC_mat

    # this function return the total force constants in the xml file as dictionary
    def get_tot_FC_dic(self):
        self.get_tot_forces()
        self.tot_FC_dic={}
        if self.has_tot_FC:
            self.get_ase_atoms()
            natm=self.ase_atoms.get_global_number_of_atoms()
            #tot_fc=self.tot_fc
            for key in self.tot_fc.keys():
                my_cell=[int(x) for x in key.split()]
                my_fc=np.reshape(self.tot_fc[key],(3*natm,3*natm))
                for atm_a in range(natm):
                    for atm_b in range(natm):
                        FC_mat=np.zeros((3,3))
                        int_key='{}_{}_{}_{}_{}'.format(atm_a,atm_b,my_cell[0],my_cell[1],my_cell[2])
                        FC_mat=my_fc[atm_a*3:atm_a*3+3,atm_b*3:atm_b*3+3]
                        self.tot_FC_dic[int_key]=FC_mat

    #number of cells in the super cell and local forces    >>>>  self.loc_cells
    def get_loc_cells(self):
        self.loc_cells=[[0,0],[0,0],[0,0]]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell=[int(x) for x in cll.split()]
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


    #number of cells in the super cell and local forces    >>>>  self.loc_cells
    def get_tot_cells(self):
        self.tot_cells=[[0,0],[0,0],[0,0]]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell=[int(x) for x in cll.split()]
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


###########################################################
##### interface harmonic potential generation: This class is given two xml file for two materilas for the interface two 3x3 matirixes(SC_mat1 and SC_mat2) which give the matirx for
##### one side of the SL (these should have SC_mat1[0,0]=SC_mat2[0,0] and SC_mat1[1,1]=SC_mat2[1,1] while SC_mat1[2,2] and SC_mat2[2,2] define the thickness of the each material
##### in two sides of the SL)

class har_interface():

    def __init__(self,xml_file1,SC_mat1,xml_file2,SC_mat2):
        self.xml1=xml_sys(xml_file1)
        self.xml2=xml_sys(xml_file2)
        self.SC_mat1 = SC_mat1
        self.SC_mat2 = SC_mat2
        self.xml1.get_ase_atoms()
        self.xml2.get_ase_atoms()
        self.UC_atoms1 = self.xml1.ase_atoms
        self.UC_atoms2 = self.xml2.ase_atoms
        self.mySC1 = make_supercell(self.UC_atoms1,self.SC_mat1)
        self.mySC2 = make_supercell(self.UC_atoms2,self.SC_mat2)
        self.SL = make_SL(self.mySC1,self.mySC2)
        self.SL_natom = self.SL.get_global_number_of_atoms()
        self.my_SC1_natom = self.mySC1.get_global_number_of_atoms()
        self.my_SC2_natom = self.mySC2.get_global_number_of_atoms()
        self.num_unit_clls1 = np.linalg.det(SC_mat1)
        self.num_unit_clls2 = np.linalg.det(SC_mat2)
        self.has_SC_FCDIC = False
        self.has_FIN_FCDIC = False

    def get_SC_FCDIC(self):
        print('model making')
        tmp_SC1=make_supercell(self.UC_atoms1,self.SC_mat2)
        SL1=make_SL(self.mySC1,tmp_SC1)
        ABC_SL1=SL1.cell.cellpar()[0:3]
        CPOS_SL1=SL1.get_positions()
        tmp_SC2=make_supercell(self.UC_atoms2,self.SC_mat1)
        SL2=make_SL(tmp_SC2,self.mySC2)
        ABC_SL2=SL2.cell.cellpar()[0:3]
        CPOS_SL2=SL2.get_positions()
        self.xml1.get_loc_FC_dic()
        loc_FCDICT1=self.xml1.loc_FC_dic
        self.xml2.get_loc_FC_dic()
        loc_FCDICT2=self.xml2.loc_FC_dic
        self.xml1.get_tot_FC_dic()
        tot_FCDICT1=self.xml1.tot_FC_dic
        self.xml2.get_tot_FC_dic()
        tot_FCDICT2=self.xml2.tot_FC_dic
        abc1=self.UC_atoms1.cell.cellpar()[0:3]
        abc2=self.UC_atoms2.cell.cellpar()[0:3]
        natm_UC1=self.UC_atoms1.get_global_number_of_atoms()
        natm_UC2=self.UC_atoms2.get_global_number_of_atoms()
        self.xml1.get_loc_cells()
        self.xml1.get_tot_cells()
        temp_loc_keys1=self.xml1.loc_cells
        temp_tot_keys1=self.xml1.tot_cells
        self.xml2.get_loc_cells()
        self.xml2.get_tot_cells()
        temp_loc_keys2=self.xml2.loc_cells
        temp_tot_keys2=self.xml2.tot_cells
        ##############################
        minxl=min(ceil(temp_loc_keys1[0][0]/self.SC_mat1[0][0]),ceil(temp_loc_keys2[0][0]/self.SC_mat1[0][0]),ceil(temp_tot_keys1[0][0]/self.SC_mat1[0][0]),ceil(temp_tot_keys2[0][0]/self.SC_mat1[0][0]))
        maxxl=max(ceil(temp_loc_keys1[0][1]/self.SC_mat1[0][0]),ceil(temp_loc_keys2[0][1]/self.SC_mat1[0][0]),ceil(temp_tot_keys1[0][1]/self.SC_mat1[0][0]),ceil(temp_tot_keys2[0][1]/self.SC_mat1[0][0]))
        minyl=min(ceil(temp_loc_keys1[1][0]/self.SC_mat1[1][1]),ceil(temp_loc_keys2[1][0]/self.SC_mat1[1][1]),ceil(temp_tot_keys1[1][0]/self.SC_mat1[1][1]),ceil(temp_tot_keys2[1][0]/self.SC_mat1[1][1]))
        maxyl=max(ceil(temp_loc_keys1[1][1]/self.SC_mat1[1][1]),ceil(temp_loc_keys2[1][1]/self.SC_mat1[1][1]),ceil(temp_tot_keys1[1][1]/self.SC_mat1[1][1]),ceil(temp_tot_keys2[1][1]/self.SC_mat1[1][1]))
        minzl=min(ceil(temp_loc_keys1[2][0]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_loc_keys2[2][0]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_tot_keys1[2][0]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_tot_keys2[2][0]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])))
        maxzl=max(ceil(temp_loc_keys1[2][1]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_loc_keys2[2][1]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_tot_keys1[2][1]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])),ceil(temp_tot_keys2[2][1]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])))
        ###############################
        self.loc_SC_FCDIC={}
        self.tot_SC_FCDIC={}
        self.tot_mykeys=[]
        self.loc_mykeys=[]
        int_atm_list, fnl_layer = self.find_intrfc_atms()
        avg_atms = [*int_atm_list, *fnl_layer]
        for prd1 in range (minxl-1,maxxl+1):
            for prd2 in range (minyl-1,maxyl+1):
                for prd3 in range (minzl-1,maxzl+1):
                    SC_cell='{} {} {}'.format(prd1,prd2,prd3)
                    for atm_i in range(self.SL_natom):
                        for atm_j in range(self.SL_natom):
                            int_mat_1=False
                            int_mat_2=False
                            AVG = False
                            W_AVG = False
                            midl_lyr=False
                            i_midl_lyr_atm = False
                            j_midl_lyr_atm = False
                            midl_lyr_atm = False
                            Asite = False

                            if atm_i in avg_atms or atm_j in avg_atms:
                                midl_lyr_atm = True
                                if (atm_i in int_atm_list and atm_j in int_atm_list) or (atm_i in fnl_layer and atm_j in fnl_layer):
                                    midl_lyr= True

                                if not midl_lyr :
                                    if atm_i in avg_atms:
                                        i_midl_lyr_atm = True
                                    elif atm_j in avg_atms:
                                        j_midl_lyr_atm = True

                            ####
                            if atm_i < self.my_SC1_natom and atm_j < self.my_SC1_natom and not midl_lyr:
                                if midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_1=True
                                else:
                                    int_mat_1=True

                            if atm_i >= self.my_SC1_natom and atm_j >= self.my_SC1_natom  and not midl_lyr:
                                if midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_2=True
                                else:
                                    int_mat_2=True
                            ####

                            if atm_i < self.my_SC1_natom and atm_j >= self.my_SC1_natom  and not midl_lyr :
                                if not midl_lyr_atm:
                                    #int_mat_1=True
                                    AVG = True
                                    if atm_i%natm_UC1==0 and atm_j%natm_UC2==0:
                                        #Asite = True
                                        AVG = True
                                elif i_midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_2 = True

                                elif j_midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_1 = True

                            if atm_i >= self.my_SC1_natom and atm_j < self.my_SC1_natom  and not midl_lyr :
                                if not midl_lyr_atm:
                                    #int_mat_2=True
                                    AVG = True
                                    if atm_i%natm_UC2==0 and atm_j%natm_UC1==0:
                                        #Asite = True
                                        AVG = True
                                elif i_midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_1 = True

                                elif j_midl_lyr_atm:
                                    W_AVG = True
                                    int_mat_2 = True

                            if  midl_lyr:
                                int_mat_1 = True
                                AVG=True

                            if int_mat_1:
                                loc_FC_dic = loc_FCDICT1
                                tot_FC_dic = tot_FCDICT1
                                UC_natm = natm_UC1
                                ABC = ABC_SL1
                                CPOS = CPOS_SL1
                                abc = abc1
                                has_tot_FC=self.xml1.has_tot_FC

                            if int_mat_2:
                                loc_FC_dic = loc_FCDICT2
                                tot_FC_dic = tot_FCDICT2
                                UC_natm = natm_UC2
                                ABC = ABC_SL2
                                CPOS = CPOS_SL2
                                abc = abc2
                                has_tot_FC=self.xml2.has_tot_FC

                            if ((not AVG) and (not Asite) and (not W_AVG)):
                                dist=(prd1*ABC[0]+CPOS[int(atm_j/UC_natm)*5][0]-CPOS[int(atm_i/UC_natm)*5][0],
                                      prd2*ABC[1]+CPOS[int(atm_j/UC_natm)*5][1]-CPOS[int(atm_i/UC_natm)*5][1],
                                      prd3*ABC[2]+CPOS[int(atm_j/UC_natm)*5][2]-CPOS[int(atm_i/UC_natm)*5][2])
                                cell_b=(int(dist[0]/(abc[0]*0.95)),int(dist[1]/(abc[1]*0.95)),int(dist[2]/(abc[2]*0.95)))
                                UC_key='{}_{}_{}_{}_{}'.format(atm_i%UC_natm,atm_j%UC_natm,cell_b[0],cell_b[1],cell_b[2])
                                SC_key='{}_{}_{}_{}_{}'.format(atm_i,atm_j,prd1,prd2,prd3)

                                if UC_key in loc_FC_dic.keys():
                                    self.loc_SC_FCDIC[SC_key]=loc_FC_dic[UC_key]
                                    if SC_cell not in (self.loc_mykeys):
                                        self.loc_mykeys.append(SC_cell)
                                if has_tot_FC and UC_key in tot_FC_dic.keys():
                                    self.tot_SC_FCDIC[SC_key]=tot_FC_dic[UC_key]
                                    if SC_cell not in (self.tot_mykeys):
                                        self.tot_mykeys.append(SC_cell)

                            if (AVG and not Asite):
                                found_lf = False
                                found_tf = False

                                dist=(prd1*ABC[0]+CPOS[int(atm_j/UC_natm)*5][0]-CPOS[int(atm_i/UC_natm)*5][0],
                                      prd2*ABC[1]+CPOS[int(atm_j/UC_natm)*5][1]-CPOS[int(atm_i/UC_natm)*5][1],
                                      prd3*ABC[2]+CPOS[int(atm_j/UC_natm)*5][2]-CPOS[int(atm_i/UC_natm)*5][2])
                                cell_b=(int(dist[0]/(abc[0]*0.95)),int(dist[1]/(abc[1]*0.95)),int(dist[2]/(abc[2]*0.95)))
                                UC_key='{}_{}_{}_{}_{}'.format(atm_i%UC_natm,atm_j%UC_natm,cell_b[0],cell_b[1],cell_b[2])
                                SC_key='{}_{}_{}_{}_{}'.format(atm_i,atm_j,prd1,prd2,prd3)

                                temp_lint1 = np.zeros((3,3))
                                temp_lint2 = np.zeros((3,3))

                                if UC_key in loc_FCDICT1.keys():
                                    temp_lint1 = loc_FCDICT1[UC_key]
                                    found_lf = True

                                if UC_key in loc_FCDICT2.keys():
                                    temp_lint2 = loc_FCDICT2[UC_key]
                                    found_lf = True

                                if found_lf:
                                    self.loc_SC_FCDIC[SC_key]=(temp_lint1+temp_lint2)/2
                                    if SC_cell not in (self.loc_mykeys):
                                        self.loc_mykeys.append(SC_cell)

                                temp_tint2 = np.zeros((3,3))
                                temp_tint1 = np.zeros((3,3))

                                if UC_key in tot_FCDICT1.keys():
                                    temp_tint1 = tot_FCDICT1[UC_key]
                                    found_tf = True

                                if UC_key in tot_FCDICT2.keys():
                                    temp_tint2 = tot_FCDICT2[UC_key]
                                    found_tf = True

                                if found_tf:
                                    self.tot_SC_FCDIC[SC_key]=(temp_tint1+temp_tint2)/2
                                    if SC_cell not in (self.tot_mykeys):
                                        self.tot_mykeys.append(SC_cell)

                            if (W_AVG and not Asite):
                                found_lf = False
                                found_tf = False

                                dist=(prd1*ABC[0]+CPOS[int(atm_j/UC_natm)*5][0]-CPOS[int(atm_i/UC_natm)*5][0],
                                      prd2*ABC[1]+CPOS[int(atm_j/UC_natm)*5][1]-CPOS[int(atm_i/UC_natm)*5][1],
                                      prd3*ABC[2]+CPOS[int(atm_j/UC_natm)*5][2]-CPOS[int(atm_i/UC_natm)*5][2])
                                cell_b=(int(dist[0]/(abc[0]*0.95)),int(dist[1]/(abc[1]*0.95)),int(dist[2]/(abc[2]*0.95)))
                                UC_key='{}_{}_{}_{}_{}'.format(atm_i%UC_natm,atm_j%UC_natm,cell_b[0],cell_b[1],cell_b[2])
                                SC_key='{}_{}_{}_{}_{}'.format(atm_i,atm_j,prd1,prd2,prd3)

                                temp_lint1 = np.zeros((3,3))
                                temp_lint2 = np.zeros((3,3))
                                temp_lintt = np.zeros((3,3))

                                if UC_key in loc_FCDICT1.keys():
                                    temp_lint1 = loc_FCDICT1[UC_key]
                                    found_lf = True

                                if UC_key in loc_FCDICT2.keys():
                                    temp_lint2 = loc_FCDICT2[UC_key]
                                    found_lf = True

                                if UC_key in loc_FC_dic.keys():
                                    temp_lintt = loc_FC_dic[UC_key]
                                    found_lf = True


                                if found_lf:
                                    self.loc_SC_FCDIC[SC_key]=((temp_lint1+temp_lint2)/2+temp_lintt)/2
                                    if SC_cell not in (self.loc_mykeys):
                                        self.loc_mykeys.append(SC_cell)

                                temp_tintt = np.zeros((3,3))
                                temp_tint2 = np.zeros((3,3))
                                temp_tint1 = np.zeros((3,3))

                                if UC_key in tot_FCDICT1.keys():
                                    temp_tint1 = tot_FCDICT1[UC_key]
                                    found_tf = True

                                if UC_key in tot_FCDICT2.keys():
                                    temp_tint2 = tot_FCDICT2[UC_key]
                                    found_tf = True

                                if UC_key in tot_FC_dic.keys():
                                    temp_tintt = tot_FC_dic[UC_key]
                                    found_tf = True

                                if found_tf:
                                    self.tot_SC_FCDIC[SC_key]=((temp_tint1+temp_tint2)/2+temp_tintt)/2
                                    if SC_cell not in (self.tot_mykeys):
                                        self.tot_mykeys.append(SC_cell)

        self.has_SC_FCDIC=True

    def set_str(self,str_mat):
        xred=self.mySC.get_scaled_positions()
        cellp=self.mySC.get_cell()
        nmbrs=self.mySC.get_atomic_numbers()
        str_cell=str_mat*cellp+cellp
        new_SC=Atoms(numbers=nmbrs,scaled_positions=xred,cell=str_cell)
        self.mySC=new_SC

    def mapping(self,tmp_scll):
        mscp=self.SL.get_scaled_positions()
        tmp_scp=tmp_scll.get_scaled_positions()
        #for i in range(len(tmp_scp)):
        mstm_lst=[]
        for i in range(len(tmp_scp)):
            ind=find_index(mscp,tmp_scp[i])
            if ind!=-1:
                mstm_lst.append(ind)
        return(mstm_lst)

    def reshape_FCDIC(self,tmp_sc=0):
        if not self.has_SC_FCDIC:
            self.get_SC_FCDIC()

        self.Fin_loc_FC={}
        self.xml1.get_tot_forces()
        self.xml2.get_tot_forces()

        if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
            self.Fin_tot_FC={}
            my_keys=self.tot_mykeys
        else:
            my_keys=self.loc_mykeys

        if tmp_sc:
            self.my_atm_list=self.mapping(tmp_sc)
        else:
            self.my_atm_list=range(self.SL_natom)

        for my_key in (my_keys):
            loc_key_found = False
            tot_key_found = False
            my_cell=[int(x) for x in my_key.split()]
            tmp_loc_FC=np.zeros((3*self.SL_natom,3*self.SL_natom))
            tmp_tot_FC=np.zeros((3*self.SL_natom,3*self.SL_natom))
            cnt_a=0
            for atm_a in self.my_atm_list:
                cnt_b=0
                for atm_b in self.my_atm_list:
                    my_index='{}_{}_{}_{}_{}'.format(atm_a,atm_b,my_cell[0],my_cell[1],my_cell[2])
                    if my_index in self.loc_SC_FCDIC.keys():
                        loc_key_found = True
                        tmp_loc_FC[cnt_a*3:cnt_a*3+3,cnt_b*3:cnt_b*3+3]=self.loc_SC_FCDIC[my_index]
                    if my_index in self.tot_SC_FCDIC.keys() and (self.xml1.has_tot_FC or self.xml2.has_tot_FC ):
                        tot_key_found = True
                        tmp_tot_FC[cnt_a*3:cnt_a*3+3,cnt_b*3:cnt_b*3+3]=self.tot_SC_FCDIC[my_index]
                    cnt_b+=1
                cnt_a+=1

            if loc_key_found:
                self.Fin_loc_FC[my_key]=tmp_loc_FC
            if tot_key_found:
                 self.Fin_tot_FC[my_key]=tmp_tot_FC

        self.has_FIN_FCDIC=True

    def write_xml(self,out_put,asr=0,asr_chk=0):
        self.xml1.get_atoms()
        self.xml1.get_Per_clls()
        self.xml1.get_loc_forces()
        self.xml1.get_tot_forces()
        self.xml1.get_ref_energy()
        self.xml1.get_prim_vects()
        self.xml1.get_eps_inf()
        self.xml1.get_ela_cons()
        #self.xml1.get_phonos()
        self.xml1.get_str_cp()
        self.xml2.get_atoms()
        self.xml2.get_Per_clls()
        self.xml2.get_loc_forces()
        self.xml2.get_tot_forces()
        self.xml2.get_ref_energy()
        self.xml2.get_prim_vects()
        self.xml2.get_eps_inf()
        self.xml2.get_ela_cons()
        #self.xml2.get_phonos()
        self.xml2.get_str_cp()
        #ncll=self.xml2.ncll
        atm_pos1=self.xml1.atm_pos
        atm_pos2=self.xml2.atm_pos
        ########## Making super cell force constants.
        # number of supercells
        self.cal_eps_inf()
        self.SL_BEC_cal()

        natm_UC1=self.UC_atoms1.get_global_number_of_atoms()
        natm_UC2=self.UC_atoms2.get_global_number_of_atoms()
        SC_FC=self.Fin_loc_FC
        #tSC_FC=self.Fin_tot_FC

        if self.xml1.has_tot_FC or self.xml2.has_tot_FC :
            tSC_FC=self.Fin_tot_FC
            keys=tSC_FC.keys()
        else:
            keys=SC_FC.keys()

        my_ref_SL = self.ref_SL()
        lt_scll=my_ref_SL.get_cell()/Bohr
        atm_pos_scll=my_ref_SL.get_positions()/Bohr
        natm=self.SL_natom
        SC_phon=np.zeros((natm,3))
        SC_dmat=np.zeros((natm,natm))
        if asr:
            self.asr_impose()
            self.asr_chk()
        if asr_chk:
            self.asr_chk()
        SC_CorForc={}
        for k in range(len(self.xml1.corr_forc)):
            lst=[]
            for j in self.my_atm_list:
                if j < self.my_SC1_natom:
                    lst.append(self.xml1.corr_forc[k][j%natm_UC1,:])
                elif j >= self.my_SC1_natom:
                    lst.append(self.xml2.corr_forc[k][j%natm_UC2,:])
            np.reshape(lst,(natm,3))
            SC_CorForc[k]=np.array(lst)
        SCL_elas=(self.num_unit_clls1*(self.xml1.ela_cons)+self.num_unit_clls2*(self.xml2.ela_cons))
        out_xml=open(out_put,'w')
        out_xml.write('<?xml version="1.0" ?>\n')
        out_xml.write('<System_definition>\n')
        out_xml.write('  <energy>\n  {:.14E}\n  </energy>\n'.format(self.num_unit_clls1*self.xml1.ref_eng+self.num_unit_clls2*self.xml2.ref_eng))  #multiply
        out_xml.write('  <unit_cell units="bohrradius">\n {}  </unit_cell>\n'.format(to_text(lt_scll))) #multiply
        out_xml.write('  <epsilon_inf units="epsilon0">\n  {}  </epsilon_inf>\n'.format(to_text((self.SL_eps_inf))))
        out_xml.write('  <elastic units="hartree">\n  {}  </elastic>\n'.format(to_text(SCL_elas)))
        for i,ii in enumerate(self.my_atm_list):
            if ii < self.my_SC1_natom:
                out_xml.write('  <atom mass="  {}" massunits="atomicmassunit">\n    <position units="bohrradius">\n   {}</position>\n    <borncharge units="abs(e)">\n   {}</borncharge>\n  </atom>\n'.format(atm_pos1[ii%5][0],one_text(atm_pos_scll[ii,:]),to_text(self.SL_BEC[i])))

            else:
                out_xml.write('  <atom mass="  {}" massunits="atomicmassunit">\n    <position units="bohrradius">\n   {}</position>\n    <borncharge units="abs(e)">\n   {}</borncharge>\n  </atom>\n'.format(atm_pos2[ii%5][0],one_text(atm_pos_scll[ii,:]),to_text(self.SL_BEC[i])))

        for key in keys:
            if key in (SC_FC.keys()):
                out_xml.write('  <local_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </local_force_constant>\n'.format(to_text((SC_FC[key])),key))
            if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                out_xml.write('  <total_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </total_force_constant>\n'.format(to_text((tSC_FC[key])),key))

       # xml.write('  <phonon>\n    <qpoint units="2pi*G0">  {}</qpoint>\n    <frequencies units="reciprocal cm">\n  {}    </frequencies>\n    <dynamical_matrix units="hartree/bohrradius**2">\n {}    </dynamical_matrix>\n   </phonon>\n'.format(self.xml.qpoint,to_text(SC_phon),to_text(SC_dmat)))
        for i in range(len(self.xml1.strain)):
            out_xml.write('  <strain_coupling voigt=" {}">\n    <strain>  {}    </strain>\n    <correction_force units="hartree/bohrradius">\n  {}    </correction_force>\n  </strain_coupling>\n'.format(i,(self.xml1.strain[i]),to_text(SC_CorForc[i])))
        out_xml.write('</System_definition>')
        out_xml.close()

    def find_intrfc_atms(self):
        mdl_layer=[]
        CPOS_SC1=self.mySC1.get_positions()
        zmax=0
        for i in range(len(CPOS_SC1)):
            if zmax <= CPOS_SC1[i,2]:
                zmax=CPOS_SC1[i,2]
        tmp_SC1=make_supercell(self.UC_atoms1,self.SC_mat2)
        SL1=make_SL(self.mySC1,tmp_SC1)
        CPOS_SL1=SL1.get_positions()
        for i in range(len(CPOS_SL1)):
            if abs(CPOS_SL1[i,2]-zmax) < 0.001:
                mdl_layer.append(i)
        finl_layer=[]
        zmax=0
        for i in range(len(CPOS_SL1)):
            if zmax <= CPOS_SL1[i,2]:
                zmax=CPOS_SL1[i,2]

        for i in range(len(CPOS_SL1)):
            if abs(CPOS_SL1[i,2]-zmax) < 0.001:
                finl_layer.append(i)

        return(mdl_layer,finl_layer)

    def asr_impose(self):
        #print('ASR imposing')
        for atm_i in range(len(self.my_atm_list)):
            #asr_sr=np.zeros((3,3))
            asr_tot=np.zeros((3,3))
            for atm_j in range(len(self.my_atm_list)):
                for key_j in self.Fin_tot_FC.keys():
                    #asr_sr+=self.Fin_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                    if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                        asr_tot+=self.Fin_tot_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
            if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                self.Fin_tot_FC['0 0 0'][3*atm_i:3*atm_i+3,3*atm_i:3*atm_i+3]-=asr_tot
            #else:
                #self.Fin_FC['0 0 0'][3*atm_i:3*atm_i+3,3*atm_i:3*atm_i+3]-=asr_sr

    def asr_chk(self):
        print('ASR chking')
        if 1:
            for atm_i in range(len(self.my_atm_list)):
                asr_sr=np.zeros((3,3))
                asr_tot=np.zeros((3,3))
                for atm_j in range(len(self.my_atm_list)):
                    for key_j in self.Fin_tot_FC.keys():
                        #asr_sr+=self.Fin_loc_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                        if self.xml1.has_tot_FC or self.xml2.has_tot_FC:
                            asr_tot+=self.Fin_tot_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                print('Total')
                print(asr_tot)
                #print('SR')
                #print(asr_sr)

    def asr_intfce(self,a,b):
        #print('ASR imposing')
        for atm_i in range(len(self.my_atm_list)):
            asr_tot=np.zeros((3,3))
            for atm_j in range(len(self.my_atm_list)):
                for key_j in self.Fin_FC.keys():
                    #asr_sr+=self.Fin_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
                    if self.xml.has_tot_FC:
                        asr_tot+=self.Fin_tot_FC[key_j][3*atm_i:3*atm_i+3,3*atm_j:3*atm_j+3]
            if  self.xml.has_tot_FC:
                self.Fin_tot_FC['0 0 0'][3*a:3*a+3,3*b:3*b+3]-=asr_tot
            #else:
                #self.Fin_FC['0 0 0'][3*atm_i:3*atm_i+3,3*atm_i:3*atm_i+3]-=asr_sr

    def SL_BEC_cal(self):
        self.cal_eps_inf()
        int_atm_list, fnl_layer = self.find_intrfc_atms()
        intr_atoms =[*int_atm_list, *fnl_layer]
        eps_SL = self.SL_eps_inf
        eps_1 = self.xml1.eps_inf
        eps_2 = self.xml2.eps_inf
        SCL_1 = [np.sqrt(eps_SL[0][0]/eps_1[0][0]),np.sqrt(eps_SL[1][1]/eps_1[1][1]),np.sqrt(eps_SL[2][2]/eps_1[2][2])]
        SCL_2 = [np.sqrt(eps_SL[0][0]/eps_2[0][0]),np.sqrt(eps_SL[1][1]/eps_2[1][1]),np.sqrt(eps_SL[2][2]/eps_2[2][2])]
        BORN1=[]
        BORN2=[]
        for i in range(len(self.xml1.atm_pos)):
            brn_tmp1=[float(j) for j in self.xml1.atm_pos[i][2].split()[:]]
            brn_tmp1=np.array(brn_tmp1)
            brn_tmp1=np.reshape(brn_tmp1,(3,3))
            BORN1.append(brn_tmp1)
            brn_tmp2=[float(j) for j in self.xml2.atm_pos[i][2].split()[:]]
            brn_tmp2=np.array(brn_tmp2)
            brn_tmp2=np.reshape(brn_tmp2,(3,3))
            BORN2.append(brn_tmp2)

        SL_BEC = np.zeros((self.SL_natom,3,3))
        for i in range(self.SL_natom):
            if i < self.my_SC1_natom and i not in (intr_atoms):
                SL_BEC[i,:,:] = SCL_1*BORN1[i%5]
            elif i >= self.my_SC1_natom and i not in (intr_atoms):
                SL_BEC[i,:,:] = SCL_2*BORN2[i%5]
            elif i in  (intr_atoms):
                SL_BEC[i,:,:] = 0.5*(SCL_1*BORN1[i%5]+SCL_2*BORN2[i%5])
        self.SL_BEC = SL_BEC

    def cal_eps_inf(self):
        self.xml1.get_eps_inf()
        self.xml2.get_eps_inf()
        self.SL_eps_inf = np.zeros((3,3))
        l_1 = self.SC_mat1[2][2]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])
        l_2 = self.SC_mat2[2][2]/(self.SC_mat1[2][2]+self.SC_mat2[2][2])
        eps_xx = l_1*self.xml1.eps_inf[0][0]+l_2*self.xml2.eps_inf[0][0]
        eps_yy = l_1*self.xml1.eps_inf[1][1]+l_2*self.xml2.eps_inf[1][1]
        eps_zz = 1/(l_1*(1/self.xml1.eps_inf[2][2])+l_2*(1/self.xml2.eps_inf[2][2]))

        self.SL_eps_inf[0,0] = eps_xx
        self.SL_eps_inf[1,1] = eps_yy
        self.SL_eps_inf[2,2] = eps_xx    ####

    def ref_SL(self):  ### ref structure according to thesis of Carlos***
        self.SC_mat1
        self.SC_mat2
        self.xml1.get_ela_cons()
        ELC1 = self.xml1.ela_cons
        self.xml2.get_ela_cons()
        ELC2 = self.xml2.ela_cons

        tmp_SC1=make_supercell(self.UC_atoms1,self.SC_mat2)
        SL1=make_SL(self.mySC1,tmp_SC1)
        ABC_SL1=SL1.cell.cellpar()[0:3]
        ScPOS_SL1=SL1.get_scaled_positions()
        SL1_cell = SL1.get_cell()

        tmp_SC2=make_supercell(self.UC_atoms2,self.SC_mat1)
        SL2=make_SL(tmp_SC2,self.mySC2)
        ABC_SL2=SL2.cell.cellpar()[0:3]
        ScPOS_SL2=SL2.get_scaled_positions()
        SL2_cell = SL2.get_cell()

        cell_1=self.UC_atoms1.get_cell()
        cell_2=self.UC_atoms2.get_cell()

        p1 = cell_1[2][2]/(cell_1[2][2]+cell_2[2][2])
        p2 = cell_2[2][2]/(cell_1[2][2]+cell_2[2][2])
        m = np.zeros((3))
        for indx in range(3):
            m[indx] = p1*ELC1[indx][indx]/(p2*ELC2[indx][indx])

        a_avg = np.zeros((3))
        for a_indx in range(3):
            a_avg[a_indx] = cell_1[a_indx][a_indx]*cell_2[a_indx][a_indx]*(m[a_indx]*cell_1[a_indx][a_indx]+cell_2[a_indx][a_indx])/(m[a_indx]*cell_1[a_indx][a_indx]**2+cell_2[a_indx][a_indx]**2)

        numbers1=self.mySC1.get_atomic_numbers()
        numbers2=self.mySC2.get_atomic_numbers()
        numbers_SL= [*numbers1, *numbers2]

        a0 = self.SC_mat1[0][0]*a_avg[0]
        a1 = self.SC_mat1[1][1]*a_avg[1]
        a2 = self.SC_mat1[2][2]*a_avg[2]+self.SC_mat2[2][2]*a_avg[2]
        cell = [a0,a1,a2]
        my_SL=Atoms(numbers=numbers_SL,scaled_positions=ScPOS_SL1, cell = SL1_cell , pbc=True)
        return(my_SL)

###########################################################
##### interface Anharmonic potential generation:

class anh_intrface():

    def __init__(self,har_xml1,anh_xml1,SC_mat1,har_xml2,anh_xml2,SC_mat2,pordc_str_trms=True):
        self.har_xml1 = xml_sys(har_xml1)
        self.anh_xml1 = anh_xml1
        self.SC_mat1 = SC_mat1
        self.har_xml2 = xml_sys(har_xml2)
        self.anh_xml2 = anh_xml2
        self.SC_mat2 = SC_mat2
        self.har_xml1.get_ase_atoms()
        self.har_xml2.get_ase_atoms()
        self.my_atoms1 = self.har_xml1.ase_atoms
        self.my_atoms2 = self.har_xml2.ase_atoms
        self.mySC1 = make_supercell(self.my_atoms1,self.SC_mat1)
        self.mySC2 = make_supercell(self.my_atoms2,self.SC_mat2)
        tmp_SC1 = make_supercell(self.my_atoms1,self.SC_mat2)
        tmp_SC2 = make_supercell(self.my_atoms2,self.SC_mat1)
        self.SL1 = make_SL(self.mySC1,tmp_SC1)
        self.SL1_has_coeff = False
        self.SL2 = make_SL(tmp_SC2,self.mySC2)
        self.SL2_has_coeff = False
        self.new_str_rtms = pordc_str_trms
        har_xml_int = har_interface(har_xml1,self.SC_mat1,har_xml2,self.SC_mat2)
        self.ref_cell = har_xml_int.ref_SL()
        self.strain_tol = 1.0*10**(-6)
        self.mdl_lyr_atms, self.fnl_lyr_atms = har_xml_int.find_intrfc_atms()

    def SC_trms(self,har_xml,anh_xml,MySL,l = 0):
        mdl_lyr_atms, fnl_lyr_atms = self.mdl_lyr_atms, self.fnl_lyr_atms #self.find_intrfc_atms()
        myxml_clss = har_xml
        myxml_clss.get_ase_atoms()
        my_atoms = myxml_clss.ase_atoms
        coeff,trms = self.xml_anha(anh_xml,my_atoms)
        mySC = MySL
        CPOS = mySC.get_positions()
        ABC = mySC.cell.cellpar()[0:3]
        cPOS = my_atoms.get_positions()
        abc = my_atoms.cell.cellpar()[0:3]
        total_coefs = len(coeff)
        my_strain = self.get_lat_mismatch(l)
        strain_flag = []
        for i in my_strain:
            if abs(i) >= self.strain_tol:
                strain_flag.append(True)
            else:
                strain_flag.append(False)

        if self.new_str_rtms and any(strain_flag):
            myxml_clss.set_tags()
            my_tags = myxml_clss.tags
            #str_cnst = 2*0.01*0.25
            #my_strain = [str_cnst*abc[0],str_cnst*abc[1],str_cnst*abc[2]]
            my_voigts = [1,2,3]
            new_coeffs, new_trms = self.get_final_coeffs(coeff,trms,my_atoms,my_tags,my_strain ,voigts=my_voigts)
            for ntrm_cntr in range(len(new_coeffs)):
                trms.append(new_trms[ntrm_cntr])
                coeff[total_coefs+ntrm_cntr] = new_coeffs[ntrm_cntr]

        my_terms = []
        intrface_terms = []
        if l == 0:
            zdirec1 = range(self.SC_mat1[2][2])
            zdirec2 = range(self.SC_mat1[2][2])
        else:
            zdirec1 = range(self.SC_mat1[2][2],self.SC_mat1[2][2]+self.SC_mat2[2][2]+1)
            zdirec2 = range(self.SC_mat1[2][2]-1,self.SC_mat1[2][2]+self.SC_mat2[2][2]+1)
        for cc in range(len(coeff)):
            my_terms.append([])
            intrface_terms.append([])
            my_SAT_terms = trms[cc]
            for tc in range(len(my_SAT_terms)):
                for prd1 in range((self.SC_mat1[0][0])):
                    for prd2 in range((self.SC_mat1[1][1])):
                        for prd3 in zdirec1:
                            for prd1p in range(self.SC_mat1[0][0]):
                                for prd2p in range(self.SC_mat1[1][1]):
                                    for prd3p in zdirec2:
                                        my_term = []
                                        disp_cnt = 0
                                        intrface_atm = False
                                        for disp in range(int(my_SAT_terms[tc][-1]['dips'])):
                                            atm_a = int(my_SAT_terms[tc][disp]['atom_a'])
                                            atm_b = int(my_SAT_terms[tc][disp]['atom_b'])
                                            cell_a0 = [int(x) for x in  my_SAT_terms[tc][disp]['cell_a'].split()]
                                            cell_b0 = [int(x) for x in my_SAT_terms[tc][disp]['cell_b'].split()]
                                            catm_a0 = cell_a0[0]*abc[0]+cPOS[atm_a][0],cell_a0[1]*abc[1]+cPOS[atm_a][1],cell_a0[2]*abc[2]+cPOS[atm_a][2]
                                            catm_b0 = cell_b0[0]*abc[0]+cPOS[atm_b][0],cell_b0[1]*abc[1]+cPOS[atm_b][1],cell_b0[2]*abc[2]+cPOS[atm_b][2]
                                            dst0 = [catm_a0[0]-catm_b0[0],catm_a0[1]-catm_b0[1],catm_a0[2]-catm_b0[2]]
                                            catm_an = prd1*abc[0]+catm_a0[0],prd2*abc[1]+catm_a0[1],prd3*abc[2]+catm_a0[2]
                                            catm_bn = prd1p*abc[0]+catm_b0[0],prd2p*abc[1]+catm_b0[1],prd3p*abc[2]+catm_b0[2]
                                            ind_an = find_index(CPOS,catm_an)
                                            ind_bn = find_index(CPOS,catm_bn)
                                            dst = [catm_an[0]-catm_bn[0],catm_an[1]-catm_bn[1],catm_an[2]-catm_bn[2]]
                                            dif_ds = np.zeros((3))
                                            for i in range(3):
                                                dif_ds[i] = abs(dst[i]-dst0[i])
                                            if (ind_an != -1) and  all(dif_ds < 0.001):
                                                if ind_bn == -1:
                                                    red_pos = catm_bn[0]/ABC[0],catm_bn[1]/ABC[1],catm_bn[2]/ABC[2]
                                                    tmp_par = np.zeros((3))
                                                    for i,ii in enumerate(red_pos):
                                                        if ii < 0:
                                                            tmp_par[i] = 1
                                                    cell_b = '{} {} {}'.format(int(int(red_pos[0])-tmp_par[0]),int(int(red_pos[1])-tmp_par[1]),int(int(red_pos[2])-tmp_par[2]))
                                                    disp_pos = (red_pos[0]-(int(red_pos[0])-tmp_par[0]))*ABC[0],(red_pos[1]-(int(red_pos[1])-tmp_par[1]))*ABC[1],(red_pos[2]-(int(red_pos[2])-tmp_par[2]))*ABC[2]
                                                    ind_bn = find_index(CPOS,disp_pos)
                                                else:
                                                    cell_b = '0 0 0'
                                                new_dis = {'atom_a':ind_an,'cell_a':'0 0 0','atom_b':ind_bn,'cell_b':cell_b,'direction':my_SAT_terms[tc][disp]['direction'],'power':my_SAT_terms[tc][disp]['power'], 'weight':my_SAT_terms[tc][disp]['weight']}

                                                if (ind_an in [*mdl_lyr_atms, *fnl_lyr_atms] or ind_bn in [*mdl_lyr_atms, *fnl_lyr_atms]) and cc < total_coefs :
                                                    intrface_atm = True
                                                my_term.append(new_dis)
                                                disp_cnt += 1
                                        if (int(my_SAT_terms[tc][-1]['dips']) == 0 or (disp_cnt == int(my_SAT_terms[tc][-1]['dips']) and (len(my_term) != 0))):
                                            tmp_d = 0
                                            if disp_cnt == 0:
                                                tmp_d = 1
                                            for str_cnt in range(int(my_SAT_terms[tc][-1]['strain'])):
                                                my_term.append({'power': my_SAT_terms[tc][disp_cnt+tmp_d+str_cnt]['power'] , 'voigt': my_SAT_terms[tc][disp_cnt+tmp_d+str_cnt]['voigt']})

                                        if len(my_term) == int(my_SAT_terms[tc][-1]['dips'])+int(my_SAT_terms[tc][-1]['strain']):
                                            if (int(my_SAT_terms[tc][-1]['dips']) == 0 and int(my_SAT_terms[tc][-1]['strain']) != 0):
                                                my_term.append({'weight': my_SAT_terms[tc][0]['weight'] })
                                            my_term.append(my_SAT_terms[tc][-1])

                                            if my_term not in(my_terms[cc]) and my_term not in(intrface_terms[cc]):
                                                if intrface_atm:
                                                    intrface_terms[cc].append(my_term)
                                                else:
                                                    my_terms[cc].append(my_term)

        if l == 0 :
            self.SL1_has_coeff = True
            self.SL1_coeff = coeff
            self.SL1_terms = my_terms
            self.interface_int1 = intrface_terms
        if l == 1 :
            self.SL2_has_coeff = True
            self.SL2_coeff = coeff
            self.SL2_terms = my_terms
            self.interface_int2 = intrface_terms

    def get_SATs(self, my_term, l=0):
        if l==0:
            hxml = self.har_xml1
        elif l==1:
            hxml = self.har_xml2
        hxml.get_ase_atoms()
        hxml.get_str_cp()
        my_strain = hxml.strain
        my_atms = hxml.ase_atoms
        sds = spg.get_symmetry_dataset(my_atms)
        print(f"space group number = {sds['number']}  INT = {sds['international']} ")
        rot = sds['rotations']
        trans = sds['translations']
        vgt_mat=np.zeros((7,3,3))
        vgt_mat[1,:,:] = np.reshape([float(v) for v in my_strain[0].split()],(3,3))
        vgt_mat[2,:,:] = np.reshape([float(v) for v in my_strain[1].split()],(3,3))
        vgt_mat[3,:,:] = np.reshape([float(v) for v in my_strain[2].split()],(3,3))
        vgt_mat[4,:,:] = np.reshape([float(v) for v in my_strain[3].split()],(3,3))
        vgt_mat[5,:,:] = np.reshape([float(v) for v in my_strain[4].split()],(3,3))
        vgt_mat[6,:,:] = np.reshape([float(v) for v in my_strain[5].split()],(3,3))
        dir_dic = {'x':1,'y':2,'z':3}
        pos_prm = my_atms.get_positions()
        wrapPos = ase.geometry.wrap_positions
        cell_prm = my_atms.get_cell()
        my_trms=[]
        for my_rot in range(len(rot)):
            disp_lst=[]
            dis_tmp_weight = 1
            wrong_acell = False
            this_term = False
            for disp in range(my_term[-1]['dips']):
                atm_a = int(my_term[disp]['atom_a'])
                cell_a = [ int(cll) for cll in (my_term[disp]['cell_a'].split())]
                atm_b = int(my_term[disp]['atom_b'])
                cell_b = [ int(cll) for cll in (my_term[disp]['cell_b'].split())]
                power = my_term[disp]['power']
                vec_dir = dir_dic[my_term[disp]['direction']]
                rot_vec_dir =vgt_mat[vec_dir,:,:]
                rot_vec_dir = np.dot(np.dot(rot[my_rot],rot_vec_dir),(alg.inv(rot[my_rot])))

                for vgt_cnt in range(1,len(vgt_mat)):
                    diff = np.matrix.flatten(abs(rot_vec_dir) - vgt_mat[vgt_cnt,:,:])
                    if not any(diff):
                        rot_voit_key = vgt_cnt

                pos_a = pos_prm[atm_a] + np.dot(cell_a,cell_prm)
                pos_b = pos_prm[atm_b] + np.dot(cell_b,cell_prm)

                rot_pos_a = np.dot(rot[my_rot],pos_a)+np.dot(trans[my_rot],cell_prm)
                rot_pos_b = np.dot(rot[my_rot],pos_b)+np.dot(trans[my_rot],cell_prm)

                wrp_a = wrapPos([rot_pos_a], cell_prm)[0]
                wrp_b = wrapPos([rot_pos_b], cell_prm)[0]

                ncell_a0 = rot_pos_a - wrp_a
                ncell_b =  rot_pos_b - wrp_b  - ncell_a0
                ncell_a = [0,0,0] #ncell_a

                atm_a_ind = find_index(pos_prm, wrp_a ,tol=0.0001)
                atm_b_ind = find_index(pos_prm, wrp_b ,tol=0.0001)

                dst0 = [pos_a[0]-pos_b[0],pos_a[1]-pos_b[1],pos_a[2]-pos_b[2]]
                dst = [rot_pos_a[0]-rot_pos_b[0],rot_pos_a[1]-rot_pos_b[1],rot_pos_a[2]-rot_pos_b[2]]

                if abs(np.linalg.norm(dst0)-np.linalg.norm(dst)) > 0.00001:
                    wrong_acell = True
                    break

                dis_tmp_weight *= (rot[my_rot][rot_voit_key-1,vec_dir-1])**int(power)

                ndis = {
                    'atom_a':atm_a_ind,
                    'cell_a': f"{round(ncell_a[0]/cell_prm[0,0])} {round(ncell_a[1]/cell_prm[1,1])} {round(ncell_a[2]/cell_prm[2,2])}",
                    'atom_b':atm_b_ind,
                    'cell_b':f"{round(ncell_b[0]/cell_prm[0,0])} {round(ncell_b[1]/cell_prm[1,1])} {round(ncell_b[2]/cell_prm[2,2])}",
                    'direction':get_key(dir_dic,rot_voit_key),
                    'power':power
                    }
                disp_lst.append(ndis)
            tem_vogt_wgt = 1
            if not(wrong_acell):
                if my_term[-1]['dips'] != 0 and my_term[-1]['strain'] != 0 and len(disp_lst) > 0:
                    for strain in range(my_term[-1]['strain']):
                        voigt =int(my_term[disp+strain+1]['voigt'])
                        pwr_str = int(my_term[disp+strain+1]['power'])
                        vgt_trans = np.dot(np.dot(rot[my_rot],vgt_mat[voigt,:,:]),(alg.inv(rot[my_rot])))
                        for vgt_cnt in range(1,len(vgt_mat)):
                            diff = np.matrix.flatten(-(vgt_trans) - vgt_mat[vgt_cnt,:,:])
                            if not any(diff):
                                rot_voit_key = vgt_cnt
                                vogt_wgt = -1
                            diff = np.matrix.flatten((vgt_trans) - vgt_mat[vgt_cnt,:,:])
                            if not any(diff):
                                rot_voit_key = vgt_cnt
                                vogt_wgt = 1
                        tem_vogt_wgt *= vogt_wgt**int(pwr_str)
                        disp_lst.append({'power':f' {pwr_str}' , 'voigt':f' {rot_voit_key}'})

                for i in range(my_term[-1]['dips']):
                    disp_lst[i]['weight'] = " %2.6f" %(dis_tmp_weight*tem_vogt_wgt)
                if  len(disp_lst)>0:
                    disp_lst.append(my_term[-1])
                    found = False
                    prm_lst =list(permutations(disp_lst[:]))
                    for prm_mem in prm_lst:
                        my_temp = [prm for prm in prm_mem]
                        if my_temp in  my_trms :
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
                    vgt_trans = np.dot(np.dot(rot[my_rot],vgt_mat[voigt,:,:]),(alg.inv(rot[my_rot])))
                    for vgt_cnt in range(1,len(vgt_mat)):
                        diff = np.matrix.flatten(-(vgt_trans) - vgt_mat[vgt_cnt,:,:])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = -1
                        diff = np.matrix.flatten((vgt_trans) - vgt_mat[vgt_cnt,:,:])
                        if not any(diff):
                            rot_voit_key = vgt_cnt
                            vogt_wgt = 1
                    disp_lst.append({'power':f' {pwr_str}' ,'voigt':f' {rot_voit_key}'})
                tem_vogt_wgt *= vogt_wgt**int(pwr_str)
                disp_lst[0]['weight']=" %2.6f" %tem_vogt_wgt

                disp_lst.append(my_term[-1])
                prm_lst =list(permutations(disp_lst[:]))
                found = False
                for prm_mem in prm_lst:
                    my_temp = [prm for prm in prm_mem]
                    if my_temp in  my_trms :
                        found = True

                if disp_lst not in my_trms and len(disp_lst)>0 and not found:
                    my_trms.append(disp_lst)

        return(my_trms)

    def intrface_avg(self):
        mdl_lyr_atms, fnl_lyr_atms = self.mdl_lyr_atms, self.fnl_lyr_atms #self.find_intrfc_atms()
        avg_atoms = [*mdl_lyr_atms, *fnl_lyr_atms]
        intrface_terms1 = self.interface_int1.copy()
        intrface_terms2 = self.interface_int2.copy()

        sc1_natoms = self.mySC1.get_global_number_of_atoms()
        coeff1,trms1 = self.xml_anha(self.anh_xml1,self.my_atoms1)
        coeff2,trms2 = self.xml_anha(self.anh_xml2,self.my_atoms2)

        disp, strain = terms_comp(trms1,trms2)

        Sim_terms1 = []
        cntr=0
        for sim_coeff in (disp):
            Coeff_1 = sim_coeff['term_1']
            Coeff_2 = sim_coeff['term_2']
            C1 = float(coeff1[Coeff_1]['value'])
            C2 = float(coeff2[Coeff_2]['value'])

            for j in range(len(intrface_terms1[Coeff_1])):
                avg_cnst = 0
                for k in range(int(intrface_terms1[Coeff_1][j][-1]['dips'])):
                    atm_a = intrface_terms1[Coeff_1][j][k]['atom_a']
                    atm_b = intrface_terms1[Coeff_1][j][k]['atom_b']
                    if atm_a in avg_atoms:
                        avg_cnst += 0.5*C1+0.5*C2
                    elif atm_a < sc1_natoms:
                        avg_cnst += C1
                    elif atm_a >= sc1_natoms:
                        avg_cnst += C2

                    if atm_b in avg_atoms:
                        avg_cnst += 0.5*C1+0.5*C2
                    elif atm_b < sc1_natoms:
                        avg_cnst += C1
                    elif atm_b >= sc1_natoms:
                        avg_cnst += C2
                avg_cnst = (1/(2*int(intrface_terms1[Coeff_1][j][-1]['dips'])))*avg_cnst
                #print(f' C1 = {C1}  C2 = {C2}   AVG =  {avg_cnst}')
                Sim_terms1.append([coeff1[Coeff_1], avg_cnst, intrface_terms1[Coeff_1][j]])
                intrface_terms1[Coeff_1][j]=[]



        cntr=0
        for sim_coeff in (disp):
            Coeff_1 = sim_coeff['term_1']
            Coeff_2 = sim_coeff['term_2']
            C1 = float(coeff1[Coeff_1]['value'])
            C2 = float(coeff2[Coeff_2]['value'])

            for j in range(len(intrface_terms2[Coeff_2])):
                avg_cnst = 0
                for k in range(int(intrface_terms2[Coeff_2][j][-1]['dips'])):
                    atm_a = intrface_terms2[Coeff_2][j][k]['atom_a']
                    atm_b = intrface_terms2[Coeff_2][j][k]['atom_b']
                    if atm_a in avg_atoms:
                        avg_cnst += 0.5*C1+0.5*C2
                    elif atm_a < sc1_natoms:
                        avg_cnst += C1
                    elif atm_a >= sc1_natoms:
                        avg_cnst += C2

                    if atm_b in avg_atoms:
                        avg_cnst += 0.5*C1+0.5*C2
                    elif atm_b < sc1_natoms:
                        avg_cnst += C1
                    elif atm_b >= sc1_natoms:
                        avg_cnst += C2
                avg_cnst = (1/(2*int(intrface_terms2[Coeff_2][j][-1]['dips'])))*avg_cnst
                #print(f' C1 = {C1}  C2 = {C2}   AVG =  {avg_cnst}')
                Sim_terms1.append([coeff2[Coeff_2], avg_cnst, intrface_terms2[Coeff_2][j]])
                intrface_terms2[Coeff_2][j]=[]

        ### Group terms with similar coefficients
        #for tmp_trm1 in Sim_terms1:
            #for tmp_trm2 in Sim_terms1:
############


        int_terms1 = []
        cntr=0
        for i in range(len(coeff1)):
            for j in range(len(intrface_terms1[i])):
                #int_terms1.append([])
                avg_cnst = 0
                if len(intrface_terms1[i][j]) != 0:
                    for k in range(int(intrface_terms1[i][j][-1]['dips'])):
                        atm_a = intrface_terms1[i][j][k]['atom_a']
                        atm_b = intrface_terms1[i][j][k]['atom_b']
                        if atm_a in avg_atoms:
                            avg_cnst += 0.5
                        elif atm_a < sc1_natoms:
                            avg_cnst += 1
                        elif atm_a >= sc1_natoms:
                            avg_cnst += 0

                        if atm_b in avg_atoms:
                            avg_cnst += 0.5
                        elif atm_b < sc1_natoms:
                            avg_cnst += 1
                        elif atm_b >= sc1_natoms:
                            avg_cnst += 0
                    avg_cnst = (1/(2*int(intrface_terms1[i][j][-1]['dips'])))*avg_cnst
                    int_terms1.append([coeff1[i], avg_cnst, intrface_terms1[i][j]])
                    cntr+=1

        int_terms2 = []
        cntr=0
        for i in range(len(coeff2)):
            for j in range(len(intrface_terms2[i])):
                #int_terms2.append([])
                if len(intrface_terms2[i][j]) != 0:
                    avg_cnst = 0
                    for k in range(int(intrface_terms2[i][j][-1]['dips'])):
                        atm_a = intrface_terms2[i][j][k]['atom_a']
                        atm_b = intrface_terms2[i][j][k]['atom_b']
                        if atm_a in avg_atoms:
                            avg_cnst += 0.5
                        elif atm_a < sc1_natoms:
                            avg_cnst += 0
                        elif atm_a >= sc1_natoms:
                            avg_cnst += 1
                        if atm_b in avg_atoms:
                            avg_cnst += 0.5
                        elif atm_b < sc1_natoms:
                            avg_cnst += 0
                        elif atm_b >= sc1_natoms:
                            avg_cnst += 1

                    avg_cnst = (1/(2*int(intrface_terms2[i][j][-1]['dips'])))*avg_cnst
                    int_terms2.append([coeff2[i], avg_cnst, intrface_terms2[i][j]])
                    cntr+=1
        return(Sim_terms1,int_terms1,int_terms2)

    def xml_anha(self,fname,atoms):
        tree = ET.parse(fname)
        root = tree.getroot()
        coeff={}
        lst_trm=[]
        car_pos=atoms.get_positions()
        abc=atoms.cell.cellpar()[0:3]
        for cc , inf in enumerate (root.iter('coefficient')):
            coeff[cc]=inf.attrib
            lst_trm.append([])
            for tt, trm in enumerate (inf.iter('term')):
                lst_trm[cc].append([])
                #print(trm.attrib)
                disp_cnt=0
                strcnt=0
                for dd, disp in enumerate(trm.iter('displacement_diff')):
                    disp_cnt+=1
                    dta=disp.attrib
                    atma=int(disp.attrib['atom_a'])
                    atmb=int(disp.attrib['atom_b'])
                    cella = (disp.find('cell_a').text)
                    cellb = (disp.find('cell_b').text)
                    cell_a=[float(x) for x in cella.split()]
                    cell_b=[float(x) for x in cellb.split()]
                    #print(abc[0])

                    pos_a=[cell_a[0]*abc[0]+car_pos[atma,0],cell_a[1]*abc[1]+car_pos[atma,1],cell_a[2]*abc[2]+car_pos[atma,2]]
                    pos_b=[cell_b[0]*abc[0]+car_pos[atmb,0],cell_b[1]*abc[1]+car_pos[atmb,1],cell_b[2]*abc[2]+car_pos[atmb,2]]
                    dist=((pos_a[0]-pos_b[0])**2+(pos_a[1]-pos_b[1])**2+(pos_a[2]-pos_b[2])**2)**0.5
                    dta['cell_a']=cella
                    dta['cell_b']=cellb
                    dta['weight']=trm.attrib['weight']
                    lst_trm[cc][tt].append(dta)
                if disp_cnt==0:
                    #dta={}
                    dist=0
                    dta=trm.attrib
                    lst_trm[cc][tt].append(dta)
                for ss, strain in enumerate(trm.iter('strain')):
                    dta=strain.find('strain')
                    lst_trm[cc][tt].append(strain.attrib)
                    strcnt+=1

                lst_trm[cc][tt].append({'dips':disp_cnt,'strain':strcnt,'distance':dist})
        return(coeff,lst_trm)

    def wrt_anxml(self, fout, str_str=0):
        if self.SL1_has_coeff and self.SL2_has_coeff:
            coeff1,trms1=self.SL1_coeff,self.SL1_terms
            intrface_terms1 = self.interface_int1
            coeff2,trms2 = self.SL2_coeff, self.SL2_terms
            intrface_terms2 = self.interface_int2
        else:
            self.SC_trms(self.har_xml1, self.anh_xml1, self.SL1, l=0)
            self.SC_trms(self.har_xml2, self.anh_xml2, self.SL2, l=1)
            coeff1,trms1=self.SL1_coeff,self.SL1_terms
            coeff2,trms2 = self.SL2_coeff, self.SL2_terms
            intrface_terms1 = self.interface_int1
            intrface_terms2 = self.interface_int2

        l1 = self.SC_mat1[2][2]
        l2 = self.SC_mat2[2][2]

        output=open(fout, 'w')
        output.write('<?xml version="1.0" ?>\n')
        output.write('<Heff_definition>\n')
        #print(f'terms_1 =  {len(trms1)}   terms_2 = {len(trms1)}    intr_trms_1 = {len(intrface_terms1)}   inter_terms_2 = {len(intrface_terms2)}')

        str_coeff_1=[]
        str_coeff_2=[]
        coef_cntr = 1
        coef_cntr_int = 1
        for i in range(len(coeff1)):
            k=0
            for j in range(len(trms1[i])):
                for k in range(int(trms1[i][j][-1]['dips'])):
                    if int(trms1[i][j][-1]['dips']) != 0 and k==0 and j==0:
                        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coef_cntr,coeff1[i]['value'],coeff1[i]['text'],))
                    if (k == 0) and int(trms1[i][j][-1]['dips']) != 0:
                        output.write('    <term weight="{}">\n'.format(trms1[i][j][k]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(trms1[i][j][k]['atom_a'],trms1[i][j][k]['atom_b'],trms1[i][j][k]['direction'],trms1[i][j][k]['power']))
                    output.write('        <cell_a>{}</cell_a>\n'.format(trms1[i][j][k]['cell_a']))
                    output.write('        <cell_b>{}</cell_b>\n'.format(trms1[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')

                if int(trms1[i][j][-1]['dips']) !=0 :
                    for l in range(int(trms1[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms1[i][j][k+l+1]['power'],trms1[i][j][k+l+1]['voigt']))
                    output.write('    </term>\n')

            if len(trms1[i]) != 0:
                if int(trms1[i][0][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1
                else:
                    str_coeff_1.append(i)


        for i in range(len(coeff2)):
            k=0
            for j in range(len(trms2[i])):
                for k in range(int(trms2[i][j][-1]['dips'])):
                    if int(trms2[i][j][-1]['dips']) != 0 and k==0 and j==0:
                        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coef_cntr,coeff2[i]['value'],coeff2[i]['text'],))
                    if (k == 0) and int(trms2[i][j][-1]['dips']) != 0:
                        output.write('    <term weight="{}">\n'.format(trms2[i][j][k]['weight']))
                    output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(trms2[i][j][k]['atom_a'],trms2[i][j][k]['atom_b'],trms2[i][j][k]['direction'],trms2[i][j][k]['power']))
                    output.write('        <cell_a>{}</cell_a>\n'.format(trms2[i][j][k]['cell_a']))
                    output.write('        <cell_b>{}</cell_b>\n'.format(trms2[i][j][k]['cell_b']))
                    output.write('      </displacement_diff>\n')

                if int(trms2[i][j][-1]['dips']) !=0 :
                    for l in range(int(trms2[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms2[i][j][k+l+1]['power'],trms2[i][j][k+l+1]['voigt']))
                    output.write('    </term>\n')

            if len(trms2[i]) != 0:
                if int(trms2[i][0][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1
                else:
                    str_coeff_2.append(i)


##########################################
        sim_trms, int_1, int_2 = self.intrface_avg()
        int_1 = [*int_1, *int_2]
        #print(f"   non similar terms  = {len(int_1)}")
        for i in range(len(int_1)):
            output.write('  <coefficient number="{}" value="{}" text="{}" >\n'.format(coef_cntr,int_1[i][0]['value'],int_1[i][0]['text']))
            for k in range(int(int_1[i][2][-1]['dips'])):
                if k==0:
                    output.write('    <term weight="{}">\n'.format(int_1[i][1]*float(int_1[i][2][k]['weight'])))
                output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(int_1[i][2][k]['atom_a'],int_1[i][2][k]['atom_b'],int_1[i][2][k]['direction'],int_1[i][2][k]['power']))
                output.write('        <cell_a>{}</cell_a>\n'.format(int_1[i][2][k]['cell_a']))
                output.write('        <cell_b>{}</cell_b>\n'.format(int_1[i][2][k]['cell_b']))
                output.write('      </displacement_diff>\n')
            if int(int_1[i][2][-1]['dips']) !=0 :
                for l in range(int(int_1[i][2][-1]['strain'])):
                    output.write('      <strain power="{}" voigt="{}"/>\n'.format(int_1[i][2][k+l+1]['power'],int_1[i][2][k+l+1]['voigt']))
                output.write('    </term>\n')
            if len(int_1[i][2]) != 0:
                if int(int_1[i][2][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1

        int_1 = sim_trms
        #print(f"   len_similar terms  = {len(int_1)}")
        for i in range(len(int_1)):
            output.write('  <coefficient number="{}" value="{}" text="{}" >\n'.format(coef_cntr,int_1[i][1],int_1[i][0]['text']))
            for k in range(int(int_1[i][2][-1]['dips'])):
                if k==0:
                    output.write('    <term weight="{}">\n'.format(int_1[i][2][k]['weight']))
                output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(int_1[i][2][k]['atom_a'],int_1[i][2][k]['atom_b'],int_1[i][2][k]['direction'],int_1[i][2][k]['power']))
                output.write('        <cell_a>{}</cell_a>\n'.format(int_1[i][2][k]['cell_a']))
                output.write('        <cell_b>{}</cell_b>\n'.format(int_1[i][2][k]['cell_b']))
                output.write('      </displacement_diff>\n')
            if int(int_1[i][2][-1]['dips']) !=0 :
                for l in range(int(int_1[i][2][-1]['strain'])):
                    output.write('      <strain power="{}" voigt="{}"/>\n'.format(int_1[i][2][k+l+1]['power'],int_1[i][2][k+l+1]['voigt']))
                output.write('    </term>\n')
            if len(int_1[i][2]) != 0:
                if int(int_1[i][2][-1]['dips']) != 0:
                    output.write('  </coefficient>\n')
                    coef_cntr += 1


##########################################

        coeff1,tmp_trms1 = self.xml_anha(self.anh_xml1,self.my_atoms1)
        coeff2,tmp_trms2 = self.xml_anha(self.anh_xml2,self.my_atoms2)
        #print(str_coeff_1)
        disp, strain = terms_comp(tmp_trms1,tmp_trms2)
        #print(strain)
        avg_strain = True
        if (avg_strain):
            for str_cntr in (strain):
                ind1 = str_coeff_1.index(str_cntr['term_1'])
                ind2 = str_coeff_2.index(str_cntr['term_2'])
                #print(ind1,ind2)
                str_coeff_1.pop(ind1)
                str_coeff_2.pop(ind2)

                avg_valu = (l1*float(coeff1[str_cntr['term_1']]['value']) + l2*float(coeff2[str_cntr['term_2']]['value']))/(l1+l2)
                output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coef_cntr,avg_valu,coeff1[str_cntr['term_1']]['text'],))
                for j in range(len(trms1[str_cntr['term_1']])):
                    output.write('    <term weight="{}">\n'.format(trms1[str_cntr['term_1']][j][-2]['weight']))
                    for l in range(int(trms1[str_cntr['term_1']][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms1[str_cntr['term_1']][j][l]['power'],trms1[str_cntr['term_1']][j][l]['voigt']))
                    output.write('    </term>\n')
                output.write('  </coefficient>\n')
                coef_cntr += 1

##########################################
        #print('##############################')
        #print(str_coeff_1)

        if (str_str==0 or str_str==1):
            for i in str_coeff_1:
                output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coef_cntr,(l1/(l1+l2))*float(coeff1[i]['value']),coeff1[i]['text'],))
                for j in range(len(trms1[i])):
                    output.write('    <term weight="{}">\n'.format(trms1[i][j][-2]['weight']))
                    for l in range(int(trms1[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms1[i][j][l]['power'],trms1[i][j][l]['voigt']))
                    output.write('    </term>\n')
                output.write('  </coefficient>\n')
                coef_cntr += 1


        if (str_str==0 or str_str==2):
            for i in str_coeff_2:
                output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(coef_cntr,(l2/(l1+l2))*float(coeff2[i]['value']),coeff2[i]['text'],))
                for j in range(len(trms2[i])):
                    output.write('    <term weight="{}">\n'.format(trms2[i][j][-2]['weight']))
                    for l in range(int(trms2[i][j][-1]['strain'])):
                        output.write('      <strain power="{}" voigt="{}"/>\n'.format(trms2[i][j][l]['power'],trms2[i][j][l]['voigt']))
                    output.write('    </term>\n')
                output.write('  </coefficient>\n')
                coef_cntr += 1
        output.write('</Heff_definition>\n')

    def find_str_phon(self,trms):
        self.str_phonon_coeff = []
        nterm = 0
        for i in range(len(trms)):
            #for j in range(len(trms[i])):
            nstrain = int(trms[i][nterm][-1]['strain'])
            ndis = int(trms[i][nterm][-1]['dips'])
            if nstrain !=0 and ndis !=0:
                self.str_phonon_coeff.append(i)

    def get_str_phonon_voigt(self,trms,voigts = [1,2,3]):
        self.find_str_phon(trms)
        nterm=0
        my_terms = []
        for i in self.str_phonon_coeff:
            nstrain = int(trms[i][nterm][-1]['strain'])
            ndis = int(trms[i][nterm][-1]['dips'])
            for l in range(nstrain):
                my_voigt = int(trms[i][nterm][ndis+l]['voigt'])
                if my_voigt in voigts:
                    my_terms.append(i)
                    break
        return(my_terms)

    def get_new_str_terms(self,trms,my_coeff,nterm=0,voigts = [1,2,3]):
        vogt_terms = []
        my_vogt_dic = {1:'x',2:'y',3:'z',4:'xx',5:'yy',6:'zz'}
        my_vogt_str = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',}
        my_voigt_temrs = self.get_str_phonon_voigt(trms,voigts)
        if 1:
        #for i in my_voigt_temrs:
            i = my_coeff
            nstrain = int(trms[i][nterm][-1]['strain'])
            ndis = int(trms[i][nterm][-1]['dips'])
            my_lst = []
            for l in range(nstrain):
                my_voigt = int(trms[i][nterm][ndis+l]['voigt'])
                my_power = int(trms[i][nterm][ndis+l]['power'])
                if int(my_voigt) in voigts:
                    my_str = ({my_vogt_dic[my_voigt]:1,my_vogt_str[my_voigt]:1},my_power,i)
                    my_lst.append(my_str)
            vogt_terms.append(my_lst)
        return(vogt_terms)

    def get_org_terms(self,trms,my_coeff,nterm=0,voigts = [1,2,3]):
        org_terms = []
        my_vogt_dic = {1:'x',2:'y',3:'z'}
        my_voigt_temrs = self.get_str_phonon_voigt(trms,voigts)

        if 1:
            i = my_coeff
            nstrain = int(trms[i][nterm][-1]['strain'])
            ndis = int(trms[i][nterm][-1]['dips'])
            my_lst = []
            for l in range(nstrain):
                my_voigt = int(trms[i][nterm][ndis+l]['voigt'])
                my_power = int(trms[i][nterm][ndis+l]['power'])
                if int(my_voigt) in voigts:
                    my_str = ({my_vogt_dic[my_voigt]:1},my_power,i)
                    my_lst.append(my_str)
            org_terms.append(my_lst)
        return(org_terms)

    def get_mult_coeffs(self,my_str_trms):
        mult_terms = []
        for i in range(len(my_str_trms)):
            tem_dic = {}
            for j in range(len(my_str_trms[i])):
                if j ==0:
                    tem_dic = get_pwr_N(my_str_trms[i][j][0],my_str_trms[i][j][1])
                else:
                    tem_dic = terms_mult(tem_dic,get_pwr_N(my_str_trms[i][j][0],my_str_trms[i][j][1]))
            mult_terms.append(tem_dic)
        return(mult_terms)

    def get_shifted_terms(self,trms,my_coeff,nterm,my_strain=[0,0,0],voigts=[1,2,3]):
        not_shift = ['x','y','z']
        a,b,c = my_strain[0],my_strain[1],my_strain[2]
        my_mul_terms = self.get_mult_coeffs(self.get_new_str_terms(trms,my_coeff,nterm,voigts))
        org_terms = self.get_mult_coeffs(self.get_org_terms(trms,my_coeff,nterm,voigts))

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

    def get_final_coeffs(self,coeff,trms,my_atoms,my_tags,my_strain,voigts=[1,2,3]):
        str_coeffs = self.get_str_phonon_voigt(trms,voigts)
        tot_nterms = len(trms)
        new_coeffs = []
        new_temrs = []
        for i,ii in enumerate(str_coeffs):
            my_str_phon_term = []
            for my_term in range(len(trms[ii])):
                my_terms = self.get_shifted_terms(trms,ii,my_term,my_strain,voigts)

                disp_text = ''
                ndisp = int(trms[ii][my_term][-1]['dips'])
                for disp in range(ndisp):
                    atm_a = int(trms[ii][my_term][disp]['atom_a'])
                    atm_b = int(trms[ii][my_term][disp]['atom_b'])
                    cell_a = [int(x) for x in  trms[ii][my_term][disp]['cell_a'].split()]
                    cell_b = [int(x) for x in trms[ii][my_term][disp]['cell_b'].split()]
                    direction = trms[ii][my_term][disp]['direction']
                    power = trms[ii][my_term][disp]['power']
                    if not any(cell_b):
                        disp_text+=(f'({my_tags[atm_a]}_{direction}-{my_tags[atm_b]}_{direction})^{power}')
                    else:
                        disp_text+=(f'({my_tags[atm_a]}_{direction}-{my_tags[atm_b]}_{direction}[{cell_b[0]} {cell_b[1]} {cell_b[2]}])^{power}')
                my_dis_term = trms[ii][my_term][0:ndisp]
                ## ad the term here for disp

                term_cnre = 0
                for tmp_key in my_terms[0].keys():
                    if len(my_str_phon_term) < len(my_terms[0].keys()):
                        my_str_phon_term.append([])
                        #if i==5:
                            #print('aded',ii)
                            #print(disp_text)
                            #print(my_terms)
                    num_str_temrs = 0
                    str_terms = []
                    # find x
                    pwer_x = tmp_key.count('x')
                    if pwer_x !=0:
                        str_terms.append({'power': f' {pwer_x}' , 'voigt': ' 1'} )
                        num_str_temrs += 1
                    # find y
                    pwer_y = tmp_key.count('y')
                    if pwer_y !=0:
                        str_terms.append({'power': f' {pwer_y}' , 'voigt': ' 2'} )
                        num_str_temrs += 1
                    # find z
                    pwer_z = tmp_key.count('z')
                    if pwer_z !=0:
                        str_terms.append({'power': f' {pwer_z}' , 'voigt': ' 3'} )
                        num_str_temrs += 1
                    for str_cntr in range(int(trms[ii][my_term][-1]['strain'])):
                        if int(trms[ii][my_term][ndisp+str_cntr]['voigt']) not in (voigts):
                            str_terms.append(trms[ii][my_term][ndisp+str_cntr])
                            num_str_temrs += 1

                    my_str_phon_term[term_cnre].append([*my_dis_term,*str_terms,{'dips':ndisp,'strain':num_str_temrs,'distance':0.0}])
                    term_cnre += 1

                if my_term == 0:
                    temp_trms = re_order_terms(my_terms[0])
                    key_cntr = 0
                    for my_key in temp_trms.keys():
                        tot_nterms+=1
                        my_value = float(coeff[ii]['value'])*temp_trms[my_key]
                        my_key = my_key.replace('x','(eta_1)')
                        my_key = my_key.replace('y','(eta_2)')
                        my_key = my_key.replace('z','(eta_3)')
                        my_text = disp_text+my_key
                        new_coeff = {'number': str(tot_nterms), 'value': str(my_value), 'text': my_text, 'org_coeff' : ii}
                        new_coeffs.append(new_coeff)
                        key_cntr +=1



            for temp_cntr in range(len(my_str_phon_term)):
                new_temrs.append(my_str_phon_term[temp_cntr])

        return(new_coeffs,new_temrs)

    def get_lat_mismatch(self,l):
        ref_cell = self.ref_cell
        if l==0:
            cell_1 = self.SL1.get_cell()
            cell_2 = ref_cell.get_cell()
            str_0 = (cell_2[0,0]-cell_1[0,0])/cell_2[0,0]
            str_1 = (cell_2[1,1]-cell_1[1,1])/cell_2[1,1]
            str_2 = (cell_2[2,2]-cell_1[2,2])/cell_2[2,2]
        if l==1:
            cell_1 = self.SL2.get_cell()
            cell_2 = ref_cell.get_cell()
            str_0 = (cell_2[0,0]-cell_1[0,0])/cell_2[0,0]
            str_1 = (cell_2[1,1]-cell_1[1,1])/cell_2[1,1]
            str_2 = (cell_2[2,2]-cell_1[2,2])/cell_2[2,2]
        return([str_0,str_1,str_2])

###########################################################
