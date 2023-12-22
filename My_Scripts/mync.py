import sys
#sys.path.append('/home/alireza/CODES/My_scr')
import netCDF4 as nc
import numpy as np
from ase import build, Atoms
from ase.units import Ha, Bohr
from os.path import exists
import os
from ase.units import Ang, Bohr, Hartree, eV, _e
from ase.io import write
import matplotlib.pyplot as plt
# from my_functions import map_strctures


class hist_write():
    def __init__(self,nc_file,t):
        file_exists = exists(str_atom, nc_file,t=0)
        if file_exists and t!= 0:
            my_char='a'
        elif file_exists and t==0:
            os.system(f'mv {nc_file} 01{nc_file}')
            my_char='w'
        else:
            my_char='w'
        ds = nc.Dataset(nc_file, my_char, format='NETCDF4')
        atom_num=str_atom.get_atomic_numbers()
        pos=str_atom.get_positions()
        cell=str_atom.get_cell()
        scld_pos=str_atom.get_scaled_positions()
        masses=str_atom.get_masses()

    def set_structures(self,t):
        red_atm_num=[]
        for i in atom_num:
            if i not in red_atm_num:
                red_atm_num.append(i)

        red_mass=[]
        for i in masses:
            if i not in red_mass:
                red_mass.append(i)

        my_natom=len(pos)

        mytyp=[red_atm_num.index(i)+1 for i in atom_num]
        if t==0:
            natom = ds.createDimension('natom', my_natom)
            ntypat = ds.createDimension('ntypat', len(red_atm_num))
            npsp = ds.createDimension('npsp', len(red_atm_num))
            xyz = ds.createDimension( 'xyz', 3)
            six = ds.createDimension( 'six', 6)
            time = ds.createDimension( 'time', None)
            two = ds.createDimension('two', 2)

            typat=ds.createVariable('typat', np.float64, ('natom',))
            typat.units = 'dimensionless'
            typat.mnemonics= 'types of atoms'
            typat[:]=mytyp

            znucl=ds.createVariable('znucl', np.float64, ('npsp',))
            znucl.units = 'atomic units'
            znucl.mnemonics= 'atomic charges'
            znucl[:]=red_atm_num

            amu=ds.createVariable('amu', np.float64, ('ntypat',))
            amu.units = 'atomic units'
            amu.mnemonics= 'atomic masses'
            amu[:]=red_mass


            dtion=ds.createVariable('dtion', np.float64)
            dtion.units = 'atomic units'
            dtion.mnemonics= 'time step'
            dtion[t]=t


            mdtemp=ds.createVariable('mdtemp', np.float64, ('two',))
            mdtemp.units = 'Kelvin'
            mdtemp.mnemonics= 'Molecular Dynamics Thermostat Temperatures'
            mdtemp[:]=[0,0]

            mdtime=ds.createVariable('mdtime', np.float64, ('time',))
            mdtime.units = 'hbar/Ha'
            mdtime.mnemonics= 'Molecular Dynamics or Relaxation TIME'
#            mdtime[t]=t


            xcart=ds.createVariable('xcart', np.float64, ('time', 'natom', 'xyz'))
            xcart.units = 'bohr'
            xcart.mnemonics= 'vectors (X) of atom positions in CARTesian coordinates'
#            xcart[t,:]=pos/Bohr

            xred=ds.createVariable('xred', np.float64, ('time', 'natom', 'xyz'))
            xred.units = 'dimensionless'
            xred.mnemonics= 'vectors (X) of atom positions in REDuced coordinates'
#            xred[t,:]=scld_pos



            fcart=ds.createVariable('fcart', np.float64, ('time', 'natom', 'xyz'))
            fcart.units = 'Ha/bohr'
            fcart.mnemonics= 'atom Forces in CARTesian coordinates'
#            fcart[t,:]=np.zeros((my_natom,3))


            fred=ds.createVariable('fred', np.float64, ('time', 'natom', 'xyz'))
            fred.units = 'dimensionless'
            fred.mnemonics= 'atom Forces in REDuced coordinates'
#            fred[t,:]=np.zeros((my_natom,3))


            vel=ds.createVariable('vel', np.float64, ('time', 'natom', 'xyz'))
            vel.units = 'bohr*Ha/hbar'
            vel.mnemonics= 'VELocities of atoms'
#            vel[t,:]=np.zeros((my_natom,3))


            rprimd=ds.createVariable('rprimd', np.float64, ('time', 'xyz', 'xyz'))
            rprimd.units = 'bohr'
            rprimd.mnemonics= 'Real space PRIMitive translations, Dimensional'
#            rprimd[t,:]=cell/Bohr


            vel_cell=ds.createVariable('vel_cell', np.float64, ('time', 'xyz', 'xyz'))
            vel_cell.units = 'bohr*Ha/hbar'
            vel_cell.mnemonics= 'VELocities of cell'
#            vel_cell[t,:]=np.zeros((3,3))


            acell=ds.createVariable('acell', np.float64, ('time', 'xyz'))
            acell.units = 'bohr'
            acell.mnemonics= 'CELL lattice vector scaling'
#            acell[t,:]=my_acell/Bohr


            strten=ds.createVariable('strten', np.float64, ('time', 'six'))
            strten.units = 'Ha/bohr^3'
            strten.mnemonics= 'STRess tensor'
#            strten[t,:]=np.zeros((1,6))


            etotal=ds.createVariable('etotal', np.float64, ('time',))
            etotal.units = 'Ha'
            etotal.mnemonics= 'TOTAL Energy'
#            etotal[t]=0


            ekin=ds.createVariable('ekin', np.float64,('time',))
            ekin.units = 'Ha'
            ekin.mnemonics= 'Energy KINetic ionic'
#            ekin[t]=0

            entropy=ds.createVariable('entropy', np.float64,('time',) )
            entropy.units = ' '
            entropy.mnemonics= 'Entropy'
#            entropy[t]=0.
        else:
            entropy=ds['entropy']
            ekin=ds['ekin']
            etotal=ds['etotal']
            strten=ds['strten']
            acell=ds['acell']
            vel_cell=ds['vel_cell']
            rprimd=ds['rprimd']
            vel=ds['vel']
            fred=ds['fred']
            fcart=ds['fcart']
            xred=ds['xred']
            xcart=ds['xcart']
            mdtime=ds['mdtime']

        entropy[t]=0
        ekin[t]=0
        etotal[t]=0
        strten[t,:]=np.zeros((1,6))
        acell[t,:]=1,1,1
        vel_cell[t,:]=np.zeros((3,3))
        rprimd[t,:]=cell/Bohr
        vel[t,:]=np.zeros((my_natom,3))
        fred[t,:]=np.zeros((my_natom,3))
        fcart[t,:]=np.zeros((my_natom,3))
        xred[t,:]=scld_pos
        xcart[t,:]=pos/Bohr
        mdtime[t]=t

        ds.close()

def write_hist(str_atom,fout,t=0):
    file_exists = exists(fout)
    if file_exists and t!= 0:
        my_char='a'
    elif file_exists and t==0:
        os.system(f'mv {fout} 01{fout}')
        my_char='w'
    else:
        my_char='w'
    ds = nc.Dataset(fout, my_char, format='NETCDF4')
    atom_num=str_atom.get_atomic_numbers()
    pos=str_atom.get_positions()
    cell=str_atom.get_cell()
    #mycell=ase.geometry.Cell(cell)
    scld_pos=str_atom.get_scaled_positions()
    #scld_pos=mycell.scaled_positions(pos)
    #my_acell=str_atom.cell.cellpar()[0:3]
    masses=str_atom.get_masses()

    red_atm_num=[]
    for i in atom_num:
        if i not in red_atm_num:
            red_atm_num.append(i)

    red_mass=[]
    for i in masses:
        if i not in red_mass:
            red_mass.append(i)

    my_natom=len(pos)

    mytyp=[red_atm_num.index(i)+1 for i in atom_num]
    if t==0:
        natom = ds.createDimension('natom', my_natom)
        ntypat = ds.createDimension('ntypat', len(red_atm_num))
        npsp = ds.createDimension('npsp', len(red_atm_num))
        xyz = ds.createDimension( 'xyz', 3)
        six = ds.createDimension( 'six', 6)
        time = ds.createDimension( 'time', None)
        two = ds.createDimension('two', 2)

        typat=ds.createVariable('typat', np.float64, ('natom',))
        typat.units = 'dimensionless'
        typat.mnemonics= 'types of atoms'
        typat[:]=mytyp

        znucl=ds.createVariable('znucl', np.float64, ('npsp',))
        znucl.units = 'atomic units'
        znucl.mnemonics= 'atomic charges'
        znucl[:]=red_atm_num

        amu=ds.createVariable('amu', np.float64, ('ntypat',))
        amu.units = 'atomic units'
        amu.mnemonics= 'atomic masses'
        amu[:]=red_mass


        dtion=ds.createVariable('dtion', np.float64)
        dtion.units = 'atomic units'
        dtion.mnemonics= 'time step'
        dtion[t]=t


        mdtemp=ds.createVariable('mdtemp', np.float64, ('two',))
        mdtemp.units = 'Kelvin'
        mdtemp.mnemonics= 'Molecular Dynamics Thermostat Temperatures'
        mdtemp[:]=[0,0]

        mdtime=ds.createVariable('mdtime', np.float64, ('time',))
        mdtime.units = 'hbar/Ha'
        mdtime.mnemonics= 'Molecular Dynamics or Relaxation TIME'
#        mdtime[t]=t


        xcart=ds.createVariable('xcart', np.float64, ('time', 'natom', 'xyz'))
        xcart.units = 'bohr'
        xcart.mnemonics= 'vectors (X) of atom positions in CARTesian coordinates'
#        xcart[t,:]=pos/Bohr

        xred=ds.createVariable('xred', np.float64, ('time', 'natom', 'xyz'))
        xred.units = 'dimensionless'
        xred.mnemonics= 'vectors (X) of atom positions in REDuced coordinates'
#        xred[t,:]=scld_pos



        fcart=ds.createVariable('fcart', np.float64, ('time', 'natom', 'xyz'))
        fcart.units = 'Ha/bohr'
        fcart.mnemonics= 'atom Forces in CARTesian coordinates'
#        fcart[t,:]=np.zeros((my_natom,3))


        fred=ds.createVariable('fred', np.float64, ('time', 'natom', 'xyz'))
        fred.units = 'dimensionless'
        fred.mnemonics= 'atom Forces in REDuced coordinates'
#        fred[t,:]=np.zeros((my_natom,3))


        vel=ds.createVariable('vel', np.float64, ('time', 'natom', 'xyz'))
        vel.units = 'bohr*Ha/hbar'
        vel.mnemonics= 'VELocities of atoms'
#        vel[t,:]=np.zeros((my_natom,3))


        rprimd=ds.createVariable('rprimd', np.float64, ('time', 'xyz', 'xyz'))
        rprimd.units = 'bohr'
        rprimd.mnemonics= 'Real space PRIMitive translations, Dimensional'
#        rprimd[t,:]=cell/Bohr


        vel_cell=ds.createVariable('vel_cell', np.float64, ('time', 'xyz', 'xyz'))
        vel_cell.units = 'bohr*Ha/hbar'
        vel_cell.mnemonics= 'VELocities of cell'
#        vel_cell[t,:]=np.zeros((3,3))


        acell=ds.createVariable('acell', np.float64, ('time', 'xyz'))
        acell.units = 'bohr'
        acell.mnemonics= 'CELL lattice vector scaling'
#        acell[t,:]=my_acell/Bohr


        strten=ds.createVariable('strten', np.float64, ('time', 'six'))
        strten.units = 'Ha/bohr^3'
        strten.mnemonics= 'STRess tensor'
#        strten[t,:]=np.zeros((1,6))


        etotal=ds.createVariable('etotal', np.float64, ('time',))
        etotal.units = 'Ha'
        etotal.mnemonics= 'TOTAL Energy'
#        etotal[t]=0


        ekin=ds.createVariable('ekin', np.float64,('time',))
        ekin.units = 'Ha'
        ekin.mnemonics= 'Energy KINetic ionic'
#        ekin[t]=0

        entropy=ds.createVariable('entropy', np.float64,('time',) )
        entropy.units = ' '
        entropy.mnemonics= 'Entropy'
#        entropy[t]=0.
    else:
        entropy=ds['entropy']
        ekin=ds['ekin']
        etotal=ds['etotal']
        strten=ds['strten']
        acell=ds['acell']
        vel_cell=ds['vel_cell']
        rprimd=ds['rprimd']
        vel=ds['vel']
        fred=ds['fred']
        fcart=ds['fcart']
        xred=ds['xred']
        xcart=ds['xcart']
        mdtime=ds['mdtime']

    entropy[t]=0
    ekin[t]=0
    etotal[t]=0
    strten[t,:]=np.zeros((1,6))
    acell[t,:]=1,1,1
    vel_cell[t,:]=np.zeros((3,3))
    rprimd[t,:]=cell/Bohr
    vel[t,:]=np.zeros((my_natom,3))
    fred[t,:]=np.zeros((my_natom,3))
    fcart[t,:]=np.zeros((my_natom,3))
    xred[t,:]=scld_pos
    xcart[t,:]=pos/Bohr
    mdtime[t]=t

    ds.close()

def get_avg_str(NC_HIST,init_stp=0):
    dso=nc.Dataset(NC_HIST)
    RSET=dso.variables['rprimd'][:]
    xcart=dso.variables['xcart'][:]
    typ0=dso.variables['typat'][:]
    numbers0=dso.variables['znucl'][:]
    numbers=[numbers0[:][int(tt)-1] for tt in typ0[:]]
    sum_str=np.zeros((len(xcart[0]),3))
    sum_Rset=np.zeros((3,3))
    cntr = 0
    temp_atms0 = Atoms(numbers=numbers,positions=xcart[init_stp]*Bohr, cell=RSET[init_stp]*Bohr)
    for str_cntr in range(init_stp,len(xcart)):
        cntr+=1
        temp_atms = Atoms(numbers=numbers,positions=xcart[str_cntr]*Bohr, cell=RSET[str_cntr]*Bohr)
        temp_atms = map_strctures(temp_atms,temp_atms0)
        sum_str+=temp_atms.get_scaled_positions()
        sum_Rset+=temp_atms.get_cell()  
    avg_str=sum_str/cntr
    avg_Rset=sum_Rset/cntr  
    AVG_Str_Hist = Atoms(numbers=numbers,scaled_positions=avg_str, cell=avg_Rset)
    return(AVG_Str_Hist)

def get_NC_str(NC_HIST,stp=0):
    dso=nc.Dataset(NC_HIST)
    RSET=dso.variables['rprimd'][:]
    xcart=dso.variables['xcart'][:]
    typ0=dso.variables['typat'][:]
    numbers0=dso.variables['znucl'][:]
    numbers=[numbers0[:][int(tt)-1] for tt in typ0[:]]
    sum_str=np.zeros((len(xcart[0]),3))
    sum_Rset=np.zeros((3,3))
    # print(len(xcart))
    My_strct = Atoms(numbers=numbers,positions=xcart[stp]*Bohr, cell=RSET[stp]*Bohr, pbc=True)
    My_strct.wrap(eps=0.008)
    return(My_strct)


class hist_reader():    
    def __init__(self,nc_file,extrct_Strs=True):
        self.nc_file = nc_file
        self.dso=nc.Dataset(self.nc_file)
        self.extrct_Strs = extrct_Strs
        if self.extrct_Strs:
            self.get_tot_numbers()
            self.get_xred()
            self.get_RSET()
            self.nsteps = len(self.RSET)
            
    def get_RSET(self,i=None):
        if i==None:
            RSET=self.dso.variables['rprimd'][:]
            if self.extrct_Strs:
                self.RSET = RSET
        else:
            RSET=self.dso.variables['rprimd'][:][i]
        return(RSET)

    def get_xred(self,i=None):
        if i==None:
            xred=self.dso.variables['xred'][:]
            if self.extrct_Strs:
                self.xred = xred
        else:
            xred=self.dso.variables['xred'][:][i]
        return(xred)

    def get_xcart(self,i=None):
        if i==None:
            xcart=self.dso.variables['xcart'][:]
        else:
            xcart=self.dso.variables['xcart'][:][i]
        return(xcart)

    def get_typat(self,i=None):
        if i==None:
            typat=self.dso.variables['typat'][:]
        else:
            typat=self.dso.variables['typat'][:][i]
        return(typat)

    def get_numbers(self,i=None):
        if i==None:
            numbers=self.dso.variables['znucl'][:]
        else:
            numbers=self.dso.variables['znucl'][:][i]
        return(numbers)

    def get_tot_numbers(self,i=-1):
        numbers0 = self.get_numbers()
        typ0 = self.get_typat()
        numbers_tot=[numbers0[:][int(i)-1] for i in typ0[:]]
        self.tot_numbers = numbers_tot
        return(numbers_tot)

    def get_fcart(self,i=None):
        if i==None:
            for_cart=self.dso.variables['fcart'][:]
        else:
            for_cart=self.dso.variables['fcart'][:][i]
        return(for_cart)

    def get_stress(self,i=None):
        if i==None:
            stress = self.dso.variables['strten'][:]
        else:
            stress = self.dso.variables['strten'][:][i]
        return(stress)

    def get_etotal(self,i=None):
        if i==None:
            etotal = self.dso.variables['etotal'][:]
        else:
            etotal = self.dso.variables['etotal'][:][i]
        return(etotal)

    def get_ase_str(self,i=-1,PBC_Flg=True):
        self.get_tot_numbers()
        self.get_xred()
        self.get_RSET()
        numbers = self.tot_numbers
        my_xred = self.xred[i]
        my_RSET = self.RSET[i]
        str_from_nc= Atoms(numbers=numbers,scaled_positions=my_xred, cell=my_RSET*Bohr, pbc=PBC_Flg)
        return(str_from_nc)

    def get_avg_str(self,initial=0):
        self.get_tot_numbers()
        self.get_xred()
        self.get_RSET()
        RSET=self.get_RSET() 
        xcart=self.get_xcart()
        numbers=self.get_tot_numbers()
        sum_str=np.zeros((len(xcart[0]),3))
        sum_Rset=np.zeros((3,3))
        cntr = 0
        temp_atms0 = Atoms(numbers=numbers,positions=xcart[initial]*Bohr, cell=RSET[initial]*Bohr)
        if initial == -1:
            initial = len(xcart-1)
        for str_cntr in range(initial,len(xcart)):
            cntr+=1
            temp_atms = Atoms(numbers=numbers,positions=xcart[str_cntr]*Bohr, cell=RSET[str_cntr]*Bohr)
            temp_atms = map_strctures(temp_atms,temp_atms0)
            sum_str+=temp_atms.get_scaled_positions()
            sum_Rset+=temp_atms.get_cell()  
        avg_str=sum_str/cntr
        avg_Rset=sum_Rset/cntr  
        AVG_Str_Hist = Atoms(numbers=numbers,scaled_positions=avg_str, cell=avg_Rset)
        return(AVG_Str_Hist)

    def get_mdtemp(self,i=None):
        if i==None:
            mdtemp = self.dso.variables['mdtemp'][:]
        else:
            mdtemp = self.dso.variables['mdtemp'][:][i]
        return(mdtemp)

    def plot_etotal(self,plot_step=0,output='Etotal'):
        fig,ax = plt.subplots()  
        SMALL_SIZE = 13
        MEDIUM_SIZE = 13
        BIGGER_SIZE = 13
        my_dpi=300
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title                                             
        Eng_tmp = np.array(self.get_etotal())
        plt.plot(Eng_tmp[plot_step:],label='E_total')
        plt.xlabel('Steps')       
        plt.ylabel('Etotal(Ha)')      
        plt.savefig(f'{output}')
    
    def plot_stress(self,plot_step=0,output='Stress'):
        fig,ax = plt.subplots()   
        SMALL_SIZE = 13
        MEDIUM_SIZE = 13
        BIGGER_SIZE = 13
        my_dpi=300
        
        cnst = (Hartree/Bohr**3)*_e*10**21 
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title                                         
        stress_tmp = np.array(self.get_stress())

        plt.plot(cnst*stress_tmp[plot_step:,0],label='sig_1')
        plt.plot(cnst*stress_tmp[plot_step:,1],label='sig_2')
        plt.plot(cnst*stress_tmp[plot_step:,2],label='sig_3')  
        plt.plot(cnst*stress_tmp[plot_step:,3],label='sig_4')
        plt.plot(cnst*stress_tmp[plot_step:,4],label='sig_5')
        plt.plot(cnst*stress_tmp[plot_step:,5],label='sig_6')   
        plt.xlabel('Steps')       
        plt.ylabel('Stress(GPa)')       
        plt.legend()                                 
        plt.savefig(f'{output}')

    def plot_acell():
        pass


def map_strctures(str_1,str_2,tol=0.5):  # the ordering is correct
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

