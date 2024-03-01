"""
Main script to make supercells and superlattice models. 
"""
from My_Scripts import SC_xml_potential as SC_pot
import numpy as np
import My_Scripts.interface_xmls as interface_xmls
from My_Scripts import xml_io
from My_Scripts.My_simulations import get_xml_files
from My_Scripts import tools 
import os
from ase.build import sort

def SC_model_maker(my_harf,my_Anhf,scll,my_SCll,har_out,anh_out,strain_in=np.zeros((3,3)),missfit_strain=True,Higher_order_strain=False,elas_const_mul = None,scnd_order_strain=False):
    """
    Make a supercell model from a primitive cell model
    Params:
        my_harf: the harmonic force constant xml file of the primitive cell
        my_Anhf: the anharmonic force constant xml file of the primitive cell
        scll: the supercell matrix
        my_SCll: the atomic structure of the supercell with the order of the atoms you want. 
        har_out: the output harmonic force constant xml file of the supercell
        anh_out: the output anharmonic force constant xml file of the supercell
        strain_in: the strain tensor of the supercell in terms of the primitive cell
        missfit_strain: whether to include the mismatch strain of the supercell
        Higher_order_strain: whether to include the higher order strain of the supercell
    """
    sc_maker=SC_pot.Har_sc_maker(my_harf,scll,strain_in,elas_const_mul = elas_const_mul)
    sc_maker.reshape_FCDIC(my_SCll)
    sc_maker.write_xml(har_out)
    anh_SCxml=SC_pot.Anh_sc_maker(my_harf,my_Anhf,strain_in,missfit_strain=missfit_strain,Higher_order_strain=Higher_order_strain,scnd_order_strain=scnd_order_strain)
    anh_SCxml.SC_trms(my_SCll,scll)
    anh_SCxml.wrt_anxml(anh_out)

def int_model_maker(xmlf1, anh_file1, scmat1, xmlf2, anh_file2, scmat2, symmetric=False, har_file='int_harmoni.xml', Anhar_file='int_harmoni.xml',
                negelect_A_SITE=False,negelect_Tot_FCs=False, NW_Strc = False,sim_eps=False):
    """
    Make a superlattice model from two primitive cell models
    Params:
        xmlf1: the harmonic force constant xml file of the primitive cell 1
        anh_file1: the anharmonic force constant xml file of the primitive cell 1
        scmat1: the supercell matrix of the primitive cell 1
        xmlf2: the harmonic force constant xml file of the primitive cell 2
        anh_file2: the anharmonic force constant xml file of the primitive cell 2
        scmat2: the supercell matrix of the primitive cell 2
        symmetric: whether to make the superlattice model symmetric
        har_file: the output harmonic force constant xml file of the superlattice model
        Anhar_file: the output anharmonic force constant xml file of the superlattice model
        negelect_A_SITE: whether to neglect the A-site atoms in the superlattice model (put the interactions of A-site between the two materails to zero.)
        negelect_Tot_FCs: whether to neglect the total force constants in the superlattice model. (only keep the short range part.)
    """
    # Harmonic_term generation
    har_xml = interface_xmls.Har_interface(xmlf1, scmat1, xmlf2, scmat2, symmetric=symmetric,negelect_A_SITE=negelect_A_SITE,negelect_Tot_FCs=negelect_Tot_FCs,NW_Strc = NW_Strc,sim_eps=sim_eps)
    har_xml.get_STR_FCDIC()
    har_xml.reshape_FCDIC()
    STRC = har_xml.ref_cell
    har_xml.write_xml(har_file)

    # Anhrmonic_term generation
    intf = interface_xmls.Anh_intrface(xmlf1, anh_file1, scmat1, xmlf2, anh_file2,
                        scmat2, symmetric=symmetric, NW_Strc = NW_Strc)
    intf.wrt_anxml(Anhar_file)
    # print(intf.FC_weights)
    return(har_xml)    

def get_avg_cell_SL(har_xml1,dim_1,har_xml2,dim_2):
    """
    Get the average cell of two primitive cells for the superlattice model depending on the elastic constants of the two primitive cells.
    Params:
        har_xml1: the harmonic force constant xml file of the primitive cell 1
        dim_1: the number of layers of the primitive cell 1
        har_xml2: the harmonic force constant xml file of the primitive cell 2
        dim_2: the number of layers of the primitive cell 2
    The algorithm is based on the  equation in Carlos's thesis:
        """
    myxml1=xml_io.Xml_sys_reader(har_xml1)
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    myxml2=xml_io.Xml_sys_reader(har_xml2)
    myxml2.get_ase_atoms()
    my_atms2=myxml2.ase_atoms
    myxml1.get_ela_cons()
    myxml2.get_ela_cons()
    ELC1 = myxml1.ela_cons
    ELC2 = myxml2.ela_cons                        
    cell_1 = my_atms1.get_cell()
    cell_2 = my_atms2.get_cell()
    p1 = dim_1/(dim_1+dim_2)
    p2 = dim_2/(dim_1+dim_2)
    m = np.zeros((3))
    for indx in range(3):
        m[indx] = p1*ELC1[indx][indx]/(p2*ELC2[indx][indx])
    a_avg = np.zeros((3))
    for a_indx in range(3):
        a_avg[a_indx] = cell_1[a_indx][a_indx]*cell_2[a_indx][a_indx] * \
            (m[a_indx]*cell_1[a_indx][a_indx]+cell_2[a_indx][a_indx]) / \
            (m[a_indx]*cell_1[a_indx][a_indx]**2+cell_2[a_indx][a_indx]**2)

    a2 = (dim_1*a_avg[2]+dim_2*a_avg[2])/(dim_1+dim_2)
    ref_cell = np.eye(3)
    ref_cell[0,0] = a_avg[0]
    ref_cell[1,1] = a_avg[1]
    ref_cell[2,2] = a2
    return(ref_cell)

def SL_MAKER(DDB1,modle1,ncell1,DDB2,modle2,ncell2,ngqptm,sim_path1=None,sim_path2=None,NCPU=1,ref_cell='M2',Har_int='Har_int',Anh_int='Anh_int',
             miss_fit_trms=True,Higher_order_strain=True,negelect_A_SITE=True,negelect_Tot_FCs=True,symmetric=True,xml_cell1=None,xml_cell2=None,if_return_atom=False,EXEC='MB_16Jun',scnd_order_strain=False,elas_const_mul = None):

    """
    Make a superlattice model from two primitive cell models
    params:
        DDB1: the  DDB file of the primitive cell 1
        modle1: the filename of the anharmonic model of primitive cell 1
        ncell1: (3 numbers) the range of the IFCS of the primitive cell 1 (to be used in multibinit and output a xml file.)
                ncell1[2] should be the size of the supercell wanted for material 1.
        DDB2:  the  DDB file of the primitive cell 2
        modle2:  the filename of the anharmonic model of primitive cell 2
        ncell2:  (3 numbers) the range of the IFCS of the primitive cell 2 (to be used in multibinit and output a xml file.)
              ncell2[2] should be the size of the supercell wanted for material 2. 
              ncell1[0/1] and ncell2[0/1] should be the same. 

        ngqptm: the number of qpoints for the superlattice model, used for both primitive cell in multibinit.
        sim_path1: the directory of the primitive cell 1 for multibinit and output a xml file.
        sim_path2: the directory of the primitive cell 2 for multibinit and output a xml file.
        NCPU: the number of CPUs for the multibinit calculation.
        ref_cell: the reference cell of the superlattice model, can be 'M1', 'M2', 'avg'
        Har_int: the output harmonic force constant xml file of the superlattice model
        Anh_int: the output anharmonic force constant xml file of the superlattice model
        miss_fit_trms: whether to include the mismatch strain of the superlattice model
        Higher_order_strain: whether to include the higher order strain due to misfit of the superlattice model
        negelect_A_SITE: whether to neglect the A-site atoms in the superlattice model (put the interactions of A-site between the two materails to zero.)
        negelect_Tot_FCs: whether to neglect the total force constants in the superlattice model. (only keep the short range part.)
        symmetric: whether to make the superlattice model symmetric
        xml_cell1: the number of layers of the primitive cell 1 for the xml file
        xml_cell2: the number of layers of the primitive cell 2 for the xml file
        if_return_atom: whether to return the atoms of the superlattice model
        EXEC: Multibinit executable
    """
    
    print('making SL Pot')


    xml_cell1 = [ncell1[0],ncell1[1],ncell1[2]+ncell2[2]]  if  xml_cell1 is None else  xml_cell1  #### XML_CELL ***
    xml_cell2 = [ncell2[0],ncell2[1],ncell1[2]+ncell2[2]]  if  xml_cell2 is None else  xml_cell2  #### XML_CELL ***


    if sim_path1 is None:
        cwd = os.getcwd()
        sim_path1 = f'{cwd}/M1'

    if sim_path2 is None:
        cwd = os.getcwd()
        sim_path2 = f'{cwd}/M2'

    Har_name = f'Har_xml_{xml_cell1[0]}{xml_cell1[1]}{xml_cell1[2]}'              
    Anh_name = f'AnHar_xml_{xml_cell1[0]}{xml_cell1[1]}{xml_cell1[2]}'

    # SR_cd_dir = max(ngqpt1[0],ngqpt2[0])
    # run multibinit for the primitive cells
    har_xml1,anh_xml1 = get_xml_files(DDB1,modle1,ngqpt=ngqptm,ncell=xml_cell1,prt_dipdip=1,output_name='Str1',
                                      sim_path=sim_path1,Har_name=Har_name,Anh_name=Anh_name,EXEC=EXEC,NCPU=NCPU)


    har_xml2,anh_xml2 = get_xml_files(DDB2,modle2,ngqpt=ngqptm,ncell=xml_cell2,prt_dipdip=1,output_name='Str2',
                                      sim_path=sim_path2,Har_name=Har_name,Anh_name=Anh_name,EXEC=EXEC,NCPU=NCPU)
    

    # read the harmonic xml files and get the atoms
    myxml1=xml_io.Xml_sys_reader(har_xml1)
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    myxml2=xml_io.Xml_sys_reader(har_xml2)
    myxml2.get_ase_atoms()
    my_atms2=myxml2.ase_atoms

    # sort the atoms of the primitive cell 2 to match the primitive cell 1
    # my_atom_list_tmp is a list to map the atoms of the primitive cell 2 to the primitive cell 1
    my_atm_list_tmp = tools.get_mapped_strcs(my_atms1,my_atms2,Ret_index=True)
    # the atoms of the primitive cell 2 after sorting
    trans_atms2 = sort(my_atms2,tags = my_atm_list_tmp) 

    avg_cell = get_avg_cell_SL(har_xml1,ncell1[2],har_xml2,ncell2[2])
    # get the reference cell of the superlattice model
    if ref_cell == 'M1':
        print(f'{my_atms1.get_chemical_formula()} as refrence')
        ref_atoms_cell = my_atms1.get_cell()
    elif ref_cell == 'M2':
        print(f'{my_atms2.get_chemical_formula()} as refrence')
        ref_atoms_cell = my_atms2.get_cell()
    elif ref_cell == 'avg':
        print(f'aerage cells of {my_atms1.get_chemical_formula()} and {my_atms2.get_chemical_formula()} as refrence')
        ref_atoms_cell = avg_cell

    print('refrence cell in SL is: \n',ref_atoms_cell)

    #  find the strain of the structure 1.
    # The c-axis strain is set to zero.
    tmp_har_xml1 = f'{sim_path1}/tmp_{Har_name}'
    tmp_anh_xml1 =f'{sim_path1}/tmp_{Anh_name}'
    temp_scll = np.eye(3,dtype=int)
    strain_ref1 = np.dot(np.linalg.inv(my_atms1.get_cell()),ref_atoms_cell)-np.eye(3)
    strain_ref1[2,2] = 0

    # build the supercell model of the primitive cell 1
    SC_model_maker(har_xml1, anh_xml1, temp_scll, my_atms1, tmp_har_xml1 , tmp_anh_xml1,strain_in=strain_ref1,Higher_order_strain=Higher_order_strain ,missfit_strain=miss_fit_trms,scnd_order_strain=scnd_order_strain,elas_const_mul = elas_const_mul)

    cor_har_xml1,cor_anh_xml1 = get_xml_files(tmp_har_xml1,tmp_anh_xml1,ngqpt=ngqptm,ncell=xml_cell1,prt_dipdip=1,output_name='tmp_Str1',
                                    sim_path=sim_path1,Har_name=f'cor_{Har_name}',Anh_name=f'cor_{Anh_name}',EXEC=EXEC,NCPU=NCPU)
    
    # find the strain of the structure 2.                
    tmp_har_xml2 = f'{sim_path2}/tmp_{Har_name}'
    tmp_anh_xml2 = f'{sim_path2}/tmp_{Anh_name}'
    strain_ref2 = np.dot(np.linalg.inv(my_atms2.get_cell()),ref_atoms_cell)-np.eye(3)
    strain_ref2[2,2] = 0

    # build the supercell model of the primitive cell 2
    SC_model_maker(har_xml2, anh_xml2, temp_scll, trans_atms2, tmp_har_xml2 , tmp_anh_xml2,strain_in=strain_ref2,Higher_order_strain=Higher_order_strain ,missfit_strain=miss_fit_trms,scnd_order_strain=scnd_order_strain,elas_const_mul = elas_const_mul)

    cor_har_xml2,cor_anh_xml2 = get_xml_files(tmp_har_xml2,tmp_anh_xml2,ngqpt=ngqptm,ncell=xml_cell2,prt_dipdip=1,output_name='tmp_Str2',
                                    sim_path=sim_path2,Har_name=f'cor_{Har_name}',Anh_name=f'cor_{Anh_name}',EXEC=EXEC,NCPU=NCPU)
        
    # 
    SC_mat1 = [[1,0,0],[0,1,0],[0,0,ncell1[2]]]
    SC_mat2 = [[1,0,0],[0,1,0],[0,0,ncell2[2]]]
    har_xml = int_model_maker(cor_har_xml1,cor_anh_xml1,SC_mat1,cor_har_xml2,cor_anh_xml2,SC_mat2,symmetric=symmetric,
                                            har_file=Har_int,Anhar_file=Anh_int,negelect_A_SITE=negelect_A_SITE,
                                            negelect_Tot_FCs=negelect_Tot_FCs,sim_eps=sim_eps)

    SL_atms = har_xml.ref_cell 
    if if_return_atom:
        return(SL_atms,my_atms1,trans_atms2,avg_cell)
                      


