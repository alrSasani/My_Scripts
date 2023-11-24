import SC_xml_potential as SC_pot
import numpy as np
import interface_xmls
import xml_io
from My_simulations import get_xml_files
import tools
from ase.build import sort

def SC_model_maker(my_harf,my_Anhf,scll,my_SCll,har_out,anh_out,strain_in=np.zeros((3,3)),missfit_strain=True,Higher_order_strain=False):
    sc_maker=SC_pot.Har_sc_maker(my_harf,scll,strain_in)
    sc_maker.reshape_FCDIC(my_SCll)
    sc_maker.write_xml(har_out)
    anh_SCxml=SC_pot.Anh_sc_maker(my_harf,my_Anhf,strain_in,missfit_strain=missfit_strain,Higher_order_strain=Higher_order_strain)
    anh_SCxml.SC_trms(my_SCll,scll)
    anh_SCxml.wrt_anxml(anh_out)

def int_model_maker(xmlf1, anh_file1, scmat1, xmlf2, anh_file2, scmat2, symmetric=False, har_file='int_harmoni.xml', Anhar_file='int_harmoni.xml',
                negelect_A_SITE=False,negelect_Tot_FCs=False, NW_Strc = False):
    # Harmonic_term generation
    har_xml = interface_xmls.Har_interface(xmlf1, scmat1, xmlf2, scmat2, symmetric=symmetric,negelect_A_SITE=negelect_A_SITE,negelect_Tot_FCs=negelect_Tot_FCs,NW_Strc = NW_Strc)
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

def SL_MAKER(DDB1,modle1,ncell1,DDB2,modle2,ncell2,ngqptm,sim_path1='./M1',sim_path2='./M2',NCPU=1,ref_cell='M2',Har_int='Har_int',Anh_int='Anh_int',
             miss_fit_trms=True,Higher_order_strain=True,negelect_A_SITE=True,negelect_Tot_FCs=True,symmetric=True,xml_cell1=None,
             xml_cell2=None,if_return_atom=False):
    
    print('making SL Pot')

    if xml_cell1==None:
        xml_cell1 = [ncell1[0],ncell1[1],ncell1[2]+ncell2[2]]
    else:
        xml_cell1=xml_cell1
    if xml_cell2==None:
        xml_cell2 = [ncell1[0],ncell1[1],ncell1[2]+ncell2[2]]
    else:
        xml_cell2=xml_cell2

    Har_name = f'Har_xml_{xml_cell1[0]}{xml_cell1[1]}{xml_cell1[2]}'           
    Anh_name = f'AnHar_xml_{xml_cell1[0]}{xml_cell1[1]}{xml_cell1[2]}'

    # SR_cd_dir = max(ngqpt1[0],ngqpt2[0])
    har_xml1,anh_xml1 = get_xml_files(DDB1,modle1,ngqpt=ngqptm,ncell=xml_cell1,prt_dipdip=1,output_name='Str1',
                                      sim_path=sim_path1,Har_name=Har_name,Anh_name=Anh_name,EXEC='MB_16Jun',NCPU=NCPU)


    har_xml2,anh_xml2 = get_xml_files(DDB2,modle2,ngqpt=ngqptm,ncell=xml_cell2,prt_dipdip=1,output_name='Str2',
                                      sim_path=sim_path2,Har_name=Har_name,Anh_name=Anh_name,EXEC='MB_16Jun',NCPU=NCPU)
    

    myxml1=xml_io.Xml_sys_reader(har_xml1)
    myxml1.get_ase_atoms()
    my_atms1=myxml1.ase_atoms
    myxml2=xml_io.Xml_sys_reader(har_xml2)
    myxml2.get_ase_atoms()
    my_atms2=myxml2.ase_atoms

    my_atm_list_tmp = tools.mapping(my_atms1,my_atms2)
    trans_atms2 = sort(my_atms2,tags = my_atm_list_tmp) 


    if ref_cell == 'M1':
        print(f'{my_atms1.get_chemical_formula()} as refrence')
        ref_atoms_cell = my_atms1.get_cell()
    elif ref_cell == 'M2':
        print(f'{my_atms2.get_chemical_formula()} as refrence')
        ref_atoms_cell = my_atms2.get_cell()
    elif ref_cell == 'avg':
        print(f'aerage cells of {my_atms1.get_chemical_formula()} and {my_atms2.get_chemical_formula()} as refrence')
        ref_atoms_cell = get_avg_cell_SL(har_xml1,ncell1[2],har_xml2,ncell2[2])

    print('refrence cell in SL is: \n',ref_atoms_cell)

    tmp_har_xml1 = f'{sim_path1}/tmp_{Har_name}'
    tmp_anh_xml1 =f'{sim_path1}/tmp_{Anh_name}'
    temp_scll = np.eye(3,dtype=int)
    strain_ref1 = np.dot(np.linalg.inv(my_atms1.get_cell()),ref_atoms_cell)-np.eye(3)
    strain_ref1[2,2] = 0

    SC_model_maker(har_xml1, anh_xml1, temp_scll, my_atms1, tmp_har_xml1 , tmp_anh_xml1,strain_in=strain_ref1,Higher_order_strain=Higher_order_strain ,missfit_strain=miss_fit_trms)

    cor_har_xml1,cor_anh_xml1 = get_xml_files(tmp_har_xml1,tmp_anh_xml1,ngqpt=ngqptm,ncell=xml_cell1,prt_dipdip=1,output_name='tmp_Str1',
                                    sim_path=sim_path1,Har_name=f'cor_{Har_name}',Anh_name=f'cor_{Anh_name}',EXEC='MB_16Jun',NCPU=NCPU)
    
                
    tmp_har_xml2 = f'{sim_path2}/tmp_{Har_name}'
    tmp_anh_xml2 = f'{sim_path2}/tmp_{Anh_name}'
    strain_ref2 = np.dot(np.linalg.inv(my_atms2.get_cell()),ref_atoms_cell)-np.eye(3)
    strain_ref2[2,2] = 0

    SC_model_maker(har_xml2, anh_xml2, temp_scll, trans_atms2, tmp_har_xml2 , tmp_anh_xml2,strain_in=strain_ref2,Higher_order_strain=Higher_order_strain ,missfit_strain=miss_fit_trms)

    cor_har_xml2,cor_anh_xml2 = get_xml_files(tmp_har_xml2,tmp_anh_xml2,ngqpt=ngqptm,ncell=xml_cell2,prt_dipdip=1,output_name='tmp_Str2',
                                    sim_path=sim_path2,Har_name=f'cor_{Har_name}',Anh_name=f'cor_{Anh_name}',EXEC='MB_16Jun',NCPU=NCPU)
        
    SC_mat1 = [[1,0,0],[0,1,0],[0,0,ncell1[2]]]
    SC_mat2 = [[1,0,0],[0,1,0],[0,0,ncell2[2]]]
    har_xml = int_model_maker(cor_har_xml1,cor_anh_xml1,SC_mat1,cor_har_xml2,cor_anh_xml2,SC_mat2,symmetric=symmetric,
                                            har_file=Har_int,Anhar_file=Anh_int,negelect_A_SITE=negelect_A_SITE,
                                            negelect_Tot_FCs=negelect_Tot_FCs)

    SL_atms = har_xml.ref_cell 
    if if_return_atom:
        return(SL_atms,my_atms1,trans_atms2)
                      


