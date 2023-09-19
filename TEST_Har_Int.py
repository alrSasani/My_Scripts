import time
import interface_xmls  
import P_interface_xmls
import os
import my_functions
from ase import Atoms
thz_cm = 33.356/1.374673102
from phonopy.units import AbinitToTHz


if __name__=='__main__':
    My_EXEC = 'MB_16Jun'
    path_0 = os.getcwd()
    xmlf1 = f'{path_0}/src_xmls/Har_xml_8812'
    anh_file10 = f'{path_0}/src_xmls/AnHar_xml_8812'

    xmlf2 = f'{path_0}/src_xmls/Har_xml_8812'
    anh_file20 = f'{path_0}/src_xmls/AnHar_xml_8812'


    # xmlf2 = f'{path_0}/src_xmls/Har_xml_1048'
    # anh_file20 = f'{path_0}/src_xmls/AnHar_xml_1048'

    # xmlf2 = f'{path_0}/src_xmls/trans_Har_xml_1048'
    # anh_file20 = f'{path_0}/src_xmls/trans_AnHar_xml_1048'

    #term = 5
    symmetric = True
    Temp = 10
    for term in [0]:  # range(10):
        # anh_file1 = f'{path_0}/xmls_files/mani1_{term}.xml'
        # anh_terms_mini(xmlf1,anh_file10,output=anh_file1,terms_to_write=[term])

        # anh_file2 = f'{path_0}/xmls_files/mani2_{term}.xml'
        # anh_terms_mini(xmlf2,anh_file20,output=anh_file2,terms_to_write=[term])

        anh_file1 = anh_file10
        anh_file2 = anh_file20

        dim_a = 2  # for Phonons
        dim_b = 2  # for Phonons
        L1 = 2
        L2 = 1

        scmat = [[1, 0, 0], [0, 1, 0], [0, 0, L1]]
        scmat2 = [[1, 0, 0], [0, 1, 0], [0, 0, L2]]
        dim_z = L1+L2  #scmat[2][2]+scmat2[2][2]

        # ts = time.perf_counter()
        har_xml = interface_xmls.Har_interface(xmlf2, scmat, xmlf2,
                                scmat2, symmetric=symmetric)
        # har_xml.get_STR_FCDIC()
        # har_xml.reshape_FCDIC()
        # har_xml.write_xml(SL_xml)
        # tf = time.perf_counter()
        # print('TIME = ',tf-ts)
        UC_1 = har_xml.uc_atoms['0']


        distance = 0.01

        # Creating distorted structure:

        str_ten = [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]]
        phon_scll = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        SIM_cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        phonon_disp = my_functions.get_phonon(f'{xmlf1}', anh_file1, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                              8, 8, 8], UC=UC_1, my_EXEC='MB_16Jun', SIM_cell=SIM_cell, path=f'{path_0}/ref', disp_amp=distance)
        os.chdir(path_0)
        phonon_disp.run_random_displacements(Temp)
        rnd_disp = phonon_disp.random_displacements.u
        new_scld_pos = UC_1.get_scaled_positions()+rnd_disp
        new_UC = Atoms(numbers=UC_1.get_atomic_numbers(),
                       scaled_positions=new_scld_pos[0], cell=UC_1.get_cell())

        # new_UC = UC_1

        SL_xml = f'{path_0}/xmls_files/N_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        anh_int = f'{path_0}/xmls_files/{term}_test_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        interface_xmls.model_maker(xmlf1, anh_file1, scmat, xmlf2, anh_file2, scmat2,
                    symmetric=symmetric, har_file=SL_xml, Anhar_file=anh_int,negelect_A_SITE=True)

        str_ten = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
        distance = 0.2
        phon_scll = [[dim_a, 0, 0], [0, dim_b, 0], [0, 0, dim_z]]
        SIM_cell = [[dim_a, 0, 0], [0, dim_b, 0], [0, 0, 1]]
        # xml_old or anh_int

        int_phonon = my_functions.get_phonon(SL_xml, anh_int, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                         8, 8, 8], UC=new_UC, my_EXEC=My_EXEC, SIM_cell=SIM_cell, path=f'{path_0}/intf', disp_amp=distance)
        os.chdir(path_0)
        my_functions.plot_phonon(
            int_phonon, name=f'0_{term}_int_{dim_z}', cpath='./')

#########################################

        phon_scll = [[dim_a, 0, 0], [0, dim_b, 0], [0, 0, dim_z]]
        SIM_cell = [[dim_a, 0, 0], [0, dim_b, 0], [0, 0, dim_z]]

        phonon_disp = my_functions.get_phonon(f'{xmlf1}', anh_file1, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
                                              8, 8, 8], UC=new_UC, my_EXEC=My_EXEC, SIM_cell=SIM_cell, path=f'{path_0}/ref', disp_amp=distance)
        os.chdir(path_0)
        my_functions.plot_phonon(
            phonon_disp, name=f'0_{term}_SC_{dim_z}', cpath='./')
               


        # pordc_str_trms = False
        # SL_xml_old = f'{path_0}/xmls_files/Old_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'
        # anh_int_old = f'{path_0}/xmls_files/Old_{term}_test_{scmat[0][0]}{scmat[1][1]}{scmat[2][2]}.xml'

        # mdl = P_interface_xmls.har_interface(xmlf1, scmat, xmlf2, scmat2,)
        # mdl.reshape_FCDIC()
        # mdl.write_xml(SL_xml_old)
        # my_intr = P_interface_xmls.anh_intrface(
        #     xmlf1, anh_file1, scmat, xmlf2, anh_file2, scmat2, pordc_str_trms)
        # my_intr.wrt_anxml(anh_int_old)

        # #anh_int_old = 'no'

        # phonon_n = my_functions.get_phonon(SL_xml_old, anh_int_old, phon_scll=phon_scll, str_ten=str_ten, factor=AbinitToTHz*thz_cm, ngqpt=[
        #                                    8, 8, 8], UC=new_UC, my_EXEC='MB_16Jun', SIM_cell=SIM_cell, path=f'{path_0}/intf', disp_amp=distance)
        # os.chdir(path_0)
        # my_functions.plot_phonon(
        #     phonon_n, name=f'0_{term}_old_plt_{dim_z}', cpath='./')