import SC_xml_potential as SC_pot
import numpy as np
import interface_xmls


def SC_model_maker(my_harf,my_Anhf,scll,my_atoms,har_out,anh_out,strain_in=np.zeros((3,3)),missfit_strain=True,Higher_order_strain=False):
    sc_maker=SC_pot.Har_sc_maker(my_harf,scll,strain_in)
    sc_maker.reshape_FCDIC(my_atoms)
    sc_maker.write_xml(har_out)
    anh_SCxml=SC_pot.Anh_sc_maker(my_harf,my_Anhf,strain_in,missfit_strain=missfit_strain,Higher_order_strain=Higher_order_strain)
    anh_SCxml.SC_trms(my_atoms,scll)
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
    return(STRC)    