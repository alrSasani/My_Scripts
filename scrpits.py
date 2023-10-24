import SC_xml_potential as SC_pot
import numpy as np


def write_SC_pot(my_harf,my_Anhf,scll,my_atoms,har_out,anh_out,strain_in=np.zeros((3,3)),missfit_strain=True,Higher_order_strain=False):
    sc_maker=SC_pot.Har_sc_maker(my_harf,scll,strain_in)
    sc_maker.reshape_FCDIC(my_atoms)
    sc_maker.write_xml(har_out)
    anh_SCxml=SC_pot.Anh_sc_maker(my_harf,my_Anhf,strain_in,missfit_strain=missfit_strain,Higher_order_strain=Higher_order_strain)
    anh_SCxml.SC_trms(my_atoms,scll)
    anh_SCxml.wrt_anxml(anh_out)