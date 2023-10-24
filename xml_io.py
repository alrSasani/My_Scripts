import numpy as np
import xml.etree.ElementTree as ET
from ase import Atoms
from ase.units import Bohr
from ase.data import atomic_masses
import tools

###########################################################

class Xml_sys_reader:
    ''' Harmonic xml reader:
    This class is used to reas harmonic xml file  and return the data needed for construction of the SL'''
    def __init__(self, xml_file, mat_id='', extract_dta=False):
        self.id = mat_id
        tree = ET.parse(xml_file)
        self.root = tree.getroot()
        if extract_dta:
            self.get_atoms()
            self.get_ref_energy()
            self.get_eps_inf()
            self.get_ela_cons()
            self.get_str_cp()
            self.get_ase_atoms()
            self.get_loc_FC_tags()
            self.get_tot_FC_tags()
            self.set_tags()

    # atomic mass and Positions        >>>>>>>>             self.atm_pos          self.natm
    def get_atoms(self):
        self.natm = 0
        self.atm_pos = []
        for Inf in self.root.iter('atom'):
            mass = Inf.attrib['mass']
            # print(Inf.attrib['massunits'])
            pos = (Inf.find('position').text)
            chrgs = (Inf.find('borncharge').text)
            self.atm_pos.append([mass, pos, chrgs])
            self.natm += 1
        self.atm_pos = np.array(self.atm_pos)

    def get_BEC(self):
        self.get_atoms()
        atm_pos = self.atm_pos
        self.BEC = {}
        for i in range(len(self.atm_pos)):
            brn_tmp = [float(j) for j in atm_pos[i][2].split()[:]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            self.BEC[i] = brn_tmp

    # getting total forces    add a condition if exists   >>  self.tot_fc
    def get_tot_forces(self):
        self.has_tot_FC = 0
        self.tot_fc = {}
        cells = [[], [], []]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.tot_fc[cll_txt] = np.array([float(i) for i in data])
            cells[0].append(int(cll.split()[0]))
            cells[1].append(int(cll.split()[1]))
            cells[2].append(int(cll.split()[2]))

        if len(self.tot_fc) > 2:
            self.has_tot_FC = True
            self.tot_cells = [[min(cells[0]), max(cells[0])], [min(
                cells[1]), max(cells[1])], [min(cells[2]), max(cells[2])]]
        else:
            self.tot_cells = [[0, 0], [0, 0], [0, 0]]

    # getting local forces        add a condition if exists  >>>> self.loc_fc
    def get_loc_forces(self):
        self.loc_fc = {}
        cells = [[], [], []]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            data = (Inf.find('data').text).split()
            cll_txt = '{} {} {}'.format(
                cll.split()[0], cll.split()[1], cll.split()[2])
            self.loc_fc[cll_txt] = np.array([float(i) for i in data])
            cells[0].append(int(cll.split()[0]))
            cells[1].append(int(cll.split()[1]))
            cells[2].append(int(cll.split()[2]))
        self.loc_cells = [[min(cells[0]), max(cells[0])], [min(
            cells[1]), max(cells[1])], [min(cells[2]), max(cells[2])]]

    # refrence energy        >>>>     self.ref_eng
    def get_ref_energy(self):
        self.ref_eng = 0
        for Inf in self.root.iter('energy'):
            self.ref_eng = float(Inf.text)

    # Unit cell        >>>>    self.prim_vects
    def get_prim_vects(self):
        self.prim_vects = np.zeros((3, 3))
        for Inf in self.root.iter('unit_cell'):
            # print(Inf.attrib['units'])
            for i in range(3):
                self.prim_vects[i, :] = float((Inf.text).split(
                )[3*i+0]), float((Inf.text).split()[3*i+1]), float((Inf.text).split()[3*i+2])
        self.prim_vects = np.array(self.prim_vects)

    # epsilon infinity        >>>>     self.eps_inf
    def get_eps_inf(self):
        self.eps_inf = np.zeros((3, 3))
        for Inf in self.root.iter('epsilon_inf'):
            # print(Inf.attrib['units'])
            for i in range(3):
                self.eps_inf[i, :] = (Inf.text).split()[
                    3*i+0], (Inf.text).split()[3*i+1], (Inf.text).split()[3*i+2]
        self.eps_inf = np.array(self.eps_inf)

    # elastic constants        >>>>   self.ela_cons
    def get_ela_cons(self):
        ndim = 6
        self.ela_cons = np.zeros((ndim, ndim))
        for Inf in self.root.iter('elastic'):
            # print(Inf.attrib['units'])
            for i in range(6):
                self.ela_cons[i, :] = (Inf.text).split()[6*i+0], (Inf.text).split()[6*i+1], (Inf.text).split()[
                    6*i+2], (Inf.text).split()[6*i+3], (Inf.text).split()[6*i+4], (Inf.text).split()[6*i+5]
        self.ela_cons = np.array(self.ela_cons)

    # reading phonons data        >>>>     self.dymat     self.freq     self.qpoint
    def get_phonos(self):
        self.has_phonons = 0
        self.freq = []
        self.dymat = []
        for Inf in self.root.iter('phonon'):
            data = (Inf.find('qpoint').text)
            self.qpoint = (data)
            data = (Inf.find('frequencies').text).split()
            self.freq.append([float(i) for i in data])
            data = (Inf.find('dynamical_matrix').text).split()
            self.dymat.append([float(i) for i in data])
        self.freq = np.reshape(self.freq, (self.natm, 3))
        self.dymat = np.reshape(self.dymat, (3*self.natm, 3*self.natm))
        if len(self.freq) > 2:
            self.has_phonons = 1

    # reading strain phonon data        >>>>      self.corr_forc       self.strain
    def get_str_cp(self):
        self.corr_forc = {}
        self.strain = {}
        for Inf in self.root.iter('strain_coupling'):
            voigt = int(Inf.attrib['voigt'])
            data = (Inf.find('strain').text)
            # np.reshape(([float(i) for i in data]),(1,9))
            self.strain[voigt] = data
            data = (Inf.find('correction_force').text).split()
            self.corr_forc[voigt] = np.reshape(
                ([float(i) for i in data]), (self.natm, 3))

    # making ase_atoms object from the structure of the xml file
    def get_ase_atoms(self):
        self.get_prim_vects()
        self.get_atoms()
        self.get_str_cp()        
        atm_pos = self.atm_pos
        natm = self.natm
        car_pos = []
        amu = []
        BEC = []
        tag_id = []
        str_ph_cp = []
        for i in range(natm):
            str_ph_cp.append([])
            for j in range(6):
                str_ph_cp[i].append(self.corr_forc[j][i, :])
            amu.append(float(atm_pos[i][0]))
            car_pos.append([float(xx)*Bohr for xx in atm_pos[i][1].split()])
            brn_tmp = [float(j) for j in atm_pos[i][2].split()[:]]
            brn_tmp = np.reshape(brn_tmp, (3, 3))
            BEC.append(brn_tmp)
            tag_id.append([i, self.id])
            # tag.append(i)
        atom_num = [get_atom_num(x) for x in amu]
        self.ase_atoms = Atoms(numbers=atom_num, positions=np.array(
            car_pos), cell=Bohr*self.prim_vects, pbc=True)
        self.ase_atoms.set_array('BEC', np.array(BEC))
        self.ase_atoms.set_array('tag_id', np.array(tag_id))
        self.ase_atoms.set_array('str_ph', np.array(str_ph_cp))
        #self.ase_atoms.set_array('tag', tag)
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
                cntr_list[f'{my_symbols[i]}_cntr'] += 1
            else:
                cntr_list[f'{my_symbols[i]}_cntr'] = 1
            if cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i] > 1:
                my_char = 1
            elif cntr_list[f'{my_symbols[i]}_cntr'] == 1 and counts[i] == 1:
                my_char = ''
            else:
                my_char = cntr_list[f'{my_symbols[i]}_cntr']
            my_tags.append(f'{my_symbols[i]}{my_char}')
        self.tags = my_tags

    def get_loc_FC_tags(self):
        self.get_ase_atoms()
        # tags = self.ase_atoms.get_array('tags') #self.tags
        tag_id = self.ase_atoms.get_array('tag_id')
        # print(tag_id)
        self.loc_FC_tgs = {}
        natm = self.natm  # self.ase_atoms.get_global_number_of_atoms()
        self.get_loc_forces()
        # lc_fc=self.loc_fc
        for key in self.loc_fc.keys():
            my_cell = [int(x) for x in key.split()]
            my_fc = np.reshape(self.loc_fc[key], (3*natm, 3*natm))
            for atm_a in range(natm):
                for atm_b in range(natm):
                    # print(tag_id[atm_a][0])
                    FC_mat = np.zeros((3, 3))
                    int_key = f'{tag_id[atm_a][1]}{tag_id[atm_a][0]}_{tag_id[atm_b][1]}{tag_id[atm_b][0]}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
                    FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                    self.loc_FC_tgs[int_key] = FC_mat

    # this function return the total force constants in the xml file as dictionary
    def get_tot_FC_tags(self):
        self.get_ase_atoms()
        #tags = self.ase_atoms.get_array('tags')
        tag_id = self.ase_atoms.get_array('tag_id')
        self.get_tot_forces()
        self.tot_FC_tgs = {}
        if self.has_tot_FC:
            natm = self.natm  # self.ase_atoms.get_global_number_of_atoms()
            # tot_fc=self.tot_fc
            for key in self.tot_fc.keys():
                my_cell = [int(x) for x in key.split()]
                my_fc = np.reshape(self.tot_fc[key], (3*natm, 3*natm))
                for atm_a in range(natm):
                    for atm_b in range(natm):
                        FC_mat = np.zeros((3, 3))
                        int_key = f'{tag_id[atm_a][1]}{tag_id[atm_a][0]}_{tag_id[atm_b][1]}{tag_id[atm_b][0]}_{my_cell[0]}_{my_cell[1]}_{my_cell[2]}'
                        FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                        self.tot_FC_tgs[int_key] = FC_mat


    def get_loc_FC_dic(self):
        self.loc_FC_dic = {}
        self.get_ase_atoms()
        natm = self.ase_atoms.get_global_number_of_atoms()
        self.get_loc_forces()
        for key in self.loc_fc.keys():
            my_cell = [int(x) for x in key.split()]
            my_fc = np.reshape(self.loc_fc[key], (3*natm, 3*natm))
            for atm_a in range(natm):
                for atm_b in range(natm):
                    FC_mat = np.zeros((3, 3))
                    int_key = '{}_{}_{}_{}_{}'.format(
                        atm_a, atm_b, my_cell[0], my_cell[1], my_cell[2])
                    FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                    self.loc_FC_dic[int_key] = FC_mat

    def get_tot_FC_dic(self):
        self.get_tot_forces()
        self.tot_FC_dic = {}
        if self.has_tot_FC:
            self.get_ase_atoms()
            natm = self.ase_atoms.get_global_number_of_atoms()
            for key in self.tot_fc.keys():
                my_cell = [int(x) for x in key.split()]
                my_fc = np.reshape(self.tot_fc[key], (3*natm, 3*natm))
                for atm_a in range(natm):
                    for atm_b in range(natm):
                        FC_mat = np.zeros((3, 3))
                        int_key = '{}_{}_{}_{}_{}'.format(
                            atm_a, atm_b, my_cell[0], my_cell[1], my_cell[2])
                        FC_mat = my_fc[atm_a*3:atm_a*3+3, atm_b*3:atm_b*3+3]
                        self.tot_FC_dic[int_key] = FC_mat

    # number of cells in the super cell and local forces    >>>>  self.ncll

    def get_Per_clls(self):
        self.ncll = [0, 0, 0]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            if cll.split()[1] == '0' and cll.split()[2] == '0':
                self.ncll[0] += 1
            if cll.split()[0] == '0' and cll.split()[2] == '0':
                # print(cll)
                self.ncll[1] += 1
            if cll.split()[0] == '0' and cll.split()[1] == '0':
                self.ncll[2] += 1

    # number of cells in the super cell and local forces    >>>>  self.loc_cells
    def get_loc_cells(self):
        self.loc_cells = [[0, 0], [0, 0], [0, 0]]
        for Inf in self.root.iter('local_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell = [int(x) for x in cll.split()]
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

    # number of cells in the super cell and local forces    >>>>  self.loc_cells

    def get_tot_cells(self):
        self.tot_cells = [[0, 0], [0, 0], [0, 0]]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            my_cell = [int(x) for x in cll.split()]
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

    # number of cells in the super cell and local forces    >>>>  self.ncll
    def get_tot_Per_clls(self):
        self.tot_ncll = [0, 0, 0]
        for Inf in self.root.iter('total_force_constant'):
            cll = (Inf.find('cell').text)
            if cll.split()[1] == '0' and cll.split()[2] == '0':
                self.tot_ncll[0] += 1
            if cll.split()[0] == '0' and cll.split()[2] == '0':
                # print(cll)
                self.tot_ncll[1] += 1
            if cll.split()[0] == '0' and cll.split()[1] == '0':
                self.tot_ncll[2] += 1

def xml_anha_reader(fname, atoms):
    tree = ET.parse(fname)
    root = tree.getroot()
    coeff = {}
    lst_trm = []
    car_pos = atoms.get_positions()
    abc = atoms.cell.cellpar()[0:3]
    for cc, inf in enumerate(root.iter('coefficient')):
        coeff[cc] = inf.attrib
        lst_trm.append([])
        for tt, trm in enumerate(inf.iter('term')):
            lst_trm[cc].append([])
            # print(trm.attrib)
            disp_cnt = 0
            strcnt = 0
            for dd, disp in enumerate(trm.iter('displacement_diff')):
                disp_cnt += 1
                dta = disp.attrib
                atma = int(disp.attrib['atom_a'])
                atmb = int(disp.attrib['atom_b'])
                cella = (disp.find('cell_a').text)
                cellb = (disp.find('cell_b').text)
                cell_a = [float(x) for x in cella.split()]
                cell_b = [float(x) for x in cellb.split()]
                cell_a_flg = [True if l != 0 else False for l in cell_a]
                # print(abc[0])
                if any(cell_a_flg):
                    cell_a -= cell_a
                    cell_b = cell_b-cell_a
                    cella = f'{cell_a[0]} {cell_a[1]} {cell_a[2]}'
                    cellb = f'{cell_b[0]} {cell_b[1]} {cell_b[2]}'

                pos_a = [cell_a[0]*abc[0]+car_pos[atma, 0], cell_a[1] *
                         abc[1]+car_pos[atma, 1], cell_a[2]*abc[2]+car_pos[atma, 2]]
                pos_b = [cell_b[0]*abc[0]+car_pos[atmb, 0], cell_b[1] *
                         abc[1]+car_pos[atmb, 1], cell_b[2]*abc[2]+car_pos[atmb, 2]]
                dist = ((pos_a[0]-pos_b[0])**2+(pos_a[1]-pos_b[1])
                        ** 2+(pos_a[2]-pos_b[2])**2)**0.5
                dta['cell_a'] = cella
                dta['cell_b'] = cellb
                dta['weight'] = trm.attrib['weight']
                lst_trm[cc][tt].append(dta)
            if disp_cnt == 0:
                # dta={}
                dist = 0
                dta = trm.attrib
                lst_trm[cc][tt].append(dta)
            for ss, strain in enumerate(trm.iter('strain')):
                dta = strain.find('strain')
                lst_trm[cc][tt].append(strain.attrib)
                strcnt += 1

            lst_trm[cc][tt].append(
                {'dips': disp_cnt, 'strain': strcnt, 'distance': dist})
    return(coeff, lst_trm)

def get_atom_num(atomic_mass, tol=0.1):
    if abs(atomic_mass-208) < 1:
        tol = 0.001
    for i in range(len(atomic_masses)):
        if abs(atomic_masses[i]-atomic_mass) < tol:
            mynum = i
    return(mynum)

def write_anh_xml(coeff,trms,fout):
    output = open(fout, 'w')
    output.write('<?xml version="1.0" ?>\n')
    output.write('<Heff_definition>\n')
    for i, key in enumerate(coeff.keys()):
        output.write('  <coefficient number="{}" value="{}" text="{}">\n'.format(
            coeff[key]['number'], coeff[key]['value'], coeff[key]['text'],))
        for j in range(len(trms[i])):
            for k in range(int(trms[i][j][-1]['dips'])):
                if (k == 0):
                    output.write('    <term weight="{}">\n'.format(
                        trms[i][j][k]['weight']))
                output.write('      <displacement_diff atom_a="{}" atom_b="{}" direction="{}" power="{}">\n'.format(
                    trms[i][j][k]['atom_a'], trms[i][j][k]['atom_b'], trms[i][j][k]['direction'], trms[i][j][k]['power']))
                output.write(
                    '        <cell_a>{}</cell_a>\n'.format(trms[i][j][k]['cell_a']))
                output.write(
                    '        <cell_b>{}</cell_b>\n'.format(trms[i][j][k]['cell_b']))
                output.write('      </displacement_diff>\n')
            if int(trms[i][j][-1]['dips']) == 0:
                k = int(trms[i][j][-1]['strain'])
                output.write('    <term weight="{}">\n'.format(
                    trms[i][j][k]['weight']))
                k = -1
            for l in range(int(trms[i][j][-1]['strain'])):
                output.write('      <strain power="{}" voigt="{}"/>\n'.format(
                    trms[i][j][k+l+1]['power'], trms[i][j][k+l+1]['voigt']))
            output.write('    </term>\n')
        output.write('  </coefficient>\n')
    output.write('</Heff_definition>\n')    

def write_sys_xml(dta,out_put):
    # ncll = dta['ncell']
    keys = dta['keys']
    SCL_elas = dta['SCL_elas'] 
    ref_eng = dta['SCL_ref_energy']
    SCL_lat = dta['SCL_lat']
    eps_inf = dta['eps_inf']
    atom_mass = dta['atoms_mass']
    SC_BEC = dta['SC_BEC']
    SC_atoms_pos = dta['SC_atoms_pos']
    SC_FC = dta['SC_local_FC']
    tSC_FC = dta['SC_total_FC']
    has_tot_FC = dta['has_tot_FC']
    SC_CorForc = dta['SC_corr_forc']
    strain = dta['strain']
    my_atm_list = dta['my_atm_list']

    out_xml = open(out_put, 'w')
    out_xml.write('<?xml version="1.0" ?>\n')
    out_xml.write('<System_definition>\n')
    out_xml.write('  <energy>\n  {:.14E}\n  </energy>\n'.format(
        ref_eng))  
    out_xml.write(
        '  <unit_cell units="bohrradius">\n {}  </unit_cell>\n'.format(tools.to_text(SCL_lat))) 
    out_xml.write(
        '  <epsilon_inf units="epsilon0">\n  {}  </epsilon_inf>\n'.format(tools.to_text(eps_inf)))
    out_xml.write(
        '  <elastic units="hartree">\n  {}  </elastic>\n'.format(tools.to_text(SCL_elas)))
    for ii in my_atm_list:

        out_xml.write('  <atom mass="  {}" massunits="atomicmassunit">\n    <position units="bohrradius">\n   {}</position>\n    <borncharge units="abs(e)">\n  {}</borncharge>\n  </atom>\n'.format(
            atom_mass[ii], tools.one_text(SC_atoms_pos[ii, :]), tools.to_text(SC_BEC[ii])))

    for key in keys:
        if key in (SC_FC.keys()):
            out_xml.write(
                '  <local_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </local_force_constant>\n'.format(tools.to_text((SC_FC[key])), key))
        if has_tot_FC:
            out_xml.write(
                '  <total_force_constant units="hartree/bohrradius**2">\n    <data>\n {}    </data> \n    <cell>\n  {}\n    </cell>\n  </total_force_constant>\n'.format(tools.to_text((tSC_FC[key])), key))

    # xml.write('  <phonon>\n    <qpoint units="2pi*G0">  {}</qpoint>\n    <frequencies units="reciprocal cm">\n  {}    </frequencies>\n    <dynamical_matrix units="hartree/bohrradius**2">\n {}    </dynamical_matrix>\n   </phonon>\n'.format(self.xml.qpoint,tools.to_text(SC_phon),tools.to_text(SC_dmat)))
    for i in range(len(strain)):
        out_xml.write('  <strain_coupling voigt=" {}">\n    <strain>  {}    </strain>\n    <correction_force units="hartree/bohrradius">\n  {}    </correction_force>\n  </strain_coupling>\n'.format(
            i, (strain[i]), tools.to_text(SC_CorForc[i])))
    out_xml.write('</System_definition>')
    out_xml.close()    

