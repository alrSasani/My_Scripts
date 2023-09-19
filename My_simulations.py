import spglib as spg
from os.path import exists
import mync
import os
import sys
sys.path.append("/home/alireza/CODES/My_scr")
# import interface_xmls
# import SC_xml_potential

class MB_sim():
    def __init__(self, EXEC, har_coeffs, Anhar_coeffs='no', ngqpt=[4, 4, 4], ncell=[2, 2, 2], ncpu=1, test_set='no', prefix='MB'):
        self.har_coeffs = har_coeffs
        self.Anhar_coeffs = Anhar_coeffs
        self.EXEC = EXEC
        self.ngqpt = ngqpt
        self.ncell = ncell
        self.ncpu = ncpu
        self.test_set = test_set
        self.prefix = prefix

    def run_MB(self, abi_files='abi.files'):
        paralel = False
        if self.ncpu > 1:
            paralel = True
        my_file = open('MB_run.sh', 'w')
        if paralel:
            my_file.write(
                f'mpirun -np {self.ncpu} {self.EXEC} < {abi_files} > log')
        else:
            my_file.write(f'{self.EXEC} < {abi_files} > log')
        #os.system(f'sh MB_run.sh')

    def common_data(self):
        self.common_data = {
            'prt_model': 0,
            'ngqpt':  self.ngqpt,
            'dipdip': 1,
            # 'dipdip_range': self.ncell,
            'ncell': self.ncell,
            'dipdip_prt': 1,
            'asr': 2,
        }

    # self.xml_inpt
    def inpt_file_xml(self, prt_model=2, prt_dipdip=1):
        self.common_data()
        temp_inp = {
            'prt_model': prt_model,
            'dipdip_prt': prt_dipdip,
            'ntime': 1,
            'dynamics': 25,
            'temperature': 1,
            'dtion': 30,
            'restartxf': 0,
            'optcell':  2,
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.xml_inpt
    def inpt_bounding(self, range_power=[4, 6], bound_penalty=1.001):
        self.common_data()
        temp_inp = {
            'bound_model': 3,
            'bound_rangePower': range_power,
            'bound_EFS': [1, 1, 1],
            'bound_penalty': bound_penalty,
            'bound_factors': [1, 1, 1],
            'bound_anhastrain': 1,
            'bound_spcoupling': 1,
            'bound_cell': 0,
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.xml_inpt
    def fit_inpt(self, prt_model=4):
        self.common_data()
        temp_inp = {
            'coeff_file_rw': 0,
            'prt_model': prt_model,
            'fit_initializedata': 0,
            'fit_coeff': 1,
            'fit_ncoeff': 10,
            'fit_generatecoeff': 1,
            'fit_rangePower': [3, 4],
            'fit_cutoff': 6.5,
            'fit_dispterms': 1,
            'fit_anhaStrain': 1,
            'fit_SPCoupling': 1,
            'fit_SPC_maxS': 1,
            'fit_iatom': -2,
            'fit_EFS': [0, 1, 1],
            'fit_factors': [0, 1, 1],
            'fit_nbancoeff': 0,
            'fit_bancoeff': [0],
            'fit_nfixcoeff': 0,
            'fit_fixcoeff': [0],
            'fit_nimposecoeff': 0,
            'fit_imposecoeff':  [0],
            # 'fit_ts_option'  : 0,
            # 'sel_EFS' : [1,1,1],
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.test_inpt
    def test_dta(self):
        self.common_data()
        temp_inp = {
            'test_prt_ph': 1,
            'test_effpot': 1,
            'analyze_anh_pot': 1,
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.rlx_inpt
    def rlx_dta(self, opt_cell=2, efield=[0, 0, 0], restartxf=-3):
        self.common_data()
        temp_inp = {
            'dynamics': 2,
            'ntime': 300,
            'nctime': 1,
            'temperature': 1,
            'dtion': 30,
            'restartxf': restartxf,
            'optcell':  opt_cell,
            'efield': efield,
            'dyn_chksym': 0
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.mc_inpt
    def MC_dta(self, ntime, Temperature=1, restartxf=0, optcell=0, efield=[0, 0, 0], nctime=30, hmcsst = 0,hmctt=30):
        self.common_data()
        temp_inp = {
            'dynamics': 25,
            'ntime': ntime,
            'hmctt': hmctt,
            'hmcsst' : hmcsst,            
            'nctime': nctime,
            'temperature': Temperature,
            'dtion': 30,
            'restartxf': restartxf,
            'optcell':  optcell,
            'efield': efield,
            'dyn_chksym': 0
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.md_inpt
    def MD_dta(self, ntime, dynamics=13 ,Temperature=1, restartxf=0, optcell=0, efield=[0, 0, 0], nctime=30):
        self.common_data()
        temp_inp = {
            'dynamics': dynamics,
            'ntime': ntime,
            'nctime': nctime,
            'temperature': Temperature,
            'dtion': 30,
            'restartxf': restartxf,
            'optcell':  optcell,
            'efield': efield,
            'dyn_chksym':  0
        }
        self.inpt = {**self.common_data, **temp_inp}

    # self.inp_text
    def get_inpt_txt(self):
        self.inp_text = ''
        for i in self.inpt.keys():
            val = ''
            if type(self.inpt[i]) in (int, float):
                tmp_text = ('{}  =  {} \n'.format(i, self.inpt[i]))
            else:
                val = ' '.join(list(map(str, self.inpt[i])))
                tmp_text = (f'{i}  = {val} \n')
            self.inp_text += tmp_text

    # self.files_txt
    def files_text(self, tmp_files='no'):
        self.tmp_files = tmp_files
        self.files_txt = ''
        self.files_txt += (f'{self.prefix}.in\n')
        self.files_txt += (f'{self.prefix}\n')
        self.files_txt += (f'{self.har_coeffs}\n')
        self.files_txt += (f'{self.Anhar_coeffs}\n')
        self.files_txt += (f'{self.tmp_files}\n')
        self.files_txt += (f'{self.test_set}\n')

    def write_inpt(self, file_name='no'):
        if file_name == 'no':
            file_name = f'{self.prefix}.in'
        self.get_inpt_txt()
        my_file = open(f'{file_name}', 'w')
        print(self.inp_text, file=my_file)
        my_file.close()

    def write_files(self, file_name):
        try:
            tmp_files = self.tmp_files
        except:
            tmp_files = 'no'
        self.files_text(tmp_files)
        my_file = open(f'{file_name}', 'w')
        print(self.files_txt, file=my_file)
        my_file.close()

    def set_prim_strc_nc(self, nc_file, step=-1, avg=False):
        NC_dta = mync.hist_reader(nc_file)
        if avg:
            if step == -1:
                step = 0
            self.Prim_str = NC_dta.get_avg_str(initial=step)
        else:
            self.Prim_str = NC_dta.get_ase_str(i=step)

    def set_prim_strc(self, strc):
        self.Prim_str = strc

    def write_hist(self, nc_file='no'):
        if nc_file == 'no':
            self.nc_file = f'{self.prefix}_HIST.nc'
        try:
            mync.write_hist(self.Prim_str, self.nc_file, t=0)
            self.inpt['restartxf'] = -3
        except:
            raise ValueError('No primary structure are defined')

    def create_path(self, my_path, prim_path='no'):
        rep_path = my_path.replace('/', ' ')
        if prim_path == 'no':
            path_0 = ''
        else:
            path_0 = prim_path
        for fldr in rep_path.split():
            if not exists(f'{path_0}/{fldr}'):
                os.system(f'mkdir {path_0}/{fldr}')
                path_0 = f'{path_0}/{fldr}'
            else:
                path_0 = f'{path_0}/{fldr}'

    def write_run_data(self, files_inpt='abi.files'):
        self.write_inpt()
        self.write_files(f'{files_inpt}')
        self.run_MB(abi_files=f'{files_inpt}')

def get_number_SG(sg):
    wrds = sg.split()
    return(wrds[1][1:-1])


def relax_NC_Str(path0, har_xml, Anhar_coeffs, nc_file, EXEC='MB_Mar20', ngqpt=[8, 8, 8], ncell=[2, 2, 2], NCPU=8, Files_prefix='BTO', opt_cell=[2], step=-1):
    strc = mync.get_NC_str(nc_file, stp=step)
    my_sim = MB_sim(EXEC, har_xml, Anhar_coeffs=Anhar_coeffs, ngqpt=ngqpt,
                    ncell=ncell, ncpu=NCPU, test_set='no', prefix=Files_prefix)
    my_sim.rlx_dta()
    for dd in opt_cell:
        SPG = get_number_SG(spg.get_spacegroup(strc))
        sim_path = f'{path0}/SG_{SPG}/dd_{dd}'
        os.makedirs(sim_path)
        # my_sim.create_path(sim_path)
        os.chdir(sim_path)
        my_sim.inpt['optcell'] = dd
        my_sim.set_prim_strc_nc(nc_file, step=step, avg=False)
        my_sim.write_hist()
        my_sim.write_run_data()
        os.system(f'sh MB_run.sh')
