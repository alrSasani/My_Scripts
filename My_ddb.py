import abipy
from collections import OrderedDict, namedtuple
import re
import os
import numpy as np
from scipy.linalg import eigh
from ase.symbols import string2symbols
from ase.data import atomic_masses, atomic_numbers
from ase.units import Ha, Bohr
from ase import Atoms




#masses = [atomic_masses[atomic_numbers[s]] for s in symbols]
class ddb_reader():

    def __init__(self, fname):
        self.fname = fname
        with open(self.fname) as myfile:
            self.lines = myfile.readlines()
        self.header, self.data = self.split_head()
#        self.get_struct()

    def split_head(self):
        finished=False
        self.secdd_line=[]
        line_counter=0
        header = []
        data = []
        in_header = True
        for line in self.lines:
            if line.strip().startswith(
                    '**** Database of total energy derivatives'):
                in_header = False
            if in_header:
                header.append(line)
            else:
                line_counter+=1
                if line.strip().startswith(
                    'Number of data blocks='):
                    self.nblocs=int(line.strip().split()[-1])
                if line.strip().startswith(
                    'List of bloks'):
                    finished=True
                if line.strip().startswith(
                    '2nd derivatives') and not(finished):
                    self.secdd_line.append(line_counter-1)
                data.append(line)
        return header, data

    def get_struct(self):
        for i,ii in enumerate(self.header):
            if ii.strip().startswith(
                'natom'):
                self.natom=int(ii.strip().split()[-1])
                self.typat=np.zeros((self.natom))
                self.xred=np.zeros((self.natom,3))
            if ii.strip().startswith(
                'ntypat'):
                self.ntypat=int(ii.strip().split()[-1])
                self.masses=np.zeros((self.ntypat))
                self.znucl=np.zeros((self.ntypat))
            if ii.strip().startswith(
                'amu'):
                x=[float(x.replace('D','E')) for x in ii.strip().split()[1:]]
                #print(x)
                self.masses[:]=[float(x.replace('D','E')) for x in ii.strip().split()[1:]]
                #if len(self.masses)<self.ntypat:
                #    self.masses.append((float(x.replace('D', 'E')) for x in self.header[i+1].strip().split()[1:]))
            if ii.strip().startswith(
                 'typat'):
                self.typat[:]=[int(x) for x in ii.strip().split()[1:]]
                #if len(self.typat)<self.natom:
                #    self.typat[0:]=((int(x.replace('D', 'E')) for x in self.header[i+1].strip().split()[1:]))
            if ii.strip().startswith(
                'acell'):
                self.acell=np.zeros((3))
                self.acell[:]=[float(x.replace('D','E')) for x in ii.strip().split()[1:]]
            if ii.strip().startswith(
                'rprim'):
                self.rprim=np.zeros((3,3))
                self.rprim[0,:]=[float(x.replace('D', 'E')) for x in ii.strip().split()[1:]]
                for ccll in range(2):
                    self.rprim[ccll+1,:]=[float(x.replace('D', 'E')) for x in self.header[i+ccll+1].strip().split()[:]]
            if ii.strip().startswith(
                 'xred'):
                 self.xred[0,:]=[float(x.replace('D', 'E')) for x in ii.strip().split()[1:]]
                 for catm in range(self.natom-1):
                     self.xred[catm+1,:]=[float(x.replace('D', 'E')) for x in self.header[i+catm+1].strip().split()[:]]
            if ii.strip().startswith(
                 'znucl'):
                self.znucl[:]=[float(x.replace('D','E')) for x in ii.strip().split()[1:]]



    def get_atoms(self):
        #pass
        self.get_struct()
        self.get_MassMat()
        self.atoms = Atoms(numbers=self.znumat, scaled_positions=self.xred, cell=self.acell*self.rprim*Bohr, pbc=True)

    def get_MassMat(self):
        self.get_struct()
        self.massmat=np.zeros((self.natom))
        self.znumat=np.zeros((self.natom))
        for i in range(self.natom):
            #pass
            #print(self.typat[i]-1)
            self.massmat[i]=(self.masses[int(self.typat[i]-1)])
            self.znumat[i]=(self.znucl[int(self.typat[i]-1)])

    def read_elem(self):
        qdym=[]
#        ds = self.data[5:]
        for d, dd in enumerate(self.secdd_line):
            dym = {}
            nelements=int(self.data[dd].strip().split()[-1])
            qpt=[float(x) for x in self.data[dd+1].strip().split()[1:4]]
            dym['qpt']=qpt
            for elm in range(nelements):
                nelm=elm+dd+2
                idir1, ipert1, idir2, ipert2 = [
                    int(x) for x in self.data[nelm].strip().split()[0:4]]
                val = float(self.data[nelm].strip().split()[4].replace('D', 'E'))
                masses = [137.3270000,178.490,159.9940,159.9940,159.9940,1,1,1,1,1,1,1]
                val = val / np.sqrt(masses[ipert1 - 1] * masses[ipert2 - 1])
                dym[(idir1, ipert1, idir2, ipert2)] = val
            qdym.append(dym)
        self.qdym=qdym
        #Gfor d in self.data[5:]
        #a = []
        #for ipert2 in range(5):
        #    a.append(dym[(1, 1, 1, ipert2 + 1)])
        #print(sum(a))


    def split_pjte_data(self):
        datahead = []
        d2matr_lines = []
        d2nfr_lines = []
        d2fr_lines = []
        d2ew_lines = []
        ind2matr = False
        ind2nfr = False
        ind2fr = False

        datahead = self.data[0:5]
        lines = iter(self.data)
        for line in lines:
            if line.strip().startswith('- Total 2nd-order matrix'):
                ind2matr = True
                while ind2matr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2matr_lines.append(l)
                    else:
                        ind2matr = False

            if line.strip().startswith(
                    '- Frozen part of the 2nd-order matrix'):
                ind2fr = True
                while ind2fr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2fr_lines.append(l)
                    else:
                        ind2fr = False

            if line.strip().startswith(
                    '- Non-frozen part of the 2nd-order matrix'):
                ind2nfr = True
                while ind2nfr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2nfr_lines.append(l)
                    else:
                        ind2nfr = False
            if line.strip().startswith('- Ewald part of the 2nd-order matrix'):
                ind2nfr = True
                while ind2nfr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2ew_lines.append(l)
                    else:
                        ind2nfr = False

        return datahead, d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines

    def gen_pjte_ddbs(self):
        datahead, d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines = self.split_pjte_data(
        )
        prename = self.fname[0:-3]
        for ddbname, data in zip(
            ['TOT', 'NFR', 'FR', 'EW'],
            [d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines]):
            fname = '%s%s_DDB' % (prename, ddbname)
            with open(fname, 'w') as myfile:
                myfile.write(''.join(self.header))
                myfile.write(''.join(datahead))
                myfile.write(''.join(data))
                #print("DDBfile %s is generated" % fname)

