import numpy as np
import os
#from phonopy_tools import *


def cif_file(atoms, outpt, tol=1.0*10**-6):
    typat = atoms.get_chemical_symbols()
    xred = atoms.get_scaled_positions()
    a, b, c, alpha, beta, gama = atoms.get_cell_lengths_and_angles()
    fout = open('{}'.format(outpt), 'w')
    fout.write('!useKeyWords\n')
    fout.write('!title\n')
    fout.write('Input file to identify_space_group.\n')
    fout.write('!atomicPositionTolerance\n')
    fout.write('{}\n'.format(tol))
    fout.write('!latticeParameters\n')
    fout.write('{}   {}   {}   {}   {}   {}  \n'.format(
        a, b, c, alpha, beta, gama))
    fout.write('!atomCount\n')
    fout.write('{}\n'.format(len(typat)))
    fout.write('!atomType\n')
    for i in range(len(typat)):
        fout.write('{} '.format(typat[i]))

    fout.write('\n!atomPosition\n')
    for i in range(len(xred)):
        fout.write('  {}  {}   {}\n '.format(
            xred[i, 0], xred[i, 1], xred[i, 2]))
    fout.close()
    os.system('findsym {} > {}.sout'.format(outpt, outpt))

    # cif_out('{}.{}'.format(outpt,'sout'),'{}.{}'.format(outpt,'cif'))
    os.system('mv findsym.cif {}.cif '.format(outpt))
    os.system('rm {} {}.{} findsym.log'.format(outpt, outpt, 'sout'))


def cif_out(fin, fout):
    fin = open(fin, 'r')
    fot = open(fout, 'w')
    lne = 1
    while lne:
        lne = fin.readline()
        wrds = lne.split()
        if len(wrds) > 1:
            if wrds[1] == 'CIF':
                while lne:
                    fot.write(lne)
                    lne = fin.readline()
    fin.close()
    fot.close()
