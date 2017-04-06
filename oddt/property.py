"""
Module containg functions for prediction of molecular properies.
"""
import numpy as np

xlogp2_smarts = {
    # sp3 carbon
    '[!#7;!#8][CX4H3]': [0.528, 0.267],  # 1-2
    '[#7,#8][CX4H3]': [-0.032],  # 3
    '[!#7;!#8][CX4H2][!#7;!#8]': [0.358, -0.008, -0.185],  # 4-6
    '[#7,#8][CX4H2]': [0.137, -0.303, -0.815],  # 7-9
    '[!#7;!#8][CX4H]([!#7;!#8])[!#7;!#8]': [0.127, -0.243, -0.499],  # 10-12
    '[#7,#8][CX4H]': [-0.205, -0.305, -0.709],  # 13-15
    '[!#7;!#8][CX4H0]([!#7;!#8])([!#7;!#8])[!#7;!#8]': [-0.006, -0.570, -0.317],  # 16-18
    '[#7,#8][CX4H0]': [-0.316, -0.723],  # 19-20

    # sp2 carbon
    '[*]=[CH2]': [0.420],  # 21
    '[*]=[CH1][!#7;!#8]': [0.466,  0.136],  # 22-23
    '[*]=[CH1][#7,#8]': [0.001, -0.310],  # 24-25
    '[*]=[CH0]([!#7;!#8])[!#7;!#8]': [0.050,  0.013],  # 26-27
    '[*]=[CH0]([!#7;!#8])[#7,#8]': [-0.030, -0.027],  # 28-29
    '[*]=[CH0]([#7,#8])[#7,#8]': [0.005, -0.315],  # 30-31

    # aromatic carbon
    'c:[cH]:c': [0.337],  # 32
    'a:[cH]:n': [0.126],  # 33
    'c:c([!#7;!#8]):c': [0.296],  # 34
    'c:c([#7,#8]):c': [-0.151],  # 35
    'a:c([!#7;!#8]):n': [0.174],  # 36
    'a:c([#7,#8]):n': [0.366],  # 37

    # sp carbon
    '[!#7;!#8]#[CH1]': [0.209],  # 38
    '[*]#[CX2H0]': [0.330],  # 39
    '[*]=[CX2H0]=[*]': [2.073],  # 40

    # sp3 nitrogen
    '[!#7;!#8][NH2]': [-0.534, -0.329],  # 41-42
    '[#7,#8][NH2]': [-1.082],  # 43
    '[!#7;!#8]!@[NH]!@[!#7;!#8]': [-0.112, 0.166],  # 44-45
    '[!#7;!#8]@[NH]@[!#7;!#8]': [0.545],  # 46
    '[*]!@[NH]!@[#7,#8]': [0.324],  # 47
    '[*]@[NH]@[#7,#8]': [0.153],  # 48
    '[!#7;!#8][NX3R0]([!#7;!#8])[!#7;!#8]': [0.159,  0.761],  # 49-50 ## not in ring
    '[!#7;!#8][NX3R]([!#7;!#8])[!#7;!#8]': [0.881],  # 51 ## in ring
    '[#7,#8][NX3H0R0]': [-0.239],  # 52 ## not in ring
    '[#7,#8][NX3H0R]': [-0.010],  # 53 ## in ring

    # amide nitrogen
    '[CX3]([NX3H2])(=[OX1])[#6]': [-0.646],  # 54
    '[!#7;!#8][NX3H1][CX3](=[OX1])[#6]': [-0.096],  # 55
    '[#7,#8][NX3H1][CX3](=[OX1])[#6]': [-0.044],  # 56
    '[!#7;!#8][NX3H0]([!#7;!#8])[CX3](=[OX1])[#6]': [0.078],  # 57
    '[#7,#8][NX3H0]([!#7;!#8])[CX3](=[OX1])[#6]': [-0.118],  # 58

    # sp2 nitrogen
    'C=N[!#7;!#8]': [0.007, -0.275],  # 59-60
    'C=N[#7,#8]': [0.366, 0.251],  # 61-62
    'N=N[!#7;!#8]': [0.536],  # 63
    'N=N[#7,#8]': [-0.597],  # 64
    '[*][NX2]=O': [0.427],  # 65
    '[*][NX2](=O)=O': [1.178],  # 66

    # aromatic nitrogen
    'a:n:a': [-0.493],  # 67

    # sp nitrogen
    'C#N': [-0.566],  # 68

    # sp3 oxygen
    '[!#7;!#8][OH]': [-0.467, 0.082],  # 69-70
    '[#7,#8][OH]': [-0.522],  # 71
    '[!#7;!#8]O[!#7;!#8]': [0.084, 0.435],  # 72-73
    '[!#7;!#8]O[#7,#8]': [0.105],  # 74

    # sp2 oxygen
    '[*]=O': [-0.399],  # 75

    # sp3 sulfur
    '[*][SH]': [0.419],  # 76
    '[*][SX2H0][*]': [0.255],  # 77

    # sp2 sulfur
    '[*]=S': [-0.148],  # 78

    # sulfoxide sulfur
    '[*][SX3](=O)-[*]': [-1.375],  # 79

    # sulfone sulfur
    '[*][SX4](=O)(=O)-[*]': [-0.168],  # 80

    # phosphorus
    'O=P([*])([*])[*]': [-0.477],  # 81
    'S=P([*])([*])[*]': [1.253],  # 82

    # fluorine
    '[*]F': [0.375, 0.202],  # 83-84

    # chlorine
    '[*]Cl': [0.512, 0.663],  # 85-86

    # bromine
    '[*]Br': [0.850, 0.839],  # 87-88

    # iodine
    '[*]I': [1.050, 1.050],  # 89-90
}


def xlogp_atom_contrib(mol):
    """
    Atoms contribution values taken from xlogp 2.0 publication:
    https://dx.doi.org/10.1023/A:1008763405023
    SMARTS patterns are in such orther that the described atom is always second.
    Values are sorted by increasing Pi bonds numbers
    """
    pi_count = [sum(bond.order > 1 or bond.isaromatic for bond in atom.bonds) for atom in mol]
    atom_contrib = np.zeros(len(pi_count))
    for smarts, contrib in xlogp2_smarts.items():
        matches = oddt.toolkit.Smarts(smarts).findall(mol)
        if matches:
            for match in matches:
                m = match[1]
                if oddt.toolkit.backend == 'ob':  # OB index is 1-based
                    m -= 1
                assert m >= 0
                atom_contrib[m] = contrib[pi_count[m]] if len(contrib) > pi_count[m] else contrib[-1]
    return atom_contrib
