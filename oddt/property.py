"""
Module containg functions for prediction of molecular properies.
"""
from collections import OrderedDict
import numpy as np
import oddt


XLOGP_SMARTS_1 = OrderedDict([
    # sp3 carbon
    ('[!#7;!#8;!#1][CX4H3]', [0.528, 0.267]),  # 1-2
    ('[#7,#8][CX4H3]', [-0.032]),  # 3
    ('[!#7;!#8;!#1][CX4H2,CX2][!#7;!#8;!#1]', [0.358, -0.008, -0.185]),  # 4-6
    ('[#7,#8][CX4H2,CX2]', [-0.137, -0.303, -0.815]),  # 7-9
    ('[!#7;!#8;!#1][CX4H]([!#7;!#8;!#1])[!#7;!#8;!#1]', [0.127, -0.243, -0.499]),  # 10-12
    ('[#7,#8][CX4H,CX3]', [-0.205, -0.305, -0.709]),  # 13-15
    ('[!#7;!#8;!#1][CX4H0]([!#7;!#8;!#1])([!#7;!#8;!#1])[!#7;!#8;!#1]', [-0.006, -0.570, -0.317]),  # 16-18
    ('[#7,#8][CX4H0]', [-0.316, -0.723]),  # 19-20

    # sp2 carbon
    ('[*]=[CH2]', [0.420]),  # 21
    # Correction on one internal Pi bond (0 and 1 are duplicated)
    ('[*]=[CH1][!#7;!#8;!#1]', [0.466, 0.466, 0.136]),  # 22-23 +
    ('[*]=[CH1][#7,#8]', [0.001, 0.001, -0.310]),  # 24-25
    ('[*]=[CH0]([!#7;!#8;!#1])[!#7;!#8;!#1]', [0.050, 0.050, 0.013]),  # 26-27
    ('[O-]-[CH0](=[!#7;!#8;!#1])-[!#7;!#8;!#1]', [0.050, 0.050, 0.013]),  # 26-27
    ('[*]=[CH0]([!#7;!#8;!#1])[#7,#8;-0]', [-0.030, -0.030, -0.027]),  # 28-29
    ('[*]=[CH0]([#7,#8])[#7,#8]', [0.005, -0.315]),  # 30-31
    ('[0X1][CX3]=[OX1]', [0.050]),  # Custom - caboxyl group C sp2 is 0.05

    # aromatic carbon
    ('a:[cH]:a', [0.337]),  # 32
    ('a:[cH]:[n,#16]', [0.126]),  # 33
    ('c:[cH0]([!#7;!#8;!#1]):c', [0.296]),  # 34
    ('c:[cH0]([#7,#8]):c', [-0.151]),  # 35
    ('a:[cH0]([!#7;!#8;!#1]):[n,#16]', [0.174]),  # 36
    ('a:[cH0]([#7,#8]):[n,#16]', [0.366]),  # 37

    # sp carbon
    ('[!#7;!#8;!#1]#[CH1]', [0.209]),  # 38
    ('[*]#[CX2H0]', [0.330]),  # 39
    ('[*]=[CX2H0]=[*]', [2.073]),  # 40

    # sp3 nitrogen
    ('[!#7;!#8;!#1]-,=[NH2,H3+]', [-0.534, -0.329]),  # 41-42
    ('[#7,#8][NH2]', [-1.082]),  # 43
    ('[!#7;!#8;!#1]!@[NH1,H2+]!@[!#7;!#8;!#1]', [-0.112, 0.166]),  # 44-45
    ('[!#7;!#8;!#1]@[NH1,H2+]@[!#7;!#8;!#1]', [0.545]),  # 46
    ('[*]!@[NH]!@[#7,#8]', [0.324]),  # 47
    ('[*]@[NH]@[#7,#8]', [0.153]),  # 48
    ('[!#7;!#8;!#1]!:[NX3H0](!:[!#7;!#8;!#1])!:[!#7;!#8;!#1]', [0.159,  0.761]),  # 49-50 ## not in ring
    ('[!#7;!#8;!#1]!:[NX4H1+](!:[!#7;!#8;!#1])!:[!#7;!#8;!#1]', [0.159,  0.761]),  # 49-50 ## not in ring
    ('[!#7;!#8;!#1]~[nX3H0R](~[!#7;!#8;!#1])~[!#7;!#8;!#1]', [0.881]),  # 51 ## in ring
    ('[#7,#8][NX3H0R0]', [-0.239]),  # 52 ## not in ring
    ('[#7,#8][NX3H0R]', [-0.010]),  # 53 ## in ring

    # amide nitrogen
    ('[CX3]([NX3H2])(=[O,S])', [-0.646]),  # 54
    ('[!#7;!#8;!#1][NH1]C(=[O,S])', [-0.096]),  # 55
    ('[#7,#8][NH]C(=O)[#6]', [-0.044]),  # 56
    ('[!#7;!#8;!#1]N([!#7;!#8;!#1])C(=O)', [0.078]),  # 57
    ('[#7,#8]N([!#7;!#8;!#1])[CX3](=[OX1])', [-0.118]),  # 58

    # sp2 nitrogen
    ('C=N[!#7;!#8;!#1]', [0.007, 0.007, -0.275]),  # 59-60
    ('C=N[#7,#8]', [0.366, 0.366, 0.251]),  # 61-62
    ('N=N[!#7;!#8;!#1]', [0.536]),  # 63
    ('N=N[#7,#8]', [-0.597]),  # 64
    ('[*][NX2]=O', [0.427]),  # 65
    ('[*][NX2](=O)=O', [1.178]),  # 66

    # aromatic nitrogen
    ('a:n:a', [-0.493]),  # 67

    # sp nitrogen
    ('C#N', [-0.566]),  # 68

    # sp3 oxygen
    ('[!#7;!#8;!#1]-[OH1]', [-0.467, 0.082]),  # 69-70
    ('[#7,#8]-[OH1]', [-0.522]),  # 71
    ('[!#7;!#8;!#1][#8X2][!#7;!#8;!#1]', [0.084, 0.435]),  # 72-73
    ('[!#7;!#8;!#1][#8X2][#7,#8]', [0.105]),  # 74

    # sp2 oxygen
    ('[*]=O', [-0.399]),  # 75
    ('[*]-[OX1-]', [-0.399]),  # 75
    ('[CX3](-[OX1])=O', [-0.399]),  # 75

    # sp3 sulfur
    ('[*][SH]', [0.419]),  # 76
    ('[*][SX2H0][*]', [0.255]),  # 77

    # sp2 sulfur
    ('[*]=[SX1]', [-0.148]),  # 78
    ('a:[#16X2]:a', [-0.148]),  # 78

    # sulfoxide sulfur
    ('[*][SX3](=O)-[*]', [-1.375]),  # 79

    # sulfone sulfur
    ('[*][SX4](=O)(=O)-[*]', [-0.168]),  # 80

    # phosphorus
    ('O=P([*])([*])[*]', [-0.477]),  # 81
    ('S=P([*])([*])[*]', [1.253]),  # 82

    # fluorine
    ('[*]F', [0.375, 0.202]),  # 83-84

    # chlorine
    ('[*]Cl', [0.512, 0.663]),  # 85-86

    # bromine
    ('[*]Br', [0.850, 0.839]),  # 87-88

    # iodine
    ('[*]I', [1.050, 1.050]),  # 89-90
])

XLOGP_SMARTS_2 = [
    # Hydrophobic carbon
    {'smarts': '[C;!$([#6]~[#7,#8,#15,#16]);'
               '!$([#6]~[*]~[#7,#8,#15,#16]);'
               '!$([#6]~[*]~[*]~[#7,#8,#15,#16])]',
     'contrib_atoms': [0],
     'indicator': False,
     'coef': 0.211},
    # Internal H-bond
    {'smarts': '[O,N;!H0]-*@*=[O,N]',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': 0.429},
    {'smarts': '[O,N;!H0]-*@*-C=[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': False,
     'coef': 0.429},
    {'smarts': '[O,N;!H0]-C-*@*=[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': False,
     'coef': 0.429},
    # Halogen 1-3 pairs
    {'smarts': '[F,Cl,Br,I][*][F,Cl,Br,I]',
     'contrib_atoms': [0, 2],
     'indicator': False,
     'coef': 0.137},
    # Aromatic nitrogen 1-4 pair
    {'smarts': 'n:*:*:n',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': 0.485},
    # Ortho sp3 oxygen pair
    {'smarts': '[OX2H0R0]-!:aa-!:[OX2H0R0]',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': -0.268},
    # Para donor pair
    {'smarts': '[O,N;!H0]-!:aaaa-!:[O,N;!H0]',
     'contrib_atoms': [0, 5],
     'indicator': False,
     'coef': -0.423},
    # sp2 oxygen 1-5 pair
    {'smarts': '[CX3](=O)-!:[*]-!:[CX3]=O',
     'contrib_atoms': [1, 4],
     'indicator': False,
     'coef': 0.580},
    # Indicator for alpha-amino acid
    {'smarts': '[NX3,NX4+][CX4H][*][CX3](=[OX1])[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': True,
     'coef': -2.166},
    # Indicator for salicylic acid
    {'smarts': '[CX3](=[OX1])([O])-a:a-!:[OX1H]',
     'contrib_atoms': [1, 5],
     'indicator': True,
     'coef': 0.554},
    # Indicator for p-amino sulfonic acid
    {'smarts': '[SX4](=O)(=O)-c1ccc([NH2])cc1',
     'contrib_atoms': [0, 7],
     'indicator': True,
     'coef': -0.501},
]


def xlogp2_atom_contrib(mol):
    """
    Atoms contribution values taken from xlogp 2.0 publication:
    https://dx.doi.org/10.1023/A:1008763405023
    SMARTS patterns are in such orther that the described atom is always second.
    Values are sorted by increasing Pi bonds numbers
    """
    # count Pi bonds in n=2 environment
    pi_count = [sum(bond.isaromatic for bond in atom.bonds) +
                sum(any(bond.order > 1 or bond.isaromatic
                        for bond in neighbor.bonds)
                    for neighbor in atom.neighbors
                    if neighbor.atomicnum in [6, 7])
                for atom in mol]
    atom_contrib = np.zeros(len(pi_count))
    for smarts, contrib in XLOGP_SMARTS_1.items():
        matches = oddt.toolkit.Smarts(smarts).findall(mol)
        if matches:
            for match in matches:
                m = match[1]
                if oddt.toolkit.backend == 'ob':  # OB index is 1-based
                    m -= 1
                assert m >= 0
                atom_contrib[m] = contrib[pi_count[m]] if len(contrib) > pi_count[m] else contrib[-1]

    for correction in XLOGP_SMARTS_2:
        matches = oddt.toolkit.Smarts(correction['smarts']).findall(mol)
        if matches:
            for match in matches:
                for contrib_idx in correction['contrib_atoms']:
                    m = match[contrib_idx]
                    if oddt.toolkit.backend == 'ob':  # OB index is 1-based
                        m -= 1
                    assert m >= 0
                    atom_contrib[m] += correction['coef'] / float(len(correction['contrib_atoms']))
                    if correction['indicator']:
                        break

    return atom_contrib
