import os

from nose.tools import assert_equal
from sklearn.utils.testing import assert_array_equal, assert_almost_equal

import oddt
from oddt.property import xlogp2_atom_contrib

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_xlogp2():
    """Test XlogP results against original implementation"""
    mol = oddt.toolkit.readstring('smi', 'Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl')
    correct_xlogp = [0.082, -0.151,  0.337, 0.296, 0.663, 0.337, 0.337, -0.151,
                     0.435, -0.151, 0.337, 0.337, 0.296, 0.663, 0.337,  0.296,
                     0.663]

    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.addh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol = oddt.toolkit.readstring('smi', 'NC(N)c1ccc(C[C@@H](NC(=O)CNS(=O)(=O)c2ccc3ccccc3c2)C(=O)N2CCCCC2)cc1')
    correct_xlogp = [-0.534, -0.305, -0.534, 0.296, 0.337, 0.337, 0.296, -0.008,
                     -0.305, -0.096, -0.030, -0.399, -0.303, -0.112, -0.168,
                     -0.399, -0.399, 0.296, 0.337, 0.337, 0.296, 0.337, 0.337,
                     0.337, 0.337, 0.296, 0.337, -0.030, -0.399, 0.078, -0.137,
                     0.358, 0.358, 0.358, -0.137, 0.337, 0.337]

    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.addh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    predicted_xlogp = []
    for mol in oddt.toolkit.readfile('smi', os.path.join(test_data_dir, 'data/dude/xiap/actives_rdkit.smi')):
        predicted_xlogp.append(xlogp2_atom_contrib(mol).sum())

    correct_xlogp = [4.185, 4.185, 4.185, 4.185, 0.739, 0.739, 3.396, 2.080,
                     2.080, 4.119, 4.119, 2.945, 2.945, 0.701, 2.050, 1.399,
                     -0.037, 0.432, 2.210, 2.989, 2.871, 2.871, 2.871, 2.871,
                     4.927, 4.927, 2.871, 2.871, 2.871, 2.871, 2.425, 4.863,
                     4.863, 0.791, 0.791, 2.828, 4.426, 4.426, 1.087, 2.826,
                     3.305, 4.736, 1.097, 3.876, 2.061, 3.303, 2.759, 2.994,
                     3.051, 1.573, 1.370, 3.149, 0.713, 2.381, 2.697, 2.220,
                     4.220, 4.093, 2.468, 3.661, 4.796, 4.051, 2.677, 4.391,
                     1.728, 2.586, 5.157, 5.157, 5.157, 5.157, 3.394, 0.488,
                     -0.188, 1.012, 1.012, 0.442, 0.442, 0.442, 0.442, 1.476,
                     1.276, 3.208, 3.693, 0.876, 2.779, 2.779, 2.991, 2.740,
                     2.693, 3.962, 3.682, 2.385, 4.501, 4.740, 2.858, 3.214,
                     2.945, 3.166, 1.799, 0.360, 0.139, 2.401, 3.997, 3.997,
                     3.216, 3.572, 1.923, 1.118, 4.013, 3.767, 4.289, 1.097,
                     4.451, 4.859, 5.756, 5.756, 1.148, 3.303, 3.693, 1.789,
                     2.791, 3.751, 3.751, 3.751, 3.751, 3.166, 3.852, 4.068,
                     2.994]
    assert_almost_equal(correct_xlogp, predicted_xlogp, decimal=3)
