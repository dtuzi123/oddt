from nose.tools import assert_equal
from sklearn.utils.testing import assert_array_equal

import oddt
from oddt.property import xlogp2_atom_contrib


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
