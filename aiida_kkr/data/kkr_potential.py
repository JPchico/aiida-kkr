"""
Define the data structure to store the potential from KKR
"""

import re
from aiida.orm import SinglefileData


class KKRPotentialData(SinglefileData):
    """Class to handle the potential file for JUKKR."""

    def set_file(self, file, filename=None, extra_dictionary=None, **kwargs):
        """Add a file to the node, parse it and set the attributes found.
        :param file: absolute path to the file or a filelike object
        :param filename: specify filename to use (defaults to name of provided file).
        """
        # pylint: disable=redefined-builtin
        super().set_file(file, filename, **kwargs)

        potential_header = parse_kkr_potential(self.get_content().splitlines())

        for key, value in potential_header.items():
            self.set_attribute(key, value)

        _keys = ['cell', 'position', 'calculation_magnetism', 'soc', 'potential_type']

        for _key in _keys:
            if extra_dictionary and _key in extra_dictionary.keys():
                self.set_attribute(_key, extra_dictionary[_key])
            else:
                self.set_attribute(_key, None)

    @property
    def atom_list(self):
        """Return the list of atoms.
        :return: a list of dictionaries containing the chemical symbol and atomic number of the atoms
        """
        return self.get_attribute('atom_list')

    @property
    def fermi_energy(self):
        """Return the fermi energy.
        :return: a float with the fermi energy in eV
        """
        return self.get_attribute('fermi_energy')

    @property
    def angular_momentum_cutoff(self):
        """Return the angular momentum cutoff (lmax).
        :return: an int with the angular momentum cutoff (lmax).
        """
        return self.get_attribute('angular_momentum_cutoff')

    @property
    def muffin_tin_radius(self):
        """Return the list of the muffin tin radius.
        :return: a list with the muffin tin radius.
        """
        return self.get_attribute('muffin_tin_radius')

    @property
    def wigner_seitz_radius(self):
        """Return the list of the Wigner-Seitz radius.
        :return: a list with the Wigner-Seitz radius.
        """
        return self.get_attribute('wigner_seitz_radius')

    @property
    def exchange_correlation(self):
        """Return the list of the exchange correlation potential (xc).
        :return: a list with the exchange correlation potential (xc).
        """
        return self.get_attribute('exchange_correlation')

    @property
    def muffin_tin_zero(self):
        """Return the list of the muffin tin zero (vbc).
        :return: a list with the muffin tin zero (vbc).
        """
        return self.get_attribute('muffin_tin_zero')

    @property
    def new_muffin_tin_radius(self):
        """Return the list of the new muffin tin zero (rmtnew).
        :return: a list with the new muffin tin zero (rmtnew).
        """
        return self.get_attribute('new_muffin_tin_radius')

    @property
    def number_non_spherical_points(self):
        """Return the list of the number of non-spherical points (irns).
        :return: a list with the number of non-spherical points (irns).
        """
        return self.get_attribute('number_non_spherical_points')


def parse_kkr_potential(lines):

    potential_header = {
        'exchange_correlation': [],
        'atom_list': [],
        'muffin_tin_radius': [],
        'alat_au': 0,
        'new_muffin_tin_radius': [],
        'wigner_seitz_radius': [],
        'fermi_energy': 0,
        'muffin_tin_zero': [],
        'irws': [],
        'radial_mesh_a': [],
        'radial_mesh_b': [],
        'angular_momentum_cutoff': 0,
        'number_non_spherical_points': [],
        'isave': []
    }
    pot_line = []
    end_header_line = []
    counter = 0
    for index, line in enumerate(lines):
        if 'POTENTIAL' in line:
            pot_line.append(index)
            counter = 1
        if len(line.split()) == 4 and counter == 1:
            end_header_line.append(index)
            counter = 0

    for index, entry in enumerate(pot_line):
        _lines = [re.split(': | #', line.strip()) for line in lines[entry:end_header_line[index] + 1]]
        potential_header['atom_list'].append({
            'symbol': re.findall(r'[A-Za-z]+|\d+', _lines[0][0].strip())[0],
            'atomic_number': float(_lines[2][-1].split()[0])
        })
        potential_header['exchange_correlation'].append(_lines[0][1].strip())
        potential_header['muffin_tin_radius'].append(float(_lines[1][-1].split()[0]))
        potential_header['alat_au'] = float(_lines[1][-1].split()[1])
        potential_header['new_muffin_tin_radius'].append(float(_lines[1][-1].split()[2]))
        potential_header['wigner_seitz_radius'].append(float(_lines[3][-1].split()[0]))
        potential_header['fermi_energy'] = float(_lines[3][-1].split()[1])
        potential_header['muffin_tin_zero'].append(float(_lines[3][-1].split()[2]))
        potential_header['irws'].append(int(_lines[4][-1].split()[0]))
        potential_header['radial_mesh_a'].append(float(_lines[5][-1].split()[0].replace('D', 'E')))
        potential_header['radial_mesh_b'].append(float(_lines[5][-1].split()[1].replace('D', 'E')))
        potential_header['angular_momentum_cutoff'] = int(_lines[-2][-1].split()[0]) + 1
        potential_header['number_non_spherical_points'].append(int(_lines[-1][-1].split()[1]))
        potential_header['isave'].append(int(_lines[-1][-1].split()[3]))

    return potential_header
