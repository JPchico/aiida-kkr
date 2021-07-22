# -*- coding: utf-8 -*-

from aiida.parsers.parser import Parser
from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida_kkr.calculations.voro import VoronoiCalculation
from aiida_kkr.data.kkr_potential import KKRPotentialData
from masci_tools.io.parsers.voroparser_functions import parse_voronoi_output
from masci_tools.io.common_functions import get_aBohr2Ang

__copyright__ = (u'Copyright (c), 2017, Forschungszentrum Jülich GmbH, ' 'IAS-1/PGI-1, Germany. All rights reserved.')
__license__ = 'MIT license, see LICENSE.txt file'
__version__ = '0.3.2'
__contributors__ = ('Jens Broeder', 'Philipp Rüßmann')


class VoronoiParser(Parser):
    """
    Parser class for parsing output of voronoi code..
    """

    def __init__(self, calc):
        """
        Initialize the instance of Voronoi_Parser
        """
        # these files should be present after success of voronoi
        self._default_files = {
            'outfile': VoronoiCalculation._OUTPUT_FILE_NAME,
            'atominfo': VoronoiCalculation._ATOMINFO,
            'radii': VoronoiCalculation._RADII
        }

        self._ParserVersion = __version__

        # reuse init of base class
        super(VoronoiParser, self).__init__(calc)

    # pylint: disable=protected-access
    def parse(self, debug=False, **kwargs):
        """
        Parse output data folder, store results in database.

        :param retrieved: a dictionary of retrieved nodes, where
          the key is the link name
        :returns: nothing if everything is fine or an exit code defined in the voronoi calculation class
        """

        success = False

        # Get retrieved folders
        try:
            out_folder = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        # Parse voronoi output, results that are stored in database are in out_dict
        if VoronoiCalculation._SHAPEFUN in list_of_files:
            with out_folder.open(VoronoiCalculation._SHAPEFUN, 'rb') as fhandle:
                shape_function = orm.SinglefileData(file=fhandle)
                self.out('shape_function', shape_function)
        else:
            shape_function = None

        # Parse the output file
        if VoronoiCalculation._OUTPUT_FILE_NAME in list_of_files:
            with out_folder.open(VoronoiCalculation._OUTPUT_FILE_NAME) as fhandle:
                outfile = fhandle.name
        else:
            self.logger.error(f"Output file '{VoronoiCalculation._OUTPUT_FILE_NAME}' not found")
            return self.exit_codes.ERROR_CRITICAL_MISSING_FILE
        # Parse the atominfo file
        if VoronoiCalculation._ATOMINFO in list_of_files:
            with out_folder.open(VoronoiCalculation._ATOMINFO) as fhandle:
                atominfo = fhandle.name
        else:
            self.logger.error(f"Output file '{VoronoiCalculation._ATOMINFO}' not found")
            return self.exit_codes.ERROR_CRITICAL_MISSING_FILE
        # Parse the radii file
        if VoronoiCalculation._RADII in list_of_files:
            with out_folder.open(VoronoiCalculation._RADII) as fhandle:
                radii = fhandle.name
        else:
            self.logger.error(f"Output file '{VoronoiCalculation._RADII}' not found")
            return self.exit_codes.ERROR_CRITICAL_MISSING_FILE
        # Parse the inputcard
        if VoronoiCalculation._INPUT_FILE_NAME in list_of_files:
            with out_folder.open(VoronoiCalculation._INPUT_FILE_NAME) as fhandle:
                inputfile = fhandle.name
                extra_dictionary = parse_inputcard(fhandle.get_content().splitlines())
                if shape_function is not None:
                    extra_dictionary['shape_function_uuid'] = shape_function.uuid
        else:
            self.logger.error(f"Output file '{VoronoiCalculation._INPUT_FILE_NAME}' not found")
            return self.exit_codes.ERROR_CRITICAL_MISSING_FILE
        # Parse the potential
        if VoronoiCalculation._OUT_POTENTIAL_voronoi in list_of_files:
            with out_folder.open(VoronoiCalculation._OUT_POTENTIAL_voronoi, 'rb') as fhandle:
                potfile = fhandle.name
                potential_file = KKRPotentialData(
                    file=fhandle,
                    extra_dictionary=extra_dictionary,
                )
                self.out('potential', potential_file)
        else:
            # cover case where potfile is overwritten from input to voronoi calculation
            if VoronoiCalculation._POTENTIAL_IN_OVERWRITE in list_of_files:
                with out_folder.open(VoronoiCalculation._POTENTIAL_IN_OVERWRITE, 'rb') as fhandle:
                    potfile = fhandle.name
                    potential_file = KKRPotentialData(
                        file=fhandle,
                        extra_dictionary=extra_dictionary,
                    )
                    self.out('potential', potential_file)
            else:
                self.logger.error(f"Output file '{VoronoiCalculation._POTENTIAL_IN_OVERWRITE}' not found")
                return self.exit_codes.ERROR_CRITICAL_MISSING_FILE

        # initialize out_dict and parse output files
        out_dict = {'parser_version': self._ParserVersion}
        out_dict['calculation_plugin_version'] = VoronoiCalculation._CALCULATION_PLUGIN_VERSION
        # TODO add job description, compound name, calculation title
        success, msg_list, out_dict = parse_voronoi_output(
            out_dict,
            outfile,
            potfile,
            atominfo,
            radii,
            inputfile,
            debug=debug,
        )
        # add file open errors to parser output of error messages
        out_dict['parser_errors'] = msg_list

        # create output node and link
        self.out('output_parameters', orm.Dict(dict=out_dict))

        if not success:
            return self.exit_codes.ERROR_VORONOI_PARSING_FAILED


def parse_inputcard(data):
    """Parse the inputcard for relevant information for the potential.

    To ensure that the attributes of the potential contain all the information
    needed so that it can be safely used for other calculations certain
    information that is present in the inputcard is necessary.
    """
    potential_types = {0: 'asa', 1: 'test', 2: 'full_potential'}
    soc_treatment = {
        0: 'non_relativistic',
        1: 'scalar_relativistic',
        2: 'full_relativistic',
    }
    inputcard_info = {
        'potential_type': 'full_potential',
        'spin_orbit_coupling': 'scalar_relativistic',
        'calculation_magnetism': 'spin_polarized',
        'cell': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'positions': [[0, 0, 0]],
    }
    alat = 1.0
    for index, line in enumerate(data):
        if 'ALATBASIS' in line:
            alat = get_aBohr2Ang() * float(line.replace('=', '').split()[-1])
        if 'BRAVAIS' in line:
            inputcard_info['cell'] = [[float(entry) for entry in data[index + 1].split()],
                                      [float(entry) for entry in data[index + 2].split()],
                                      [float(entry) for entry in data[index + 3].split()]]
        if 'NSPIN' in line:
            magnetic_status = int(line.replace('=', '').split()[-1])
            inputcard_info['calculation_magnetism'] = 'non_magnetic' if magnetic_status == 1 else 'spin_polarized'
        if '<RBASIS>' in line:
            not_found = True
            counter = 1
            inputcard_info['positions'] = []
            while not_found:
                _line = data[index + counter]
                if _line and all([is_number(entry) for entry in _line.split()]):
                    inputcard_info['positions'].append([float(entry) for entry in _line.split()])
                    counter += 1
                else:
                    not_found = False
        if 'KSHAPE' in line:
            potential_shape = int(line.replace('=', '').split()[-1])
            inputcard_info['potential_type'] = potential_types[potential_shape]
        if 'KVREL' in line:
            soc = int(line.replace('=', '').split()[-1])
            inputcard_info['spin_orbit_coupling'] = soc_treatment[soc]
    inputcard_info['cell'] = [[entry * alat for entry in vector] for vector in inputcard_info['cell']]
    return inputcard_info


def is_number(data):
    """Determine if a string is a floating number or not"""
    try:
        float(data)
        return True
    except ValueError:
        return False
