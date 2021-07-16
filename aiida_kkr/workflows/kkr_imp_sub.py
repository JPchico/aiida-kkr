# -*- coding: utf-8 -*-
"""
In this module you find the sub workflow for the kkrimp self consistency cycle
and some helper methods to do so with AiiDA
"""
import os
import tarfile
import numpy as np
from masci_tools.io.kkr_params import kkrparams
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, calcfunction
from aiida_kkr.tools.common_workfunctions import (
    test_and_get_codenode,
    get_inputs_kkrimp,
    kick_out_corestates_wf,
)
from aiida_kkr.calculations.kkrimp import KkrimpCalculation
from aiida_kkr.tools.save_output_nodes import create_out_dict_node

__copyright__ = (u'Copyright (c), 2017, Forschungszentrum Jülich GmbH, ' 'IAS-1/PGI-1, Germany. All rights reserved.')
__license__ = 'MIT license, see LICENSE.txt file'
__version__ = '0.9.4'
__contributors__ = (u'Fabian Bertoldo', u'Philipp Ruessmann')

# TODO: work on return results function
# TODO: edit inspect_kkrimp function
# TODO: get rid of create_scf_result node and create output nodes differently
# TODO: check if calculation parameters from previous calculation have to be loaded (in validate input, compare to kkr workflow)
# TODO: maybe add decrease mixing factor option as in kkr_scf wc
# TODO: add option to check if the convergence is on track


class kkr_imp_sub_wc(WorkChain):
    """
    Workchain of a kkrimp self consistency calculation starting from the
    host-impurity potential of the system. (Not the entire kkr_imp workflow!)

    :param options: (Dict), Workchain specifications
    :param wf_parameters: (Dict), specifications for the calculation
    :param host_imp_startpot: (RemoteData), mandatory; input host-impurity potential
    :param kkrimp: (Code), mandatory; KKRimp code converging the host-imp-potential
    :param remote_data: (RemoteData), mandatory; remote folder of a previous
                           kkrflex calculation containing the flexfiles ...
    :param kkrimp_remote: (RemoteData), remote folder of a previous kkrimp calculation
    :param impurity_info: (Dict), Parameter node with information
                          about the impurity cluster

    :return workflow_info: (Dict), Information of workflow results
                                   like success, last result node, list with
                                   convergence behavior
    :return host_imp_pot: (SinglefileData), output potential of the sytem
    """

    _workflowversion = __version__
    _wf_label = 'kkr_imp_sub_wc'
    _wf_description = 'Workflow for a KKRimp self consistency calculation to converge a given host-impurity potential'

    _options_default = AttributeDict()
    # Queue name to submit jobs to
    _options_default.queue_name = ''
    # walltime after which the job gets killed (gets parsed to KKR)}
    _options_default.max_wallclock_seconds = 60 * 60
    # some additional scheduler commands
    _options_default.custom_scheduler_commands = ''
    # execute KKR with mpi or without
    _options_default.withmpi = True
    # resources to allowcate for the job
    _options_default.resources = AttributeDict()
    _options_default.resources.num_machines = 1

    _wf_default = AttributeDict()
    # Maximum number of kkr jobs/starts (defauld iterations per start)
    _wf_default.kkr_runmax = 5
    # Stop if charge density is converged below this value
    _wf_default.convergence_criterion = 1 * 10**-7
    # reduce mixing factor by this factor if calculation fails due to too large mixing
    _wf_default.mixreduce = 0.5
    # threshold after which agressive mixing is used
    _wf_default.threshold_aggressive_mixing = 1 * 10**-2
    # mixing factor of simple mixing
    _wf_default.strmix = 0.03
    # number of iterations done per KKR calculation
    _wf_default.nsteps = 50
    # type of aggressive mixing
    # (3: broyden's 1st, 4: broyden's 2nd, 5: generalized anderson)
    _wf_default.aggressive_mix = 5
    # mixing factor of aggressive mixing
    _wf_default.aggrmix = 0.05
    # number of potentials to 'remember' for Broyden's mixing
    _wf_default['broyden-number'] = 20
    # number of simple mixing step at the beginning of Broyden mixing
    _wf_default.nsimplemixfirst = 0
    # initialize and converge magnetic calculation
    _wf_default.mag_init = False
    # external magnetic field used in initialization step in Ry
    _wf_default.hfield = [0.02, 5]
    # position in unit cell where magnetic field is applied
    # [default (None) means apply to all]
    _wf_default.init_pos = None
    # specify if DOS should be calculated
    # (!KKRFLEXFILES with energy contour necessary as GF_remote_data!)
    _wf_default.dos_run = False
    # specify if DOS calculation should calculate l-resolved or l and m resolved output
    _wf_default.lmdos = True
    # specify if Jijs should be calculated (!changes behavior of the code!!!)
    _wf_default.jij_run = False
    # decide whether or not to clean up intermediate files
    # (THIS BREAKS CACHABILITY!)
    _wf_default.do_final_cleanup = True
    # Some parameter for direct solver
    # (if None, use the same as in host code, otherwise overwrite)
    _wf_default.accuracy_params = AttributeDict()
    # where to set change of logarithmic to linear radial mesh
    _wf_default.accuracy_params.RADIUS_LOGPANELS = None
    # number of panels in log mesh
    _wf_default.accuracy_params.NPAN_LOG = None
    # number of panels in linear mesh
    _wf_default.accuracy_params.NPAN_EQ = None
    # number of chebychev polynomials in each panel
    # (total number of points in radial mesh NCHEB*(NPAN_LOG+NPAN_EQ))
    _wf_default.accuracy_params.NCHEB = None

    @classmethod
    def get_wf_defaults(cls, silent=False):
        """Print and return _wf_defaults dictionary.

        Can be used to easily create set of wf_parameters.

        returns _wf_defaults
        """
        if not silent:
            print(f'Version of workflow: {cls._workflowversion}')
        return cls._wf_default

    @classmethod
    def define(cls, spec):
        """
        Defines the outline of the workflow
        """

        super(kkr_imp_sub_wc, cls).define(spec)

        # Define the inputs of the workflow
        spec.input(
            'kkrimp',
            valid_type=orm.Code,
            required=True,
        )
        spec.input(
            'host_imp_startpot',
            valid_type=orm.SinglefileData,
            required=False,
        )
        spec.input(
            'remote_data',
            valid_type=orm.RemoteData,
            required=False,
        )
        spec.input(
            'remote_data_Efshift',
            valid_type=orm.RemoteData,
            required=False,
        )
        spec.input(
            'kkrimp_remote',
            valid_type=orm.RemoteData,
            required=False,
        )
        spec.input(
            'impurity_info',
            valid_type=orm.Dict,
            required=False,
        )
        spec.input(
            'options',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict=cls._options_default),
        )
        spec.input(
            'wf_parameters',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict=cls._wf_default),
        )
        spec.input(
            'settings_LDAU',
            valid_type=orm.Dict,
            required=False,
            help='LDA+U settings. See KKRimpCalculation for details.',
        )

        # Here the structure of the workflow is defined
        spec.outline(
            cls.start,
            if_(cls.validate_input)(
                while_(cls.condition)(
                    cls.update_kkrimp_params,
                    # TODO: encapsulate this in restarting mechanism (should be a base class of workflows that start calculations)
                    # i.e. use base_restart_calc workchain as parent
                    cls.run_kkrimp,
                    cls.inspect_kkrimp
                ),
                cls.return_results
            ),
            cls.error_handler
        )

        # exit codes
        spec.exit_code(
            121,
            'ERROR_HOST_IMP_POT_GF',
            message='ERROR: Not both host-impurity potential and GF remote '
            'found in the inputs. Provide either both of them or a '
            'RemoteData from a previous kkrimp calculation.',
        )
        spec.exit_code(
            122,
            'ERROR_INVALID_INPUT_KKRIMP',
            message='ERROR: The code you provided for KKRimp does not '
            'use the plugin kkr.kkrimp',
        )
        spec.exit_code(
            123,
            'ERROR_INVALID_HOST_IMP_POT',
            message='ERROR: Unable to extract parent paremeter node of '
            'input remote folder',
        )
        # probably not necessary
        spec.exit_code(
            124,
            'ERROR_NO_CALC_PARAMS',
            message='ERROR: No calculation parameters provided',
        )

        spec.exit_code(
            125,
            'ERROR_SUB_FAILURE',
            message='ERROR: Last KKRcalc in SUBMISSIONFAILED state!\nstopping now',
        )
        spec.exit_code(
            126,
            'ERROR_MAX_STEPS_REACHED',
            message='ERROR: Maximal number of KKR restarts reached. Exiting now!',
        )
        spec.exit_code(
            127,
            'ERROR_SETTING_LAST_REMOTE',
            message='ERROR: Last_remote could not be set to a previous successful calculation',
        )
        spec.exit_code(
            128,
            'ERROR_MISSING_PARAMS',
            message='ERROR: There are still missing calculation parameters',
        )
        spec.exit_code(
            129,
            'ERROR_PARAMETER_UPDATE',
            message='ERROR: Parameters could not be updated',
        )
        spec.exit_code(
            130,
            'ERROR_LAST_CALC_NOT_FINISHED_OK',
            message='ERROR: Last calculation is not in finished state',
        )
        spec.exit_code(
            131,
            'ERROR_NO_CALC_FOUND_FOR_REMOTE_DATA',
            message='The input `remote_data` node has no valid calculation parent.',
        )
        spec.exit_code(
            132,
            'ERROR_REMOTE_DATA_CALC_UNSUCCESSFUL',
            message='The parent calculation of the input `remote_data` node was not successful.',
        )
        spec.exit_code(
            133,
            'ERROR_NO_OUTPUT_POT_FROM_LAST_CALC',
            message='ERROR: Last calculation does not have an output potential.',
        )

        # Define the outputs of the workflow
        spec.output(
            'workflow_info',
            valid_type=orm.Dict,
        )
        spec.output(
            'host_imp_pot',
            valid_type=orm.SinglefileData,
            required=False,
        )

    def start(self):
        """
        init context and some parameters
        """
        message = f'INFO: started KKR impurity convergence workflow version {self._workflowversion}'
        self.report(message)

        ####### init #######

        # internal para /control para
        self.ctx.loop_count = 0
        self.ctx.last_mixing_scheme = 0
        self.ctx.calcs = []
        self.ctx.exit_code = None
        # flags used internally to check whether the individual steps were successful
        self.ctx.kkr_converged = False
        self.ctx.kkr_step_success = False
        self.ctx.kkr_higher_accuracy = False
        # links to previous calculations
        self.ctx.last_calc = None
        self.ctx.last_params = None
        self.ctx.last_remote = None
        # link to previous host impurity potential
        self.ctx.last_pot = None
        # intermediate single file data objects that contain potential files which can be clean up at the end
        self.ctx.sfd_pot_to_clean = []
        # convergence info about rms etc. (used to determine convergence behavior)
        self.ctx.last_rms_all = []
        self.ctx.rms_all_steps = []
        self.ctx.last_neutr_all = []
        self.ctx.neutr_all_steps = []
        # LDA+U settings, either from input port or extracted from kkrimp_remote
        self.ctx.settings_LDAU = None

        # input para
        wf_dict = self.inputs.wf_parameters.get_dict()
        options_dict = self.inputs.options.get_dict()

        if options_dict == {}:
            options_dict = self._options_default
            message = 'INFO: using default options'
            self.report(message)

        if wf_dict == {}:
            wf_dict = self._wf_default
            message = 'INFO: using default wf parameter'
            self.report(message)

        # cleanup intermediate calculations (WARNING: THIS PREVENTS USING CACHING!!!)
        self.ctx.do_final_cleanup = wf_dict.get(
            'do_final_cleanup',
            self._wf_default['do_final_cleanup'],
        )

        # set option parameters from input, or defaults
        self.ctx.withmpi = options_dict.get(
            'withmpi',
            self._options_default['withmpi'],
        )
        self.ctx.resources = options_dict.get(
            'resources',
            self._options_default['resources'],
        )
        self.ctx.max_wallclock_seconds = options_dict.get(
            'max_wallclock_seconds',
            self._options_default['max_wallclock_seconds'],
        )
        self.ctx.queue = options_dict.get(
            'queue_name',
            self._options_default['queue_name'],
        )
        self.ctx.custom_scheduler_commands = options_dict.get(
            'custom_scheduler_commands',
            self._options_default['custom_scheduler_commands'],
        )

        # set workflow parameters from input, or defaults
        self.ctx.max_number_runs = wf_dict.get(
            'kkr_runmax',
            self._wf_default['kkr_runmax'],
        )
        self.ctx.description_wf = self.inputs.get(
            'description', 'Workflow for '
            'a KKR impurity calculation'
            'starting from a host-impurity'
            'potential'
        )
        self.ctx.label_wf = self.inputs.get(
            'label',
            'kkr_imp_sub_wc',
        )
        self.ctx.strmix = wf_dict.get(
            'strmix',
            self._wf_default['strmix'],
        )
        self.ctx.convergence_criterion = wf_dict.get(
            'convergence_criterion',
            self._wf_default['convergence_criterion'],
        )
        self.ctx.mixreduce = wf_dict.get(
            'mixreduce',
            self._wf_default['mixreduce'],
        )
        self.ctx.threshold_aggressive_mixing = wf_dict.get(
            'threshold_aggressive_mixing',
            self._wf_default['threshold_aggressive_mixing'],
        )
        self.ctx.type_aggressive_mixing = wf_dict.get(
            'aggressive_mix',
            self._wf_default['aggressive_mix'],
        )
        self.ctx.aggrmix = wf_dict.get(
            'aggrmix',
            self._wf_default['aggrmix'],
        )
        self.ctx.nsteps = wf_dict.get(
            'nsteps',
            self._wf_default['nsteps'],
        )
        self.ctx.broyden_num = wf_dict.get(
            'broyden-number',
            self._wf_default['broyden-number'],
        )
        self.ctx.nsimplemixfirst = wf_dict.get(
            'nsimplemixfirst',
            self._wf_default['nsimplemixfirst'],
        )

        # initial magnetization
        self.ctx.mag_init = wf_dict.get(
            'mag_init',
            self._wf_default['mag_init'],
        )
        self.ctx.hfield = wf_dict.get(
            'hfield',
            self._wf_default['hfield'],
        )
        self.ctx.xinit = wf_dict.get(
            'init_pos',
            self._wf_default['init_pos'],
        )
        self.ctx.mag_init_step_success = False

        # accuracy parameter
        self.ctx.mesh_params = wf_dict.get(
            'accuracy_params',
            self._wf_default['accuracy_params'],
        )

        # DOS
        self.ctx.dos_run = wf_dict.get(
            'dos_run',
            self._wf_default['dos_run'],
        )
        self.ctx.lmdos = wf_dict.get(
            'lmdos',
            self._wf_default['lmdos'],
        )
        # Jij
        self.ctx.jij_run = wf_dict.get(
            'jij_run',
            self._wf_default['jij_run'],
        )

        self.report(
            f'INFO: use the following parameter:\n'
            f'\nGeneral settings\n'
            f'use mpi: {self.ctx.withmpi}\n'
            f'max number of KKR runs: {self.ctx.max_number_runs}\n'
            f'Resources: {self.ctx.resources}\n'
            f'Walltime (s): {self.ctx.max_wallclock_seconds}\n'
            f'queue name: {self.ctx.queue}\n'
            f'scheduler command: {self.ctx.custom_scheduler_commands}\n'
            f'description: {self.ctx.description_wf}\n'
            f'label: {self.ctx.label_wf}\n'
            f'\nMixing parameter\n'
            f'Straight mixing factor: {self.ctx.strmix}\n'
            f'Nsteps scf cycle: {self.ctx.nsteps}\n'
            f'threshold_aggressive_mixing: {self.ctx.threshold_aggressive_mixing}\n'
            f'Aggressive mixing technique: {self.ctx.type_aggressive_mixing}\n'
            f'Aggressive mixing factor: {self.ctx.aggrmix}\n'
            f'Mixing decrease factor if convergence fails: {self.ctx.mixreduce}\n'
            f'Convergence criterion: {self.ctx.convergence_criterion}\n'
            f'\nAdditional parameter\n'
            f'init magnetism in first step: {self.ctx.mag_init}\n'
            f'init magnetism, hfield: {self.ctx.hfield}\n'
            f'init magnetism, init_pos: {self.ctx.xinit}\n'
        )

        # return para/vars
        self.ctx.successful = False
        self.ctx.rms = []
        self.ctx.neutr = []
        self.ctx.warnings = []
        self.ctx.formula = ''

        # for results table each list gets one entry per iteration that has been performed
        self.ctx.KKR_steps_stats = {}
        # later contains these keys:
        # 'success', 'isteps', 'imix', 'mixfac', 'qbound', 'high_sett', 'first_rms', 'last_rms'
        # 'first_neutr', 'last_neutr', 'pk', 'uuid'

    def validate_input(self):
        """
        validate input and catch possible errors from the input
        """

        inputs = self.inputs
        inputs_ok = True

        if not 'kkrimp_remote' in inputs and not ('host_imp_startpot' in inputs and 'remote_data' in inputs):
            inputs_ok = False
            self.ctx.exit_code = self.exit_codes.ERROR_HOST_IMP_POT_GF  # pylint: disable=maybe-no-member

        if 'kkr' in inputs:
            try:
                test_and_get_codenode(inputs.kkr, 'kkr.kkrimp', use_exceptions=True)
            except ValueError:
                inputs_ok = False
                self.ctx.exit_code = self.exit_codes.ERROR_INVALID_INPUT_KKRIMP  # pylint: disable=maybe-no-member

        # check if LDA+U settings should be set from input port
        if 'settings_LDAU' in inputs:
            self.ctx.settings_LDAU = inputs.settings_LDAU

        if 'kkrimp_remote' in inputs:
            self.ctx.start_from_imp_remote = True
            kkrimp_remote = inputs.kkrimp_remote
            self.ctx.last_remote = kkrimp_remote
            # check if LDA+U settings should be set kkrimp parent calculation
            if 'settings_LDAU' not in inputs:
                # check if kkrimp parent calculation has LDA+U input
                parent_kkrimp_calc = kkrimp_remote.get_incoming(node_class=orm.CalcJobNode).first().node
                if 'settings_LDAU' in parent_kkrimp_calc.inputs:
                    self.ctx.settings_LDAU = parent_kkrimp_calc.inputs.settings_LDAU

        # check if input remote_data node is fine
        if 'remote_data' in inputs:
            if len(inputs.remote_data.get_incoming(link_label_filter='remote_folder').all()) < 1:
                self.ctx.exit_code = self.exit_codes.ERROR_NO_CALC_FOUND_FOR_REMOTE_DATA  # pylint: disable=maybe-no-member
            else:
                if not inputs.remote_data.get_incoming(link_label_filter='remote_folder').first().node.is_finished_ok:
                    self.ctx.exit_code = self.exit_codes.ERROR_REMOTE_DATA_CALC_UNSUCCESSFUL  # pylint: disable=maybe-no-member

        # set starting potential
        if 'host_imp_startpot' in inputs:
            self.ctx.last_pot = inputs.host_imp_startpot

        # TBD!!!
        if 'wf_parameters' in inputs:
            self.ctx.last_params = inputs.wf_parameters
        else:
            inputs_ok = False
            self.ctx.exit_code = self.exit_codes.ERROR_NO_CALC_PARAMS  # pylint: disable=maybe-no-member

        message = f'INFO: validated input successfully: {inputs_ok}'
        self.report(message)
        if not inputs_ok:
            message = f'Exit code: {self.exit_codes.ERROR_NO_CALC_PARAMS}'  # pylint: disable=maybe-no-member
            self.report(message)

        return inputs_ok

    def condition(self):
        """
        check convergence condition
        """

        do_kkr_step = True
        stopreason = ''

        #increment KKR runs loop counter
        self.ctx.loop_count += 1

        # check if previous calculation reached convergence criterion
        if self.ctx.kkr_converged:
            if not self.ctx.kkr_higher_accuracy:
                do_kkr_step = do_kkr_step & True
            else:
                stopreason = 'KKR converged'
                self.ctx.successful = True
                do_kkr_step = False
        else:
            do_kkr_step = do_kkr_step & True

        # check if previous calculation was successful
        if self.ctx.loop_count > 1 and not self.ctx.last_calc.is_finished_ok:
            message = 'ERROR: last calc not finished_ok'
            self.report(message)
            return self.exit_codes.ERROR_SUB_FAILURE  # pylint: disable=maybe-no-member

        # next check only needed if another iteration should be done after validating convergence etc. (previous checks)
        if do_kkr_step:
            # check if maximal number of iterations has been reached
            if self.ctx.loop_count <= self.ctx.max_number_runs:
                do_kkr_step = do_kkr_step & True
            else:
                do_kkr_step = False

        message = f'INFO: done checking condition for kkr step (result={do_kkr_step})'
        self.report(message)

        if not do_kkr_step:
            message = f'INFO: Stopreason={stopreason}'
            self.report(message)

        return do_kkr_step

    def update_kkrimp_params(self):
        """
        update set of KKR parameters (check for reduced mixing, change of
        mixing strategy, change of accuracy setting)
        """

        decrease_mixing_fac = False
        switch_agressive_mixing = False
        switch_higher_accuracy = False
        initial_settings = False

        # only do something other than simple mixing after first kkr run
        if self.ctx.loop_count != 1:
            # first determine if previous step was successful (otherwise try to find some rms value and decrease mixing to try again)
            if not self.ctx.kkr_step_success:
                decrease_mixing_fac = True
                message = 'INFO: last KKR calculation failed. Trying decreasing mixfac'
                self.report(message)

            convergence_on_track = self.convergence_on_track()

            # check if calculation was on its way to converge
            if not convergence_on_track:
                decrease_mixing_fac = True
                message = 'INFO: Last KKR did not converge. Trying decreasing mixfac'
                self.report(message)
                # reset last_remote to last successful calculation
                last_calcs_list = list(range(len(self.ctx.calcs)))  # needs to be list to support slicing
                if len(last_calcs_list) > 1:
                    last_calcs_list = np.array(last_calcs_list)[::-1
                                                                ]  # make sure to go from latest calculation backwards
                for icalc in last_calcs_list:
                    message = f'INFO: last calc success? {icalc} {self.ctx.KKR_steps_stats["success"][icalc]}'
                    self.report(message)
                    if self.ctx.KKR_steps_stats['success'][icalc]:
                        if self.ctx.KKR_steps_stats['last_rms'][icalc] < self.ctx.KKR_steps_stats['first_rms'][icalc]:
                            self.ctx.last_remote = self.ctx.calcs[icalc].outputs.remote_folder
                            break  # exit loop if last_remote was found successfully
                        else:
                            self.ctx.last_remote = None
                    else:
                        self.ctx.last_remote = None
                # now cover case when last_remote needs to be set to initial remote folder (from input)
                if self.ctx.last_remote is None:
                    if 'kkrimp_remote' in self.inputs:
                        messager = 'INFO: no successful and converging calculation to take RemoteData from. Reuse RemoteData from input instead.'
                        self.report(message)
                        self.ctx.last_remote = self.inputs.kkrimp_remote
                    elif 'impurity_info' in self.inputs or 'remote_data' in self.inputs:
                        self.ctx.last_remote = None
                # check if last_remote has finally been set and abort if this is not the case
                if self.ctx.last_remote is None:
                    messager = 'ERROR: last remote not found'
                    self.report(message)
                    return self.exit_codes.ERROR_SETTING_LAST_REMOTE  # pylint: disable=maybe-no-member

            # check if mixing strategy should be changed
            last_mixing_scheme = self.ctx.last_params.get_dict()['IMIX']
            if last_mixing_scheme is None:
                last_mixing_scheme = 0

            if convergence_on_track:
                last_rms = self.ctx.last_rms_all[-1]
                if last_rms < self.ctx.threshold_aggressive_mixing and last_mixing_scheme == 0:
                    switch_agressive_mixing = True
                    message = 'INFO: rms low enough, switch to agressive mixing'
                    self.report(message)

                # check if switch to higher accuracy should be done
                # or last_rms < self.ctx.threshold_switch_high_accuracy:
                if not self.ctx.kkr_higher_accuracy and self.ctx.kkr_converged:
                    switch_higher_accuracy = True
#                        self.report("INFO: rms low enough, switch to higher accuracy settings")
        else:
            initial_settings = True
            self.ctx.kkr_step_success = True

        if self.ctx.loop_count > 1:
            last_rms = self.ctx.last_rms_all[-1]

        # extract values from host calculation
        host_GF_calc = self.inputs.remote_data.get_incoming(node_class=orm.CalcJobNode).first().node
        host_GF_outparams = host_GF_calc.outputs.output_parameters.get_dict()
        host_GF_inparams = host_GF_calc.inputs.parameters.get_dict()
        nspin = host_GF_outparams.get('nspin')
        non_spherical = host_GF_inparams.get('INS')
        if non_spherical is None:
            non_spherical = kkrparams.get_KKRcalc_parameter_defaults()[0].get('INS')
        self.ctx.spinorbit = host_GF_outparams.get('use_newsosol')

        # if needed update parameters
        if decrease_mixing_fac or switch_agressive_mixing or switch_higher_accuracy or initial_settings or self.ctx.mag_init:
            if initial_settings:
                label = 'initial KKR scf parameters'
                description = 'initial parameter set for scf calculation'
            else:
                label = ''
                description = ''

            # step 1: extract info from last input parameters and check consistency
            para_check = kkrparams(params_type='kkrimp')
            para_check.get_all_mandatory()
            message = 'INFO: get kkrimp keywords'
            self.report(message)

            # init new_params dict where updated params are collected
            new_params = {}

            # step 1.2: check if all mandatory keys are there and add defaults if missing
            missing_list = para_check.get_missing_keys(use_aiida=True)
            if missing_list != []:
                kkrdefaults = kkrparams.get_KKRcalc_parameter_defaults()[0]
                kkrdefaults_updated = []
                for key_default, val_default in list(kkrdefaults.items()):
                    if key_default in missing_list:
                        new_params[key_default] = kkrdefaults.get(key_default)
                        kkrdefaults_updated.append(key_default)
                if len(kkrdefaults_updated) > 0:
                    message = 'ERROR: no default param found'
                    self.report(message)
                    return self.exit_codes.ERROR_MISSING_PARAMS  # pylint: disable=maybe-no-member
                else:
                    message = f'updated KKR parameter node with default values: {kkrdefaults_updated}'
                    self.report(message)

            # step 2: change parameter (contained in new_params dictionary)
            last_mixing_scheme = para_check.get_value('IMIX')
            if last_mixing_scheme is None:
                last_mixing_scheme = 0

            strmixfac = self.ctx.strmix
            aggrmixfac = self.ctx.aggrmix
            nsteps = self.ctx.nsteps

            # TODO: maybe add decrease mixing factor option as in kkr_scf wc
            # step 2.1 fill new_params dict with values to be updated
            if decrease_mixing_fac:
                if last_mixing_scheme == 0:
                    self.report(f'(strmixfax, mixreduce)= ({strmixfac}, {self.ctx.mixreduce})')
                    self.report(f'type(strmixfax, mixreduce)= {type(strmixfac)} {type(self.ctx.mixreduce)}')
                    strmixfac = strmixfac * self.ctx.mixreduce
                    self.ctx.strmix = strmixfac
                    label += f'decreased_mix_fac_str (step {self.ctx.loop_count})'
                    description += f'decreased STRMIX factor by {self.ctx.mixreduce}'
                else:
                    self.report(f'(aggrmixfax, mixreduce)= ({aggrmixfac}, {self.ctx.mixreduce})')
                    self.report(f'type(aggrmixfax, mixreduce)= {type(aggrmixfac)} {type(self.ctx.mixreduce)}')
                    aggrmixfac = aggrmixfac * self.ctx.mixreduce
                    self.ctx.aggrmix = aggrmixfac
                    label += 'decreased_mix_fac_bry'
                    description += f'decreased AGGRMIX factor by {self.ctx.mixreduce}'

            if switch_agressive_mixing:
                last_mixing_scheme = self.ctx.type_aggressive_mixing
                label += ' switched_to_agressive_mixing'
                description += f' switched to agressive mixing scheme (IMIX={last_mixing_scheme})'

            # add number of scf steps, spin
            new_params['SCFSTEPS'] = nsteps
            new_params['NSPIN'] = nspin
            new_params['INS'] = non_spherical

            # add ldos runoption if dos_run = True
            if self.ctx.dos_run:
                if self.ctx.lmdos:
                    runflags = new_params.get('RUNFLAG', []) + ['lmdos']
                else:
                    runflags = new_params.get('RUNFLAG', []) + ['ldos']
                new_params['RUNFLAG'] = runflags
                new_params['SCFSTEPS'] = 1

            # turn on Jij calculation if jij_run == True
            if self.ctx.jij_run:
                new_params['CALCJIJMAT'] = 1

            # add newsosol
            if self.ctx.spinorbit:
                testflags = new_params.get('TESTFLAG', []) + ['tmatnew']
                new_params['TESTFLAG'] = testflags
                new_params['SPINORBIT'] = 1
                new_params['NCOLL'] = 1
                if self.ctx.mesh_params.get('RADIUS_LOGPANELS', None) is not None:
                    new_params['RADIUS_LOGPANELS'] = self.ctx.mesh_params['RADIUS_LOGPANELS']
                if self.ctx.mesh_params.get('NCHEB', None) is not None:
                    new_params['NCHEB'] = self.ctx.mesh_params['NCHEB']
                if self.ctx.mesh_params.get('NPAN_LOG', None) is not None:
                    new_params['NPAN_LOG'] = self.ctx.mesh_params['NPAN_LOG']
                if self.ctx.mesh_params.get('NPAN_EQ', None) is not None:
                    new_params['NPAN_EQ'] = self.ctx.mesh_params['NPAN_EQ']
                new_params['CALCORBITALMOMENT'] = 1
            else:
                new_params['SPINORBIT'] = 0
                new_params['NCOLL'] = 0
                new_params['CALCORBITALMOMENT'] = 0
                new_params['TESTFLAG'] = []

            # set mixing schemes and factors
            if last_mixing_scheme > 2:
                new_params['ITDBRY'] = self.ctx.broyden_num
                new_params['IMIX'] = last_mixing_scheme
                new_params['MIXFAC'] = aggrmixfac
                new_params['NSIMPLEMIXFIRST'] = self.ctx.nsimplemixfirst
            elif last_mixing_scheme == 0:
                new_params['IMIX'] = last_mixing_scheme
                new_params['MIXFAC'] = strmixfac

            # add mixing scheme to context
            self.ctx.last_mixing_scheme = last_mixing_scheme

            if switch_higher_accuracy:
                self.ctx.kkr_higher_accuracy = True


#                convergence_settings = self.ctx.convergence_setting_fine
#                label += ' use_higher_accuracy'
#                description += ' using higher accuracy settings goven in convergence_setting_fine'
#            else:
#                convergence_settings = self.ctx.convergence_setting_coarse

# add convergence settings
            if self.ctx.loop_count == 1 or self.ctx.last_mixing_scheme == 0:
                new_params['QBOUND'] = self.ctx.threshold_aggressive_mixing
            else:
                new_params['QBOUND'] = self.ctx.convergence_criterion

            # initial magnetization
            if initial_settings and self.ctx.mag_init:
                if self.ctx.hfield[0] <= 0.0 or self.ctx.hfield[1] == 0:
                    self.report(
                        f'\nWARNING: magnetization initialization chosen but hfield is zero.'
                        f' Automatically change back to default value (hfield={self._wf_default["hfield"]})\n'
                    )
                    self.ctx.hfield = self._wf_default['hfield']
                new_params['HFIELD'] = self.ctx.hfield
            elif self.ctx.mag_init and self.ctx.mag_init_step_success:  # turn off initialization after first (successful) iteration
                new_params['HFIELD'] = [0.0, 0]
            elif not self.ctx.mag_init:
                self.report("INFO: mag_init is False. Overwrite 'HFIELD' to '0.0' and 'LINIPOL' to 'False'.")
                # reset mag init to avoid resinitializing
                new_params['HFIELD'] = [0.0, 0]

            # set nspin to 2 if mag_init is used
            if self.ctx.mag_init:
                nspin_in = nspin
                if nspin_in is None:
                    nspin_in = 1
                if nspin_in < 2:
                    self.report('WARNING: found NSPIN=1 but for maginit needs NPIN=2. Overwrite this automatically')
                    new_params['NSPIN'] = 2
            message = f'new_params: {new_params}'
            self.report(message)

            # step 2.2 update values
            try:
                for key, val in new_params.items():
                    para_check.set_value(key, val, silent=True)
            except:
                message = 'ERROR: failed to set some parameters'
                self.report(message)
                return self.exit_codes.ERROR_PARAMETER_UPDATE  # pylint: disable=maybe-no-member

            # step 3:
            message = f'INFO: update parameters to: {para_check.get_set_values()}'
            self.report(message)

            #test
            self.ctx.last_params = orm.Dict(dict={})

            updatenode = orm.Dict(dict=para_check.get_dict())
            updatenode.label = label
            updatenode.description = description

            paranode_new = updatenode  #update_params_wf(self.ctx.last_params, updatenode)
            self.ctx.last_params = paranode_new
        else:
            message = 'INFO: reuse old settings'
            self.report(message)

        message = 'INFO: done updating kkr param step'
        self.report(message)
        return None

    def run_kkrimp(self):
        """
        submit a KKR impurity calculation
        """
        message = f'INFO: setting up kkrimp calculation step {self.ctx.loop_count}'
        self.report(message)

        label = f'KKRimp calculation step {self.ctx.loop_count} (IMIX={self.ctx.last_mixing_scheme})'
        description = f'KKRimp calculation of step {self.ctx.loop_count}, using mixing scheme {self.ctx.last_mixing_scheme}'
        code = self.inputs.kkrimp
        params = self.ctx.last_params
        host_GF = self.inputs.remote_data
        imp_pot = self.ctx.last_pot
        last_remote = self.ctx.last_remote
        if 'remote_data_Efshift' in self.inputs:
            host_GF_Efshift = self.inputs.remote_data_Efshift
        else:
            host_GF_Efshift = None

        options = {
            'max_wallclock_seconds': self.ctx.max_wallclock_seconds,
            'resources': self.ctx.resources,
            'queue_name': self.ctx.queue
        }
        if self.ctx.custom_scheduler_commands:
            options['custom_scheduler_commands'] = self.ctx.custom_scheduler_commands

        if last_remote is None:
            # make sure no core states are in energy contour
            # extract emin from output of GF host calculation
            GF_out_params = host_GF.get_incoming(link_label_filter='remote_folder'
                                                 ).first().node.outputs.output_parameters
            emin = GF_out_params.get_dict().get('energy_contour_group').get('emin')
            # then use this value to get rid of all core states that are
            # lower than emin (return the same input potential if no states have been removed
            imp_pot = kick_out_corestates_wf(imp_pot, orm.Float(emin))
            self.ctx.sfd_pot_to_clean.append(imp_pot)
            if 'impurity_info' in self.inputs:
                message = 'INFO: using impurity_info node as input for kkrimp calculation'
                self.report(message)
                imp_info = self.inputs.impurity_info
                label = f'KKRimp calculation step {self.ctx.loop_count}'\
                    + f' (IMIX={self.ctx.last_mixing_scheme}, Zimp: {imp_info.get_dict().get("Zimp")})'
                description = f'KKRimp calculation of step {self.ctx.loop_count},'\
                    + f' using mixing scheme {self.ctx.last_mixing_scheme}'
                inputs = get_inputs_kkrimp(
                    code,
                    options,
                    label,
                    description,
                    params,
                    not self.ctx.withmpi,
                    imp_info=imp_info,
                    host_GF=host_GF,
                    imp_pot=imp_pot,
                    host_GF_Efshift=host_GF_Efshift
                )
            else:
                message = 'INFO: getting impurity_info node from previous GF calculation'
                self.report(message)
                label = f'KKRimp calculation step {self.ctx.loop_count}'\
                    + f' (IMIX={self.ctx.last_mixing_scheme}, GF_remote: {host_GF.pk})'
                description = f'KKRimp calculation of step {self.ctx.loop_count},'\
                    + f' using mixing scheme {self.ctx.last_mixing_scheme}'
                inputs = get_inputs_kkrimp(
                    code,
                    options,
                    label,
                    description,
                    params,
                    not self.ctx.withmpi,
                    host_GF=host_GF,
                    imp_pot=imp_pot,
                    host_GF_Efshift=host_GF_Efshift
                )
        elif last_remote is not None:
            # fix to get Zimp properly
            if 'impurity_info' in self.inputs:
                message = 'INFO: using RemoteData from previous kkrimp calculation and impurity_info node as input'
                self.report(message)
                imp_info = self.inputs.impurity_info
                label = f'KKRimp calculation step {self.ctx.loop_count}'\
                    + f' (IMIX={self.ctx.last_mixing_scheme}, Zimp: {imp_info.get_dict().get("Zimp")})'
                description = f'KKRimp calculation of step {self.ctx.loop_count},'\
                    + f' using mixing scheme {self.ctx.last_mixing_scheme}'
                inputs = get_inputs_kkrimp(
                    code,
                    options,
                    label,
                    description,
                    params,
                    not self.ctx.withmpi,
                    imp_info=imp_info,
                    host_GF=host_GF,
                    kkrimp_remote=last_remote,
                    host_GF_Efshift=host_GF_Efshift
                )
            else:
                message = 'INFO: using RemoteData from previous kkrimp calculation as input'
                self.report(message)
                label = f'KKRimp calculation step {self.ctx.loop_count}'\
                    + f' (IMIX={self.ctx.last_mixing_scheme}, Zimp: {None})'
                description = f'KKRimp calculation of step {self.ctx.loop_count},'\
                    + f' using mixing scheme {self.ctx.last_mixing_scheme}'
                inputs = get_inputs_kkrimp(
                    code,
                    options,
                    label,
                    description,
                    params,
                    not self.ctx.withmpi,
                    host_GF=host_GF,
                    kkrimp_remote=last_remote,
                    host_GF_Efshift=host_GF_Efshift
                )

        # add LDA+U input node if it was set in parent calculation of last kkrimp_remote or from input port
        if self.ctx.settings_LDAU is not None:
            inputs['settings_LDAU'] = self.ctx.settings_LDAU

        # run the KKR calculation
        message = 'INFO: doing calculation'
        self.report(message)
        kkrimp_run = self.submit(KkrimpCalculation, **inputs)
        print('caching_info KKRimpCalc:', kkrimp_run.get_cache_source())
        print('hash: ', kkrimp_run.get_hash())
        print('_get_objects_to_hash: ', kkrimp_run._get_objects_to_hash())

        return ToContext(kkr=kkrimp_run, last_calc=kkrimp_run)

    def inspect_kkrimp(self):
        """
        check for convergence and store some of the results of the last calculation to context
        """

        self.ctx.calcs.append(self.ctx.last_calc)
        self.ctx.kkrimp_step_success = True

        # check calculation state
        if not self.ctx.last_calc.is_finished_ok:
            self.ctx.kkrimp_step_success = False
            message = 'ERROR: last calc not finished_ok'
            self.report(message)
            return self.exit_codes.ERROR_LAST_CALC_NOT_FINISHED_OK  # pylint: disable=maybe-no-member

        message = f'INFO: kkrimp_step_success: {self.ctx.kkrimp_step_success}'
        self.report(message)

        # get potential from last calculation
        try:
            retrieved_folder = self.ctx.kkr.outputs.retrieved
            imp_pot_sfd = extract_imp_pot_sfd(retrieved_folder)
            self.ctx.last_pot = imp_pot_sfd
            self.ctx.sfd_pot_to_clean.append(self.ctx.last_pot)
            print('use potfile sfd:', self.ctx.last_pot)
        except:
            message = 'ERROR: no output potential found'
            self.report(message)
            return self.exit_codes.ERROR_NO_OUTPUT_POT_FROM_LAST_CALC  # pylint: disable=maybe-no-member

        # extract convergence info about rms etc. (used to determine convergence behavior)
        try:
            message = f'INFO: trying to find output of last_calc: {self.ctx.last_calc}'
            self.report(message)
            last_calc_output = self.ctx.last_calc.outputs.output_parameters.get_dict()
            found_last_calc_output = True
        except:
            found_last_calc_output = False
        message = f'INFO: found_last_calc_output: {found_last_calc_output}'
        self.report(message)

        # try to extract remote folder
        try:
            self.ctx.last_remote = self.ctx.kkr.outputs.remote_folder
        except:
            self.ctx.last_remote = None
            self.ctx.kkrimp_step_success = False

        message = f'INFO: last_remote: {self.ctx.last_remote}'
        self.report(message)

        if self.ctx.kkrimp_step_success and found_last_calc_output:
            # check convergence
            self.ctx.kkr_converged = last_calc_output['convergence_group']['calculation_converged']
            # check rms
            self.ctx.rms.append(last_calc_output['convergence_group']['rms'])
            rms_all_iter_last_calc = list(last_calc_output['convergence_group']['rms_all_iterations'])

            # add lists of last iterations
            self.ctx.last_rms_all = rms_all_iter_last_calc
            if self.ctx.kkrimp_step_success and self.convergence_on_track():
                self.ctx.rms_all_steps += rms_all_iter_last_calc
        else:
            self.ctx.kkr_converged = False

        message = f'INFO: kkr_converged: {self.ctx.kkr_converged}'
        self.report(message)
        message = f'INFO: rms: {self.ctx.rms}'
        self.report(message)
        message = f'INFO: last_rms_all: {self.ctx.last_rms_all}'
        self.report(message)

        # turn off initial magnetization once one step was successful (update_kkr_params) used in
        if self.ctx.mag_init and self.convergence_on_track():  # and self.ctx.kkrimp_step_success:
            self.ctx.mag_init_step_success = True
        else:
            self.ctx.mag_init_step_success = False

        # store some statistics used to print table in the end of the report
        tmplist = self.ctx.KKR_steps_stats.get('success', [])
        message = f'INFO: append kkr_step_success {tmplist}, {self.ctx.kkr_step_success}'
        self.report(message)
        tmplist.append(self.ctx.kkr_step_success)
        self.ctx.KKR_steps_stats['success'] = tmplist
        try:
            isteps = self.ctx.last_calc.outputs.output_parameters.get_dict(
            )['convergence_group']['number_of_iterations']
        except:
            self.ctx.warnings.append('cound not set isteps in KKR_steps_stats dict')
            isteps = -1

        try:
            first_rms = self.ctx.last_rms_all[0]
            last_rms = self.ctx.last_rms_all[-1]
        except:
            self.ctx.warnings.append('cound not set first_rms, last_rms in KKR_steps_stats dict')
            first_rms = -1
            last_rms = -1

        if self.ctx.last_mixing_scheme == 0:
            mixfac = self.ctx.strmix
        elif self.ctx.last_mixing_scheme > 2:
            mixfac = self.ctx.aggrmix

        if self.ctx.kkr_higher_accuracy:
            qbound = self.ctx.convergence_criterion
        else:
            qbound = self.ctx.threshold_aggressive_mixing

        # store some values in self.ctx.KKR_steps_stats
        for name, val in {
            'isteps': isteps,
            'imix': self.ctx.last_mixing_scheme,
            'mixfac': mixfac,
            'qbound': qbound,
            'high_sett': self.ctx.kkr_higher_accuracy,
            'first_rms': first_rms,
            'last_rms': last_rms,
            'pk': self.ctx.last_calc.pk,
            'uuid': self.ctx.last_calc.uuid
        }.items():
            tmplist = self.ctx.KKR_steps_stats.get(name, [])
            tmplist.append(val)
            self.ctx.KKR_steps_stats[name] = tmplist

        message = 'INFO: done inspecting kkrimp results step'
        self.report(message)
        return None

    def convergence_on_track(self):
        """
        Check if convergence behavior of the last calculation is on track (i.e. going down)
        """

        on_track = True
        threshold = 5.  # used to check condition if at least one of charnge_neutrality, rms-error goes down fast enough

        # first check if previous calculation was stopped due to reaching the QBOUND limit
        try:
            calc_reached_qbound = self.ctx.last_calc.outputs.output_parameters.get_dict(
            )['convergence_group']['calculation_converged']
        except AttributeError:  # captures error when last_calc dies not have an output node
            calc_reached_qbound = False
        except KeyError:  # captures
            calc_reached_qbound = False

        if self.ctx.kkrimp_step_success and not calc_reached_qbound:
            first_rms = self.ctx.last_rms_all[0]
            # skip first if this is the initial LDA+U iteration because there
            # we see the original non-LDAU convergence value
            if 'settings_LDAU' in self.inputs and self.ctx.loop_count < 2 and len(self.ctx.last_rms_all) > 1:
                first_rms = self.ctx.last_rms_all[1]
            last_rms = self.ctx.last_rms_all[-1]
            # use this trick to avoid division by zero
            if last_rms == 0:
                last_rms = 10**-16
            r = last_rms / first_rms
            message = f'INFO: convergence check: first/last rms {first_rms}, {last_rms}'
            self.report(message)
            if r < 1:
                message = 'INFO: convergence check: rms goes down'
                self.report(message)
                on_track = True
            elif r > threshold:
                message = 'INFO: convergence check: rms goes up too fast, convergence is not expected'
                self.report(message)
                on_track = False
            elif len(self.ctx.last_rms_all) == 1:
                message = 'INFO: convergence check: already converged after single iteration'
                self.report(message)
                on_track = True
            else:
                message = 'INFO: convergence check: rms does not shrink fast enough, convergence is not expected'
                self.report(message)
                on_track = False
        elif calc_reached_qbound:
            message = 'INFO: convergence check: calculation reached QBOUND'
            self.report(message)
            on_track = True
        else:
            message = 'INFO: convergence check: calculation unsuccessful'
            self.report(message)
            on_track = False

        message = f'INFO: convergence check result: {on_track}'
        self.report(message)

        return on_track

    def return_results(self):
        """Return the results of the calculations.

        This should run through and produce output nodes even if everything failed,
        therefore it only uses results from context.
        """

        message = 'INFO: entering return_results'
        self.report(message)

        # try/except to capture as mnuch as possible (everything that is there even when workflow exits unsuccessfully)
        # capture pk and uuids of last calc, params and remote
        try:
            last_calc_uuid = self.ctx.last_calc.uuid
            last_calc_pk = self.ctx.last_calc.pk
            last_params_uuid = self.ctx.last_params.uuid
            last_params_pk = self.ctx.last_params.pk
            last_remote_uuid = self.ctx.last_remote.uuid
            last_remote_pk = self.ctx.last_remote.pk
        except:
            last_calc_uuid = None
            last_calc_pk = None
            last_params_uuid = None
            last_params_pk = None
            last_remote_uuid = None
            last_remote_pk = None

        all_pks = []
        for calc in self.ctx.calcs:
            try:
                all_pks.append(calc.pk)
            except:
                self.ctx.warnings.append(f'cound not get pk of calc {calc}')

        # capture links to last parameter, calcualtion and output
        try:
            last_calc_out = self.ctx.kkr.out['output_parameters']
            last_calc_out_dict = last_calc_out.get_dict()
            last_RemoteData = self.ctx.last_remote
            last_InputParameters = self.ctx.last_params
        except:
            last_InputParameters = None
            last_RemoteData = None
            last_calc_out = None
            last_calc_out_dict = {}

        # capture convergence info
        try:
            last_rms = self.ctx.rms[-1]
        except:
            last_rms = None

        # now collect results saved in results node of workflow
        message = 'INFO: collect outputnode_dict'
        self.report(message)
        outputnode_dict = {}
        outputnode_dict['workflow_name'] = self.__class__.__name__
        outputnode_dict['workflow_version'] = self._workflowversion
        outputnode_dict['material'] = self.ctx.formula
        outputnode_dict['loop_count'] = self.ctx.loop_count
        outputnode_dict['warnings'] = self.ctx.warnings
        outputnode_dict['successful'] = self.ctx.successful
        outputnode_dict['last_params_nodeinfo'] = {'uuid': last_params_uuid, 'pk': last_params_pk}
        outputnode_dict['last_remote_nodeinfo'] = {'uuid': last_remote_uuid, 'pk': last_remote_pk}
        outputnode_dict['last_calc_nodeinfo'] = {'uuid': last_calc_uuid, 'pk': last_calc_pk}
        outputnode_dict['pks_all_calcs'] = all_pks
        outputnode_dict['convergence_value'] = last_rms
        outputnode_dict['convergence_values_all_steps'] = np.array(self.ctx.rms_all_steps)
        outputnode_dict['convergence_values_last_step'] = np.array(self.ctx.last_rms_all)
        outputnode_dict['convergence_reached'] = self.ctx.kkr_converged
        outputnode_dict['kkr_step_success'] = self.ctx.kkr_step_success
        outputnode_dict['used_higher_accuracy'] = self.ctx.kkr_higher_accuracy

        # report the status
        if self.ctx.successful:
            self.report(
                f'STATUS: Done, the convergence criteria are reached.\n'
                f'INFO: The charge density of the KKR calculation pk= {last_calc_pk} '
                f'converged after {self.ctx.loop_count - 1} KKR runs and'
                f' {sum(self.ctx.KKR_steps_stats.get("isteps", []))} '
                f'iterations to {self.ctx.last_rms_all[-1]} \n'
            )
        else:  # Termination ok, but not converged yet...
            self.report(
                f'STATUS/WARNING: Done, the maximum number of runs '
                f'was reached or something failed.\n INFO: The '
                f'charge density of the KKR calculation pk= '
                f'after {self.ctx.loop_count - 1} KKR runs and'
                f' {sum(self.ctx.KKR_steps_stats.get("isteps", []))}'
                f' iterations is {self.ctx.last_rms_all[-1]} "me/bohr^3"\n'
            )

        # create results node and link all calculations
        message = 'INFO: create results nodes'
        self.report(message)
        link_nodes = {}
        icalc = 0
        for calc in self.ctx.calcs:
            link_nodes[f'KkrimpCalc{icalc}'] = calc.outputs.remote_folder
            icalc += 1
        if not self.ctx.dos_run:
            link_nodes['final_imp_potential'] = self.ctx.last_pot
        outputnode_t = create_out_dict_node(orm.Dict(dict=outputnode_dict), **link_nodes)
        outputnode_t.label = 'kkr_scf_wc_results'
        outputnode_t.description = 'Contains results of workflow (e.g. workflow version number, info about success of wf, lis tof warnings that occured during execution, ...)'

        self.out('workflow_info', outputnode_t)
        # store out_potential as SingleFileData only if this was no DOS run
        if not self.ctx.dos_run:
            self.out('host_imp_pot', self.ctx.last_pot)

        # print results table for overview
        # table layout:
        message = 'INFO: overview of the result:\n\n'
        message += '|------|---------|--------|------|--------|---------|-----------------|---------------------------------------------|\n'
        message += '| irun | success | isteps | imix | mixfac | qbound  |       rms       |                pk and uuid                  |\n'
        message += '|      |         |        |      |        |         | first  |  last  |                                             |\n'
        message += '|------|---------|--------|------|--------|---------|--------|--------|---------------------------------------------|\n'
        KKR_steps_stats = self.ctx.KKR_steps_stats
        for irun in range(len(KKR_steps_stats.get('success', []))):
            message += f'|{irun+1:6d}|'\
                + f'{KKR_steps_stats.get("success")[irun]:9s}|'\
                + f'{KKR_steps_stats.get("isteps")[irun]:8d}|'\
                + f'{KKR_steps_stats.get("imix")[irun]:6d}|'\
                + f'{KKR_steps_stats.get("mixfac")[irun]:.2e}|'\
                + f'{KKR_steps_stats.get("qbound")[irun]:.3e}|'\
                + f'{KKR_steps_stats.get("first_rms")[irun]:.2e}|'\
                + f'{KKR_steps_stats.get("last_rms")[irun]:.2e}|'
            message += f' {KKR_steps_stats.get("pk")[irun]} | {KKR_steps_stats.get("uuid")[irun]}|\n'
            message += '|------|---------|--------|------|--------|---------|-----------------|---------------------------------------------|\n'
        self.report(message)

        # cleanup of unnecessary files after convergence
        # WARNING: THIS DESTROYS CACHABILITY OF THE WORKFLOW!!!
        if self.ctx.do_final_cleanup:
            if self.ctx.successful:
                self.report('INFO: clean output of calcs')
                remove_out_pot_impcalcs(self.ctx.successful, all_pks)
                self.report('INFO: clean up raw_input folders')
                clean_raw_input(self.ctx.successful, all_pks)

            # clean intermediate single file data which are not needed after successful run or after DOS run
            if self.ctx.successful or self.ctx.dos_run:
                self.final_cleanup()

        self.report('INFO: done with kkr_scf workflow!\n')

    def error_handler(self):
        """Capture errors raised in validate_input"""
        if self.ctx.exit_code is not None:
            return self.ctx.exit_code

    def final_cleanup(self):
        uuid_last_calc = self.ctx.last_pot.uuid
        if not self.ctx.dos_run:
            sfds_to_clean = [i for i in self.ctx.sfd_pot_to_clean if i.uuid != uuid_last_calc]
        else:
            # in case of DOS run we can also clean the last output sfd file since this is never used
            sfds_to_clean = self.ctx.sfd_pot_to_clean
        # now clean all sfd files that are not needed anymore
        for sfd_to_clean in sfds_to_clean:
            clean_sfd(sfd_to_clean)


def remove_out_pot_impcalcs(successful, pks_all_calcs, dry_run=False):
    """
    Remove out_potential file from all but the last KKRimp calculation if workflow was successful

    Usage::

        imp_wf = load_node(266885) # maybe start with outer workflow
        pk_imp_scf = imp_wf.outputs.workflow_info['used_subworkflows'].get('kkr_imp_sub')
        imp_scf_wf = load_node(pk_imp_scf) # this is now the imp scf sub workflow
        successful = imp_scf_wf.outputs.workflow_info['successful']
        pks_all_calcs = imp_scf_wf.outputs.workflow_info['pks_all_calcs']
    """
    from aiida.common.folders import SandboxFolder
    from aiida_kkr.calculations import KkrimpCalculation

    if dry_run:
        print('test', successful, len(pks_all_calcs))

    # name of tarfile
    tfname = KkrimpCalculation._FILENAME_TAR

    # cleanup only if calculation was successful
    if successful and len(pks_all_calcs) > 1:
        # remove out_potential for calculations
        # note that also last calc can be cleaned since output potential is stored in single file data
        pks_for_cleanup = pks_all_calcs[:]

        # loop over all calculations
        for pk in pks_for_cleanup:
            if dry_run:
                print('pk_for_cleanup:', pk)
            # get getreived folder of calc
            calc = orm.load_node(pk)
            ret = calc.outputs.retrieved

            # open tarfile if present
            if tfname in ret.list_object_names():
                delete_and_retar = False
                with ret.open(tfname) as tf:
                    tf_abspath = tf.name

                # create Sandbox folder which is used to temporarily extract output_all.tar.gz
                tmpfolder = SandboxFolder()
                tmpfolder_path = tmpfolder.abspath
                with tarfile.open(tf_abspath) as tf:
                    tar_filenames = [ifile.name for ifile in tf.getmembers()]
                    # check if out_potential is in tarfile
                    if KkrimpCalculation._OUT_POTENTIAL in tar_filenames:
                        tf.extractall(tmpfolder_path)
                        delete_and_retar = True

                if delete_and_retar and not dry_run:
                    # delete out_potential
                    os.remove(os.path.join(tmpfolder_path, KkrimpCalculation._OUT_POTENTIAL))
                    with tarfile.open(tf_abspath, 'w:gz') as tf:
                        # remove out_potential from list of files
                        tar_filenames = [i for i in tar_filenames if i != KkrimpCalculation._OUT_POTENTIAL]
                        for f in tar_filenames:
                            # create new tarfile without out_potential file
                            fabs = os.path.join(tmpfolder_path, f)
                            tf.add(fabs, arcname=os.path.basename(fabs))
                elif dry_run:
                    print('dry run:')
                    print('delete and retar?', delete_and_retar)
                    print('tmpfolder_path', tmpfolder_path)

                # clean up temporary Sandbox folder
                if not dry_run:
                    tmpfolder.erase()


def clean_raw_input(successful, pks_calcs, dry_run=False):
    """
    Clean raw_input directories that contain copies of shapefun and potential files
    This however breaks provenance (strictly speaking) and therefore should only be done
    for the calculations of a successfully finished workflow (see email on mailing list from 25.11.2019).
    """
    from aiida_kkr.calculations import KkrimpCalculation
    if successful:
        for pk in pks_calcs:
            node = orm.load_node(pk)
            # clean only nodes that are KkrimpCalculations
            if node.process_class == KkrimpCalculation:
                raw_input_folder = node._raw_input_folder
                # clean potential and shapefun files
                for filename in [KkrimpCalculation._POTENTIAL, KkrimpCalculation._SHAPEFUN]:
                    if filename in raw_input_folder.get_content_list():
                        if dry_run:
                            print(f'clean {filename}')
                        else:
                            raw_input_folder.remove_path(filename)
    elif dry_run:
        print('no raw_inputs to clean')


def clean_sfd(sfd_to_clean, nkeep=30):
    """
    Clean up potential file (keep only header) to save space in the repository
    WARNING: this breaks cachability!
    """
    with sfd_to_clean.open(sfd_to_clean.filename) as _file:
        txt = _file.readlines()
    # remove all lines after nkeep lines
    txt2 = txt[:nkeep]
    # add note to end of file
    txt2 += [u'WARNING: REST OF FILE WAS CLEANED SO SAVE SPACE!!!\n']
    # overwrite file
    with sfd_to_clean.open(sfd_to_clean.filename, 'w') as fnew:
        fnew.writelines(txt2)


@calcfunction
def extract_imp_pot_sfd(retrieved_folder):
    """
    Extract potential file from retrieved folder and save as SingleFileData
    """
    # take output potential file either from tarfile or directy from output folder

    if KkrimpCalculation._FILENAME_TAR in retrieved_folder.list_object_names():
        print('take potfile from tar file of retrieved')
        # take potfile after extracting tar file
        # get full filename
        with retrieved_folder.open(KkrimpCalculation._FILENAME_TAR) as tar_file:
            tarfilename = tar_file.name
        print('tarfile name:', tarfilename)
        # open tarfile and extract potfile
        with tarfile.open(tarfilename) as tar_file:
            print('extract potfile:', KkrimpCalculation._OUT_POTENTIAL)
            tar_file.extract(KkrimpCalculation._OUT_POTENTIAL, os.path.dirname(tarfilename))
            with retrieved_folder.open(KkrimpCalculation._OUT_POTENTIAL, 'rb') as pot_file:
                print('get potfile sfd:', pot_file)
                imp_pot_sfd = orm.SinglefileData(file=pot_file)

        # delete extracted potfile again
        print('delete potfile from outfile:', KkrimpCalculation._OUT_POTENTIAL)
        retrieved_folder.delete_object(KkrimpCalculation._OUT_POTENTIAL, force=True)
    else:
        # take potfile directly from output
        with retrieved_folder.open(KkrimpCalculation._OUT_POTENTIAL, 'rb') as pot_file:
            imp_pot_sfd = orm.SinglefileData(file=pot_file)

    return imp_pot_sfd
