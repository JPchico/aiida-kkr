from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, append_, BaseRestartWorkChain,\
    process_handler, ProcessHandlerReport, ExitCode, while_
from aiida_kkr.calculations.kkr import KkrCalculation


class BaseKKRWorkchain(BaseRestartWorkChain):

    _process_class = KkrCalculation

    @classmethod
    def define(cls, spec):

        super(BaseKKRWorkchain, cls).define(spec)
        spec.expose_inputs(KkrCalculation, exclude=("metadata",))
        spec.input(
            "options", valid_type=orm.Dict, required=False,
            help="""
            Optional parameters to affect the way the calculation job and the
            parsing are performed.
            """
        )
        spec.input(
            "settings", valid_type=orm.Dict, required=False,
            help="Extra parameters needed for the calculation"
        )
        spec.expose_outputs(KkrCalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

    def setup(self):
        super().setup()

        self.ctx.restart_calc = None
        self.ctx.inputs = AttributeDict()
        self.ctx.inputs.code = self.inputs.code
        self.ctx.inputs.parameters = self.inputs.parameters

        if "impurity_info" in self.inputs:
            self.ctx.inputs.impurity_info = self.inputs.impurity_info

        if "kpoints" in self.inputs:
            self.ctx.inputs.kpoints = self.inputs.kpoints

        if "initial_noco_angles" in self.inputs:
            self.ctx.inputs.initial_noco_angles = self.inputs.initial_noco_angles

        if "deciout_parent" in self.inputs:
            self.ctx.inputs.deciout_parent = self.inputs.deciout_parent

        if "options" in self.inputs:
            _options = self.inputs.options.get_dict()
            if "resources" not in _options:
                _resources = {
                    "num_machines": 1,
                    "tot_num_mpiprocs": 1
                }
                _options["resources"] = _resources
            if "max_wallclock_seconds" not in _options:
                _options["max_wallclock_seconds"] = 3600
            if "withmpi" not in _options:
                _options["withmpi"] = False

        self.ctx.inputs.metadata = AttributeDict()
        self.ctx.inputs.metadata.options = _options

        if "settings" in self.inputs:
            _settings = self.inputs.settings.get_dict()
            if "description" in _settings:
                self.ctx.inputs.metadata.description = _settings["description"]
            if "label" in _settings:
                self.ctx.inputs.metadata.label = _settings["label"]
            if "dry_run" in _settings:
                self.ctx.inputs.metadata.dry_run = _settings["dry_run"]

        if "parent_folder" in self.ctx.inputs:
            self.ctx.restart_calc = self.ctx.inputs.parent_folder.creator

        if "parent_folder" in self.inputs:
            self.ctx.inputs.parent_folder = self.inputs.parent_folder

    def prepare_process(self):
        """
        Prepare the inputs for the next calculation.
        If a `restart_calc` has been set in the context, its `remote_folder`
        will be used as the `parent_folder` input for the next calculation and
        the `restart_mode` is set to `restart`. Otherwise, no `parent_folder`
        is used and `restart_mode` is set to `from_scratch`.
        """
        if self.ctx.restart_calc:
            self.ctx.inputs.parent_folder = self.ctx.restart_calc.outputs.remote_folder

    def report_error_handled(self, calculation, action):
        """
        Report an action taken for a calculation that has failed.
        This should be called in a registered error handler if its condition is
        met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [
            calculation.process_label, calculation.pk,
            calculation.exit_status, calculation.exit_message
        ]
        self.report("{}<{}> failed with exit status {}: {}".format(*arguments))
        for _action in action:
            self.report("Action taken: {}".format(_action))
