import openmdao.api as om
from .coupling_group import CouplingGroup


class CouplingBEMAerodynamic(CouplingGroup):
    """
    The standard BEM aerodynamic coupling problem.
    """

    def initialize(self):
        self.options.declare('bem_builder', recordable=False)
        self.options.declare('aero_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        bem_builder = self.options["bem_builder"]
        aero_builder = self.options['aero_builder']
        scenario_name = self.options['scenario_name']

        bem = bem_builder.get_coupling_group_subsystem(scenario_name)
        aero = aero_builder.get_coupling_group_subsystem(scenario_name)

        self.mphys_add_subsystem('bem', bem)
        self.mphys_add_subsystem('aero', aero)

    def _mphys_promote_coupling_variables(self):
        super()._mphys_promote_coupling_variables()

    def _mphys_promote_inputs(self):
        super()._mphys_promote_inputs()

    def _mphys_promote_mesh_coordinates(self):
        super()._mphys_promote_mesh_coordinates()

    def _mphys_promote_results(self):
        super()._mphys_promote_results()
