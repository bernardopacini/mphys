import openmdao.api as om
from .scenario import Scenario
from .coupling_bemaerodynamic import CouplingBEMAerodynamic


class ScenarioBEMAerodynamic(Scenario):
    def initialize(self):
        """
        A class to perform a coupled BEM-aerodynamic case.
        The Scenario will add the BEM and aerodynamic builders' precoupling subsystems,
        the coupling subsystems, and the postcoupling subsystems.
        """
        super().initialize()

        self.options.declare("bem_builder",recordable=False,desc="The MPhys builder for the BEM solver")
        self.options.declare("aero_builder",recordable=False,desc="The MPhys builder for the aerodynamic solver")

    def _mphys_scenario_setup(self):
        bem_builder = self.options["bem_builder"]
        aero_builder = self.options["aero_builder"]

        self._mphys_add_pre_coupling_subsystem_from_builder(
            "bem", bem_builder, self.name
        )
        self._mphys_add_pre_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )

        coupling_group = CouplingBEMAerodynamic(
            bem_builder=bem_builder,
            aero_builder=aero_builder,
            scenario_name=self.name,
        )

        self.mphys_add_subsystem("coupling", coupling_group)

        self._mphys_add_post_coupling_subsystem_from_builder(
            "bem", bem_builder, self.name
        )
        self._mphys_add_post_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )
