import openmdao.api as om
from .scenario import Scenario
from .coupling_bemaerostructural import CouplingBEMAeroStructural


class ScenarioBEMAeroStructural(Scenario):
    def initialize(self):
        """
        A class to perform a coupled BEM-aerodynamic case.
        The Scenario will add the BEM and aerodynamic builders' precoupling subsystems,
        the coupling subsystems, and the postcoupling subsystems.
        """
        super().initialize()

        self.options.declare("bem_builder",recordable=False,desc="The MPhys builder for the BEM solver")
        self.options.declare('bem_ldxfer_builder', recordable=False)
        self.options.declare("cfd_builder",recordable=False,desc="The MPhys builder for the CFD solver")
        self.options.declare('cfd_ldxfer_builder', recordable=False)
        self.options.declare("struct_builder",recordable=False,desc="The MPhys builder for the structural solver")

    def _mphys_scenario_setup(self):
        bem_builder = self.options["bem_builder"]
        bem_ldxfer_builder = self.options['bem_ldxfer_builder']
        cfd_builder = self.options['cfd_builder']
        cfd_ldxfer_builder = self.options['cfd_ldxfer_builder']
        struct_builder = self.options['struct_builder']

        self._mphys_add_pre_coupling_subsystem_from_builder("bem", bem_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder("cfd", cfd_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder("struct", struct_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder("bem_ldxfer", bem_ldxfer_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder("cfd_ldxfer", cfd_ldxfer_builder, self.name)

        coupling_group = CouplingBEMAeroStructural(
            bem_builder=bem_builder,
            bem_ldxfer_builder=bem_ldxfer_builder,
            cfd_builder=cfd_builder,
            cfd_ldxfer_builder=cfd_ldxfer_builder,
            struct_builder=struct_builder,
            scenario_name=self.name,
        )
        self.mphys_add_subsystem("coupling", coupling_group)

        self._mphys_add_post_coupling_subsystem_from_builder("cfd_ldxfer", cfd_ldxfer_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder("bem_ldxfer", bem_ldxfer_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder("bem", bem_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder("cfd", cfd_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder("struct", struct_builder, self.name)
