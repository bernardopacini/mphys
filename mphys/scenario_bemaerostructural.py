from mphys.scenario import Scenario
from mphys.coupling_bemaerostructural import CouplingBEMAeroStructural
from mphys import MaskedConverter, UnmaskedConverter, MaskedVariableDescription

import numpy as np

class ScenarioBEMAeroStructural(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()

        self.options.declare(
            "bem_builder",
            recordable=False,
            desc="The MPhys builder for the BEM solver",
        )
        self.options.declare(
            "cfd_builder",
            recordable=False,
            desc="The MPhys builder for the CFD solver",
        )
        self.options.declare(
            "struct_builder",
            recordable=False,
            desc="The MPhys builder for the structural solver",
        )
        self.options.declare(
            "ldxfer_builder",
            recordable=False,
            desc="The MPhys builder for the load and displacement transfer",
        )
        self.options.declare(
            "pre_coupling_order",
            default=["bem", "cfd", "unmasker", "struct", "ldxfer"],
            recordable=False,
            desc="The order of the pre coupling subsystems",
        )
        self.options.declare(
            "post_coupling_order",
            default=["ldxfer", "bem", "cfd", "struct"],
            recordable=False,
            desc="The order of the post coupling subsystems",
        )

    def _mphys_scenario_setup(self):
        self._mphys_add_pre_coupling_subsystems()
        self._mphys_add_coupling_group()
        self._mphys_add_post_coupling_subsystems()

    def _mphys_check_coupling_order_inputs(self, given_options):
        valid_options = ["bem", "cfd", "unmasker", "struct", "ldxfer"]

        length = len(given_options)
        if length > 5:
            raise ValueError(
                f"Specified too many items in the pre/post coupling order list, len={length}"
            )

        for option in given_options:
            if option not in valid_options:
                raise ValueError(
                    f"""Unknown pre/post order option: {option}. valid options are ["{'", "'.join(valid_options)}"]"""
                )

    def _mphys_add_pre_coupling_subsystems(self):
        self._mphys_check_coupling_order_inputs(self.options["pre_coupling_order"])
        for discipline in self.options["pre_coupling_order"]:
            if discipline == "unmasker":
                # Add Unmasking for coordinates
                n_nodes_bem = self.options["bem_builder"].get_number_of_nodes()
                n_nodes_cfd = self.options["cfd_builder"].get_number_of_nodes()
                n_nodes = n_nodes_bem + n_nodes_cfd

                mask = []
                input = []
                promotes_inputs = []
                promotes_outputs = []

                # BEM Inputs
                mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
                mask[0][:] = False
                mask[0][:n_nodes_bem * 3] = True
                input.append(MaskedVariableDescription("x_bem0", shape=(n_nodes_bem) * 3, tags=["mphys_coupling"]))
                promotes_inputs.append("x_bem0")

                # CFD Inputs
                mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
                mask[1][:] = False
                mask[1][n_nodes_bem * 3:] = True
                input.append(MaskedVariableDescription("x_cfd0", shape=(n_nodes_cfd) * 3, tags=["mphys_coupling"]))
                promotes_inputs.append("x_cfd0")

                # Combined Outputs
                output = MaskedVariableDescription("x_aero0", shape=(n_nodes) * 3, tags=["mphys_coupling"])
                promotes_outputs.append("x_aero0")
                unmasker = UnmaskedConverter(input=input, output=output, mask=mask, distributed=True, default_values=0.0)
                self.add_subsystem(
                    "AeroCoordUnmasker", unmasker, promotes_inputs=promotes_inputs, promotes_outputs=promotes_outputs
                )
            else:
                self._mphys_add_pre_coupling_subsystem_from_builder(
                    discipline, self.options[f"{discipline}_builder"], self.name
                )

    def _mphys_add_coupling_group(self):
        coupling_group = CouplingBEMAeroStructural(
            bem_builder=self.options["bem_builder"],
            cfd_builder=self.options["cfd_builder"],
            struct_builder=self.options["struct_builder"],
            ldxfer_builder=self.options["ldxfer_builder"],
            scenario_name=self.name,
        )
        self.mphys_add_subsystem("coupling", coupling_group)

    def _mphys_add_post_coupling_subsystems(self):
        self._mphys_check_coupling_order_inputs(self.options["post_coupling_order"])
        for discipline in self.options["post_coupling_order"]:
            self._mphys_add_post_coupling_subsystem_from_builder(
                discipline, self.options[f"{discipline}_builder"], self.name
            )

    def _mphys_initialize_builders(self):
        self.options["bem_builder"].initialize(self.comm)
        self.options["cfd_builder"].initialize(self.comm)
        self.options["struct_builder"].initialize(self.comm)
        self.options["ldxfer_builder"].initialize(self.comm)

    def _mphys_add_mesh_and_geometry_subsystems(self):
        cfd_builder = self.options["cfd_builder"]
        struct_builder = self.options["struct_builder"]

        self.mphys_add_subsystem(
            "cfd_mesh", cfd_builder.get_mesh_coordinate_subsystem(self.name)
        )
        self.mphys_add_subsystem(
            "struct_mesh", struct_builder.get_mesh_coordinate_subsystem(self.name)
        )