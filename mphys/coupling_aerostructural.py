import openmdao.api as om
import numpy as np
from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp

from mphys import MaskedConverter, UnmaskedConverter, MaskedVariableDescription
nodes_prop = 5

class CouplingAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('ldxfer_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        ldxfer_builder = self.options['ldxfer_builder']
        scenario_name = self.options['scenario_name']

        disp_xfer, load_xfer = ldxfer_builder.get_coupling_group_subsystem(scenario_name)
        aero = aero_builder.get_coupling_group_subsystem(scenario_name)
        struct = struct_builder.get_coupling_group_subsystem(scenario_name)

        if self.comm.rank == 0:
            geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes() + nodes_prop)
        else:
            geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes())

        self.mphys_add_subsystem('disp_xfer', disp_xfer)
        self.mphys_add_subsystem('geo_disp', geo_disp)

        # TEMPORARY -- Adding here to do MPHYS propeller masking
        masking = True
        if masking:
            number_of_nodes = aero_builder.get_number_of_nodes()

            if self.comm.rank == 0:
                mesh_mask = np.zeros([(number_of_nodes + nodes_prop)*3], dtype=bool)
                mesh_mask[:] = True
                mesh_mask[-3*nodes_prop:] = False

                prop_mask = np.zeros([(number_of_nodes + nodes_prop)*3], dtype=bool)
                prop_mask[:] = False
                prop_mask[-3*nodes_prop:] = True

                mask_input = MaskedVariableDescription("x_aero_unmasked", shape=(number_of_nodes+nodes_prop)*3,tags=['mphys_coupling'])
                mask_output_prop = MaskedVariableDescription("x_prop", shape=(nodes_prop)*3, tags=['mphys_coupling'])

                unmask_input_prop = MaskedVariableDescription("f_aero_prop", shape=(nodes_prop)*3, tags=['mphys_coupling'])
                unmask_output = MaskedVariableDescription("f_aero_unmasked", shape=(number_of_nodes + nodes_prop)*3,tags=['mphys_coupling'])

            else:
                mesh_mask = np.zeros([number_of_nodes*3], dtype=bool)
                mesh_mask[:] = True

                prop_mask = np.zeros([number_of_nodes*3], dtype=bool)
                prop_mask[:] = False

                mask_input = MaskedVariableDescription("x_aero_unmasked", shape=(number_of_nodes)*3,tags=['mphys_coupling'])
                mask_output_prop = MaskedVariableDescription("x_prop", shape=(0)*3, tags=['mphys_coupling'])

                unmask_input_prop = MaskedVariableDescription("f_aero_prop", shape=(0)*3, tags=['mphys_coupling'])
                unmask_output = MaskedVariableDescription("f_aero_unmasked", shape=(number_of_nodes)*3,tags=['mphys_coupling'])

            mask_output_mesh = MaskedVariableDescription("x_aero", shape=(number_of_nodes)*3, tags=['mphys_coupling'])
            unmask_input_mesh = MaskedVariableDescription("f_aero", shape=(number_of_nodes)*3, tags=['mphys_coupling'])

            masker = MaskedConverter(input=mask_input, output=[mask_output_mesh, mask_output_prop], mask=[mesh_mask, prop_mask], distributed=True, init_output=0.0)
            self.mphys_add_subsystem('masker', masker)

        self.mphys_add_subsystem('aero', aero)

        # TEMPORARY --- Adding Masker for DAFoam
        if masking:
            unmasker = UnmaskedConverter(input=[unmask_input_mesh, unmask_input_prop], output=unmask_output, mask=[mesh_mask,prop_mask], distributed=True, default_values=0.0)
            self.mphys_add_subsystem('unmasker', unmasker)

        self.mphys_add_subsystem('load_xfer', load_xfer)
        self.mphys_add_subsystem('struct', struct)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
                                                    atol=1e-8, rtol=1e-8,
                                                    use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
                                              atol=1e-8, rtol=1e-8,
                                              use_aitken=True)
