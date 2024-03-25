import openmdao.api as om
from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp

from mphys import MaskedConverter, UnmaskedConverter, MaskedVariableDescription

import numpy as np

class CouplingBEMAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare('bem_builder', recordable=False)
        self.options.declare('cfd_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('ldxfer_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        bem_builder = self.options['bem_builder']
        cfd_builder = self.options['cfd_builder']
        struct_builder = self.options['struct_builder']
        ldxfer_builder = self.options['ldxfer_builder']
        scenario_name = self.options['scenario_name']

        bem = bem_builder.get_coupling_group_subsystem(scenario_name)
        cfd = cfd_builder.get_coupling_group_subsystem(scenario_name)
        disp_xfer, load_xfer = ldxfer_builder.get_coupling_group_subsystem(scenario_name)
        struct = struct_builder.get_coupling_group_subsystem(scenario_name)

        n_nodes_bem = bem_builder.get_number_of_nodes()
        n_nodes_cfd = cfd_builder.get_number_of_nodes()
        n_nodes = n_nodes_bem + n_nodes_cfd
        geo_disp = GeoDisp(number_of_nodes=n_nodes)

        self.mphys_add_subsystem('disp_xfer', disp_xfer)
        self.mphys_add_subsystem('geo_disp', geo_disp)
        masker, masker_promotes_inputs, masker_promotes_outputs = self.setup_masking(n_nodes_bem, n_nodes_cfd, n_nodes)
        self.add_subsystem("masker", masker, promotes_inputs=masker_promotes_inputs, promotes_outputs=masker_promotes_outputs)
        self.mphys_add_subsystem('bem', bem)
        self.mphys_add_subsystem('cfd', cfd)
        unmasker, unmasker_promotes_inputs, unmasker_promotes_outputs = self.setup_unmasking(n_nodes_bem, n_nodes_cfd, n_nodes)
        self.add_subsystem("unmasker", unmasker, promotes_inputs=unmasker_promotes_inputs, promotes_outputs=unmasker_promotes_outputs)
        self.mphys_add_subsystem('load_xfer', load_xfer)
        self.mphys_add_subsystem('struct', struct)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
                                                    atol=1e-8, rtol=1e-8,
                                                    use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
                                              atol=1e-8, rtol=1e-8,
                                              use_aitken=True)

    def setup_masking(self, n_nodes_bem, n_nodes_cfd, n_nodes):
        mask = []
        output = []
        promotes_inputs = []
        promotes_outputs = []

        input = MaskedVariableDescription("x_aero", shape=n_nodes * 3, tags=["mphys_coupling"])
        promotes_inputs.append("x_aero")

        mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
        mask[0][:] = False
        mask[0][:n_nodes_bem * 3] = True
        output.append(MaskedVariableDescription("x_bem", shape=n_nodes_bem * 3, tags=["mphys_coupling"]))
        promotes_outputs.append("x_bem")

        mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
        mask[1][:] = False
        mask[1][n_nodes_bem * 3:] = True
        output.append(MaskedVariableDescription("x_cfd", shape=n_nodes_cfd * 3, tags=["mphys_coupling"]))
        promotes_outputs.append("x_cfd")

        masker = MaskedConverter(input=input, output=output, mask=mask, distributed=True, init_output=0.0)
        return masker, promotes_inputs, promotes_outputs

    def setup_unmasking(self, n_nodes_bem, n_nodes_cfd, n_nodes):
        mask = []
        input = []
        promotes_inputs = []
        promotes_outputs = []

        mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
        mask[0][:] = False
        mask[0][:n_nodes_bem * 3] = True
        input.append(MaskedVariableDescription("f_bem", shape=n_nodes_bem * 3, tags=["mphys_coupling"]))
        promotes_inputs.append("f_bem")

        mask.append(np.zeros([(n_nodes) * 3], dtype=bool))
        mask[1][:] = False
        mask[1][n_nodes_bem * 3:] = True
        input.append(MaskedVariableDescription("f_cfd", shape=n_nodes_cfd * 3, tags=["mphys_coupling"]))
        promotes_inputs.append("f_cfd")

        output = MaskedVariableDescription("f_aero", shape=n_nodes * 3, tags=["mphys_coupling"])
        promotes_outputs.append("f_aero")

        unmasker = UnmaskedConverter(input=input, output=output, mask=mask, distributed=True, default_values=0.0)
        return unmasker, promotes_inputs, promotes_outputs
