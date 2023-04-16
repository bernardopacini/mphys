import openmdao.api as om
from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp
from .summation import Summation, SummationVariableDescription

import numpy as np

class CouplingBEMAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare('bem_builder', recordable=False)
        self.options.declare('bem_ldxfer_builder', recordable=False)
        self.options.declare('cfd_builder', recordable=False)
        self.options.declare('cfd_ldxfer_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        bem_builder = self.options["bem_builder"]
        bem_ldxfer_builder = self.options['bem_ldxfer_builder']
        cfd_builder = self.options['cfd_builder']
        cfd_ldxfer_builder = self.options['cfd_ldxfer_builder']
        struct_builder = self.options['struct_builder']
        scenario_name = self.options['scenario_name']

        bem = bem_builder.get_coupling_group_subsystem(scenario_name)
        cfd = cfd_builder.get_coupling_group_subsystem(scenario_name)
        struct = struct_builder.get_coupling_group_subsystem(scenario_name)

        bem_disp_xfer, bem_load_xfer = bem_ldxfer_builder.get_coupling_group_subsystem(scenario_name)
        cfd_disp_xfer, cfd_load_xfer = cfd_ldxfer_builder.get_coupling_group_subsystem(scenario_name)

        bem_geo_disp = GeoDisp(number_of_nodes=bem_builder.get_number_of_nodes())
        cfd_geo_disp = GeoDisp(number_of_nodes=cfd_builder.get_number_of_nodes())

        input = []
        input.append(SummationVariableDescription(name="f_struct_bem"))
        input.append(SummationVariableDescription(name="f_struct_cfd"))
        output = SummationVariableDescription(name="f_struct", tags=["mphys_coupling"])
        summation = Summation(input=input, output=output, distributed=True, init_output = 0.0)

        self.mphys_add_subsystem('bem_disp_xfer', bem_disp_xfer)
        self.mphys_add_subsystem('bem_geo_disp', bem_geo_disp)
        self.mphys_add_subsystem('bem', bem)
        self.mphys_add_subsystem('cfd_disp_xfer', cfd_disp_xfer)
        self.mphys_add_subsystem('cfd_geo_disp', cfd_geo_disp)
        self.mphys_add_subsystem('cfd', cfd)
        self.mphys_add_subsystem('bem_load_xfer', bem_load_xfer)
        self.mphys_add_subsystem('cfd_load_xfer', cfd_load_xfer)
        self.mphys_add_subsystem('f_summation', summation)
        self.mphys_add_subsystem('struct', struct)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
                                                    atol=1e-8, rtol=1e-8,
                                                    use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
                                              atol=1e-8, rtol=1e-8,
                                              use_aitken=True)
