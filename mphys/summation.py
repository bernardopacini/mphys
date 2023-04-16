import openmdao.api as om


class SummationVariableDescription:
    """
    Attributes of a variable for conversion from masked to unmasked
    or unmasked to masked
    """

    def __init__(self, name: str, tags=None):
        self.name = name
        self.tags = tags

class Summation(om.ExplicitComponent):
    """
    An ExplicitComponent used to filter out a predefined set of indices from a larger input array.
    This is useful in cases, for instance, where it desired to prevent certain fea nodes from participating
    in the load and displacement transfers in an aerostructural scenario.

    The masking operation breaks down to the python assignment:
    masked_vector = unmasked_vector[mask_indices]
    """

    def initialize(self):
        self.options.declare(
            'input',
            desc='Summation object of input that will be summed')
        self.options.declare(
            'output',
            desc='Summation output')
        self.options.declare(
            'init_output', default=1.0,
            desc='initail value of the ouput. The default value matches the default value for val in add_output')
        self.options.declare(
            'distributed', default=False,
            desc='Flag to determine if the inputs and outputs should be distributed arrays')

    def setup(self):
        distributed = self.options['distributed']
        input = self.options['input']
        output = self.options['output']

        for i in range(len(input)):
            self.add_input(input[i].name, shape_by_conn=True, tags=input[i].tags, val=self.options['init_output'], distributed=distributed)
        self.add_output(output.name, copy_shape=input[0].name, tags=output.tags, distributed=distributed)

    def compute(self, inputs, outputs):
        input = self.options['input']
        output = self.options['output']

        outputs[output.name][:] = 0.0
        for i in range(len(input)):
            outputs[output.name] += inputs[input[i].name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        input = self.options['input']
        output = self.options['output']

        for i in range(len(input)):
            if mode == 'fwd':
                if input[i].name in d_inputs and output.name in d_outputs:
                    d_outputs[output.name] += d_inputs[input[i].name]

            if mode == 'rev':
                if input[i].name in d_inputs and output.name in d_outputs:
                    d_inputs[input[i].name] += d_outputs[output.name]
