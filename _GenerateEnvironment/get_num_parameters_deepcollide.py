import numpy as np
from cspace_net import CSpaceNet
import matplotlib.pyplot as plt

def get_dependence_on_dof(dof, num_freq):

    model = CSpaceNet(dof=42*3, num_freq=12, sigma=2)

    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    model_parameters = [x for x in filter(lambda p: p.requires_grad, model.parameters())]
    # dependent_parameters_on_dof = []
    # for p in model_parameters:
    #     if any(elem > 256 for elem in list(p.size())):
    #         dependent_parameters_on_dof.append(p)
    #     print(p.size())

    total_num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('total:', total_num_params)

    # total_num_dependent_params = sum([np.prod(p.size()) for p in dependent_parameters_on_dof]) - (256 * 256)
    # print('dependent on DoF:', total_num_dependent_params)

    # percent_dependent_on_dof = total_num_dependent_params / total_num_params
    # print('{:.4f} dependent on DoF'.format(percent_dependent_on_dof))

    return total_num_params

