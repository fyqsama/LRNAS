import torch
import utils
import model
import numpy as np
import torchattacks
from genotypes import Genotype, PRIMITIVES


'''
normal op params:  {'none': 0.0, 'sep_conv_3x3': 0.3598560000000002, 'sep_conv_5x5': 0.40824000000000016, 'dil_conv_3x3': 0.17992799999999987, 'dil_conv_5x5': 0.20412000000000008, 'max_pool_3x3': 0.0, 'skip_connect': 0.0, 'avg_pool_3x3': 0.0}
reduce op params:  {'none': 0.0, 'sep_conv_3x3': 0.0565920000000002, 'sep_conv_5x5': 0.063504, 'dil_conv_3x3': 0.0282960000000001, 'dil_conv_5x5': 0.031752, 'max_pool_3x3': 0.0, 'skip_connect': 0.026352000000000153, 'avg_pool_3x3': 0.0}
delta_params_normal = [0.0, 0.0, 0.0, 0.0, 0.3598560000000002, 0.40824000000000016, 0.17992799999999987, 0.20412000000000008]
delta_params_reduce = [0.0, 0.0, 0.0, 0.026352000000000153, 0.0565920000000002, 0.063504, 0.0282960000000001, 0.031752]
'''


def remove_op(normal_weights, reduce_weights, op):
    selected_cell = str(op.split('_')[0])
    selected_eid = int(op.split('_')[1])
    opid = int(op.split('_')[-1])
    proj_mask = torch.ones_like(normal_weights[selected_eid])
    proj_mask[opid] = 0
    if selected_cell in ['normal']:
        normal_weights[selected_eid] = normal_weights[selected_eid] * proj_mask
    else:
        reduce_weights[selected_eid] = reduce_weights[selected_eid] * proj_mask

    return normal_weights, reduce_weights


def compute_value(valid_queue, model, ops, num_samples):
    permutations = []
    for _ in range(num_samples):
        permutations.append(np.random.permutation(ops))

    eval_values_std = np.zeros((len(ops), num_samples))
    eval_values_adv = np.zeros((len(ops), num_samples))
    eval_values = np.zeros((len(ops), num_samples))

    for sample_iter in range(num_samples):
        input, target = next(iter(valid_queue))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits_std = model(input)
        std_acc, = utils.accuracy(logits_std, target)
        std_acc = std_acc.item()

        atk = torchattacks.FGSM(model, eps=8/255)
        input_adv = atk(input, target)
        logits_adv = model(input_adv)
        adv_acc, = utils.accuracy(logits_adv, target)
        adv_acc = adv_acc.item()

        normal_weights = model.get_projected_weights('normal')
        reduce_weights = model.get_projected_weights('reduce')

        print('Sampling permutation: %d' % (sample_iter + 1))

        with torch.no_grad():
            for op in permutations[sample_iter]:
                normal_weights, reduce_weights = remove_op(normal_weights, reduce_weights, op)

                cur_logits_std = model(input, weights_dict={'normal': normal_weights, 'reduce': reduce_weights})
                cur_std_acc, = utils.accuracy(cur_logits_std, target)
                cur_std_acc = cur_std_acc.item()

                cur_logits_adv = model(input_adv, weights_dict={'normal': normal_weights, 'reduce': reduce_weights})
                cur_adv_acc, = utils.accuracy(cur_logits_adv, target)
                cur_adv_acc = cur_adv_acc.item()

                delta_std_acc = std_acc - cur_std_acc
                delta_adv_acc = adv_acc - cur_adv_acc

                op_index = np.where(ops == op)[0].item()
                eval_values_std[op_index][sample_iter] = delta_std_acc
                eval_values_adv[op_index][sample_iter] = delta_adv_acc

        eval_values_std[:, sample_iter] = (eval_values_std[:, sample_iter] -
                                           np.mean(eval_values_std[:, sample_iter])) / np.std(eval_values_std[:, sample_iter])

        eval_values_adv[:, sample_iter] = (eval_values_adv[:, sample_iter] -
                                           np.mean(eval_values_adv[:, sample_iter])) / np.std(eval_values_adv[:, sample_iter])

    eval_values += eval_values_std + eval_values_adv

    normal_values = np.zeros((model.num_edges, model.num_ops))
    reduce_values = np.zeros((model.num_edges, model.num_ops))
    for i in range(len(ops)):
        op = ops[i]
        selected_cell = str(op.split('_')[0])
        selected_eid = int(op.split('_')[1])
        opid = int(op.split('_')[-1])
        '''
        for j in range(num_samples):
            if selected_cell == 'normal':
                eval_values[i][j] -= delta_params_normal[genotypes.PRIMITIVES[opid]]
            else:
                eval_values[i][j] -= delta_params_reduce[genotypes.PRIMITIVES[opid]]
        '''
        if selected_cell == 'normal':
            normal_values[selected_eid][opid] = np.mean(eval_values[i])
        else:
            reduce_values[selected_eid][opid] = np.mean(eval_values[i])

    return normal_values, reduce_values


def update_alpha(eval_values, prev_value, step_size=0.1, momentum=0.8):
    values = []
    for i in range(len(eval_values)):
        values.append(torch.from_numpy(eval_values[i]).cuda())

    inc = []
    for i in range(len(values)):
        mean = values[i].data.mean()
        std = values[i].data.std()
        values[i].data.add_(-mean).div_(std)

        v = momentum * prev_value[i] + (1 - momentum) * values[i]
        inc.append(v)

    delta_alpha_normal = step_size * inc[0]
    delta_alpha_reduce = step_size * inc[1]

    return [delta_alpha_normal, delta_alpha_reduce]


def ranking(alpha_normal, alpha_reduce, threshold, classes):
    alpha_normal = alpha_normal.cpu().numpy()

    alpha_reduce = alpha_reduce.cpu().numpy()

    # 0 represents the normal cell and 1 represents the reduction cell
    selected = []
    for i in range(len(alpha_normal)):
        value = np.max(alpha_normal[i])
        opid = np.argmax(alpha_normal[i])
        selected.append([0, i, opid, value])

    for i in range(len(alpha_reduce)):
        value = np.max(alpha_reduce[i])
        opid = np.argmax(alpha_reduce[i])
        selected.append([1, i, opid, value])
    selected = np.array(selected)

    nodes_normal = [0, 0, 0, 0]
    nodes_reduce = [0, 0, 0, 0]
    genotype = Genotype(normal=[['none', 0], ['none', 1], ['none', 0], ['none', 1], ['none', 0], ['none', 1], ['none', 0], ['none', 1]], normal_concat=[2, 3, 4, 5],
                        reduce=[['none', 0], ['none', 1], ['none', 0], ['none', 1], ['none', 0], ['none', 1], ['none', 0], ['none', 1]], reduce_concat=[2, 3, 4, 5])

    while selected.size != 0:
        op = np.argmax(selected[:, 3])
        test_model = model.NetworkCIFAR(36, classes, 20, False, genotype)
        total_params = utils.count_parameters_in_MB(test_model)

        add_or_not = False
        if total_params > threshold:
            if int(selected[op][2]) <= 3:
                add_or_not = True
        else:
            add_or_not = True

        if add_or_not == True:
            if selected[op][0] == 0:
                if int(selected[op][1]) >= 0 and int(selected[op][1]) <= 1:
                    genotype.normal[nodes_normal[0]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.normal[nodes_normal[0]][1] = int(selected[op][1])
                    nodes_normal[0] += 1

                elif int(selected[op][1]) >= 2 and int(selected[op][1]) <= 4 and nodes_normal[1] < 2:
                    genotype.normal[2 + nodes_normal[1]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.normal[2 + nodes_normal[1]][1] = int(selected[op][1]) - 2
                    nodes_normal[1] += 1

                elif int(selected[op][1]) >= 5 and int(selected[op][1]) <= 8 and nodes_normal[2] < 2:
                    genotype.normal[4 + nodes_normal[2]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.normal[4 + nodes_normal[2]][1] = int(selected[op][1]) - 5
                    nodes_normal[2] += 1

                elif int(selected[op][1]) >= 9 and int(selected[op][1]) <= 13 and nodes_normal[3] < 2:
                    genotype.normal[6 + nodes_normal[3]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.normal[6 + nodes_normal[3]][1] = int(selected[op][1]) - 9
                    nodes_normal[3] += 1

            else:
                if int(selected[op][1]) >= 0 and int(selected[op][1]) <= 1:
                    genotype.reduce[nodes_reduce[0]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.reduce[nodes_reduce[0]][1] = int(selected[op][1])
                    nodes_reduce[0] += 1

                elif int(selected[op][1]) >= 2 and int(selected[op][1]) <= 4 and nodes_reduce[1] < 2:
                    genotype.reduce[2 + nodes_reduce[1]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.reduce[2 + nodes_reduce[1]][1] = int(selected[op][1]) - 2
                    nodes_reduce[1] += 1

                elif int(selected[op][1]) >= 5 and int(selected[op][1]) <= 8 and nodes_reduce[2] < 2:
                    genotype.reduce[4 + nodes_reduce[2]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.reduce[4 + nodes_reduce[2]][1] = int(selected[op][1]) - 5
                    nodes_reduce[2] += 1

                elif int(selected[op][1]) >= 9 and int(selected[op][1]) <= 13 and nodes_reduce[3] < 2:
                    genotype.reduce[6 + nodes_reduce[3]][0] = PRIMITIVES[int(selected[op][2])]
                    genotype.reduce[6 + nodes_reduce[3]][1] = int(selected[op][1]) - 9
                    nodes_reduce[3] += 1

        selected = np.delete(selected, op, axis=0)

    return genotype