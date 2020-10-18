"""
print results in Table 1 and Table S1 and S3 for CIFAR-10 and CIFAR-100 over multiple runs

Set the random seeds
    --seeds seed_0 seed_1 seed_2 ...

"""

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Print Results')
parser.add_argument('--seeds', nargs='+', type=int, metavar='lists of random seeds used in all experiments', default=[0,1,2,3,4],
                    help='0 1 2 3')
args = parser.parse_args()


def print_results_line_runs(model_name='Receiver', pickle_path='', eval_set='cifar100', seeds=args.seeds):
    acc = []
    for i in seeds:
        with open(pickle_path + '_' + str(i)+'.pickle', 'rb') as f:
            results = pickle.load(f)
            res = results[eval_set]
            acc.append(res['acc'].item())

    line = model_name
    line += '& $%3.2f \\pm %3.2f$ ' % (100 * np.mean(acc), 100 * np.std(acc))
    print('& ', line, '\\\\')
    print('Accs of all trails: ', acc)


def print_exp1_table_runs(eval_set='cifar100'):

    # ks =3
    print('======> %s' % eval_set)
    print("******* ks = 3")
    print_results_line_runs(model_name='SmallNet',
                            pickle_path='results/evals/tiny_' + eval_set + '_3', eval_set=eval_set)
    print_results_line_runs(model_name='FC(Arora18)',
                            pickle_path='results/evals/tiny_' + eval_set + '_3_enet_fc', eval_set=eval_set)
    print_results_line_runs(model_name='ExpandNet-CL',
                            pickle_path='results/evals/tiny_' + eval_set + '_3_enet_cl', eval_set=eval_set)
    print_results_line_runs(model_name='ExpandNet-CL+FC',
                            pickle_path='results/evals/tiny_' + eval_set + '_3_enet_cl_fc', eval_set=eval_set)

    # ks =7
    print("******* ks = 7")
    print_results_line_runs(model_name='SmallNet',
                            pickle_path='results/evals/tiny_' + eval_set + '_7', eval_set=eval_set)
    print_results_line_runs(model_name='FC(Arora18)',
                            pickle_path='results/evals/tiny_' + eval_set + '_7_enet_fc', eval_set=eval_set)
    print_results_line_runs(model_name='ExpandNet-CL',
                            pickle_path='results/evals/tiny_' + eval_set + '_7_enet_cl', eval_set=eval_set)
    print_results_line_runs(model_name='ExpandNet-CL+FC',
                            pickle_path='results/evals/tiny_' + eval_set + '_7_enet_cl_fc', eval_set=eval_set)

    print_results_line_runs(model_name='ExpandNet-CK',
                            pickle_path='results/evals/tiny_' + eval_set + '_7_enet_ck', eval_set=eval_set)
    print_results_line_runs(model_name='ExpandNet-CK+FC',
                            pickle_path='results/evals/tiny_' + eval_set + '_7_enet_ck_fc', eval_set=eval_set)


if __name__ == '__main__':

    print_exp1_table_runs(eval_set='cifar10')

    print_exp1_table_runs(eval_set='cifar100')
