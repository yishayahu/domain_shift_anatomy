import itertools
import json

import numpy as np
from matplotlib import pyplot as plt

def plot_stats(stats):
    def get_common_couples(size,sgd_or_adam):
        ks = [f's_{s} t_{t}' for s,t in itertools.combinations(range(6),2)]
        for exp_name in stats.keys():
            if sgd_or_adam == 'adam' and 'adam' not in exp_name:
                continue
            if sgd_or_adam == 'sgd' and 'adam'  in exp_name:
                continue
            not_in = [x for x in ks if x not in stats[exp_name][size].keys()]
            for x in not_in:
                ks.remove(x)
        return ks
    target_sizes = [1,2,4]
    stats['oracle_sgd'] ={}
    stats['oracle_adam'] ={}
    for size in target_sizes:
        stats['oracle_sgd'][size] = {'s_0 t_1':0.84, 's_1 t_4':0.86, 's_2 t_3':0.857, 's_3 t_5':0.836, 's_4 t_5':0.857, 's_0 t_2':0.84}
        stats['oracle_adam'][size] = {'s_0 t_1':0.95, 's_1 t_4':0.935, 's_2 t_3':0.95, 's_3 t_5':0.945, 's_4 t_5':0.952, 's_0 t_2':0.95}
    n_to_s_sgd = {'posttrain':'base','gradual_tl':'g_DA','spottune':'st','posttrain_continue_optimizer':'c_o','clustering':'clus','oracle_sgd':'oracle'}
    n_to_s_adam = {'posttrain_adam':'base', 'gradual_tl_adam':'g_DA', 'spottune_adam':'st', 'posttrain_continue_optimizer_adam':'c_o','spot_with_grad_adam':'comb',
                   'posttrain_continue_optimizer_from_step_adam':'c_o_fs',
                   'gradual_tl__continue_optimzer_adam_from_step':'g_da_co_fs','oracle_adam':'oracle','clustering_adam_start_from_sgd':'clus'}

    def plot_stats_aux(sgd_or_adam):


        n_to_s =  n_to_s_sgd if sgd_or_adam =='sgd' else n_to_s_adam
        for size in target_sizes:
            names = []
            means = []
            errors = []
            all1 = []
            ks = get_common_couples(size,sgd_or_adam)
            print(f'len ks is {len(ks)}')
            for exp_name, exp_stats in stats.items():
                curr_stats = [v for k,v in stats[exp_name][size].items() if v > 0 and k in ks]
                if sgd_or_adam == 'adam' and 'adam' not in exp_name:
                    continue
                if sgd_or_adam == 'sgd' and 'adam'  in exp_name:
                    continue
                all1.append((n_to_s[exp_name],np.mean(list(curr_stats)),np.std(list(curr_stats))))

            for n,m,s in all1:
                names.append(n)
                means.append(m)
                errors.append(s)
            print(f'ts {size}, means {means},erros {errors}')

            fig, ax = plt.subplots()
            rects = ax.bar(list(range(len(names))), means, yerr=errors, align='center', alpha=0.8, ecolor='black', capsize=10,color=['b','g','r','y','m'])
            ax.set_ylabel('sdice score')
            ax.set_xticks(list(range(len(names))))
            ax.set_xticklabels(names)
            plt.title(f'{sgd_or_adam} size {size}')

            # Save the figure and show
            plt.tight_layout()
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.3f' % height,
                        ha='center', va='bottom')
            plt.savefig(f'bar_plot_with_error_{sgd_or_adam}.png')
            plt.show()
            plt.cla()
            plt.clf()

    plot_stats_aux('sgd')
    plot_stats_aux('adam')

if __name__ == '__main__':
    plot_stats(json.load(open('all_stats.json','r')))