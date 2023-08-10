"""
Calculate the statistical moments and save their corresponding graphs into the
    output directory.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def visualize(dates, red, green, blue, group_key, out_dir, prefix, stat_name):
    fig = plt.figure()
    x = list(range(len(dates)))
    plt.plot(x, red, color='red', label='Red')
    plt.plot(x, green, color='green', label='Green')
    plt.plot(x, blue, color='blue', label='Blue')
    plt.xticks(x, dates, rotation=90)
    plt.legend()
    plt.title(f'{stat_name} {prefix} ' + group_key.title())
    plt.grid(True)
    plt.savefig(fname=os.path.join(out_dir,
                                   f'{stat_name}_{prefix}_{group_key.upper()}.png'),
                dpi=600,
                format='png',
                orientation='portrait',
                pad_inches=0.1,
                bbox_inches='tight')

if __name__ == '__main__':
    extracted_info_path = 'data/batch1/op/colors.json'
    out_dir = 'data/batch2/test_dir/'
    prefix = 'OP'
    with open(extracted_info_path, 'r') as fin:
        colors = json.load(fin)

    for group_key in colors.keys():
        print('Process:', group_key)
        group = colors[group_key].items()
        dates = []
        red = {'mean': [],
               'var': [],
               'ris': []}
        green = {'mean': [],
                 'var': [],
                 'ris': []}
        blue = {'mean': [],
                'var': [],
                'ris': []}
        for item in group:
            dates.append(item[0])
            r, g, b = np.array(item[1]).mean(axis=0) / 255
            red['mean'].append(r)
            green['mean'].append(g)
            blue['mean'].append(b)
            r, g, b = np.array(item[1]).var(axis=0) / 255
            red['var'].append(r)
            green['var'].append(g)
            blue['var'].append(b)
            # Calculate the Relative intensity smoothness Criteria.
            red['ris'].append(1 - (1 / (1 + r)))
            green['ris'].append(1 - (1 / (1 + g)))
            blue['ris'].append(1 - (1 / (1 + b)))

        visualize(dates, red['mean'], green['mean'], blue['mean'],
                  group_key, out_dir, prefix, stat_name='Mean')
        visualize(dates, red['var'], green['var'], blue['var'],
                  group_key, out_dir, prefix, stat_name='Variance')
        visualize(dates, red['ris'], green['ris'], blue['ris'],
                  group_key, out_dir, prefix, stat_name='RIS')
