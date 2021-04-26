import sys
import cPickle as cp
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', help='Save directory.')
parser.add_argument('--max_n', type=int, help='Upper bound on graph size.')
parser.add_argument('--min_n', type=int,  help='Lower bound on graph size.')
parser.add_argument('--num_graph', type=int, help='Number of graphs to generate')
parser.add_argument('--p', type=float, help='Connectivity parameter.')
parser.add_argument('--n_comp', type=int, help='Number of connected components.')
args = parser.parse_args()


def get_component():
    """Generate a connected ER component with min_n <= n <= max_n."""
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    g = nx.erdos_renyi_graph(n=cur_n, p=p)

    comps = [c for c in nx.connected_component_subgraphs(g)]
    random.shuffle(comps)
    for i in range(1, len(comps)):
        x = random.choice(comps[i - 1].nodes())
        y = random.choice(comps[i].nodes())
        g.add_edge(x, y)
    assert nx.is_connected(g)
    return g


if __name__ == '__main__':
    max_n = args.max_n
    min_n = args.min_n
    p = args.p
    n_comp = args.n_comp

    fout_name = '%s/ncomp-%d-nrange-%d-%d-n_graph-%d-p-%.2f.pkl' % (args.save_dir, n_comp, min_n, max_n, args.num_graph, p)
    print('Final Output: ' + fout_name)
    print("Generating graphs...")
    min_n = min_n // n_comp
    max_n = max_n // n_comp

    for i in tqdm(range(args.num_graph)):

        for j in range(n_comp):
            g = get_component()
            
            if j == 0:
                g_all = g
            else:
                g_all = nx.disjoint_union(g_all, g)
        assert nx.number_connected_components(g_all) == n_comp

        with open(fout_name, 'ab') as fout:
            cp.dump(g_all, fout, cp.HIGHEST_PROTOCOL)
