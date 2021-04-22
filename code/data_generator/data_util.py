import cPickle as cp

def load_pkl(fname, num_graph):
    """Load `num_graph` graphs from `fname`."""
    g_list = []
    with open(fname, 'rb') as f:
        for i in range(num_graph):
            g = cp.load(f)
            g_list.append(g)
    return g_list