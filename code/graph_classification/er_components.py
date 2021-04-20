"""Train or evaluate a graph classifier."""
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import torch
import random
import torch.optim as optim
import cPickle as cp
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args, save_args
from dnn import GraphClassifier
sys.path.append('%s/../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset, load_er_data



if __name__ == '__main__':
    # set seed
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)    

    # load the data
    # label_map maps number of connected components to label i.e. {1: 0, 2: 1, 3: 2}
    # train_glist and test_glist are S2VGraph from the following link
    # https://github.com/Hanjun-Dai/pytorch_structure2vec/blob/bcf20c90f21e468f862f13e2f5809a52cd247d4e/graph_classification/util.py
    label_map, train_glist, test_glist = load_er_data()

    # load model if specified else create a new one
    if cmd_args.saved_model is not None and cmd_args.saved_model != '':        
        print('loading model from %s' % cmd_args.saved_model)
        with open('%s-args.pkl' % cmd_args.saved_model, 'rb') as f:
            base_args = cp.load(f)
        classifier = GraphClassifier(label_map, **vars(base_args))            
        classifier.load_state_dict(torch.load(cmd_args.saved_model + '.model'))
    else:
        classifier = GraphClassifier(label_map, **vars(cmd_args))

    # move classifier to gpu if available
    if cmd_args.ctx == 'gpu':
        classifier = classifier.cuda()

    # if phase is test look at the test accuracy and loss
    if cmd_args.phase == 'test':
        test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))))
        print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

    # if phase is train then fit parameters to the model
    if cmd_args.phase == 'train':
        optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

        train_idxes = list(range(len(train_glist)))
        best_loss = None
        for epoch in range(cmd_args.num_epochs):
            random.shuffle(train_idxes)
            avg_loss = loop_dataset(train_glist, classifier, train_idxes, optimizer=optimizer)
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
            
            test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))))
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

            if best_loss is None or test_loss[0] < best_loss:
                best_loss = test_loss[0]
                print('----saving to best model since this is the best valid loss so far.----')
                torch.save(classifier.state_dict(), cmd_args.save_dir + '/epoch-best.model')
                save_args(cmd_args.save_dir + '/epoch-best-args.pkl', cmd_args)