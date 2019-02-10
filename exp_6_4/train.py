from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import exp_6_4.problems as pblm
from exp_6_4.trainer import *
import math
import numpy as np

def select_model(m): 
    if m == 'large': 
        model = pblm.mnist_model_large()
        _, test_loader = pblm.mnist_loaders(8)
    elif m == 'wide': 
        print("Using wide model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//args.model_factor)
        model = pblm.mnist_model_wide(args.model_factor)
    elif m == 'deep': 
        print("Using deep model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//(2**args.model_factor))
        model = pblm.mnist_model_deep(args.model_factor)
    else: 
        model = pblm.mnist_model()
    return model


if __name__ == "__main__": 
    args = pblm.argparser(opt='adam', verbose=200, starting_epsilon=0.01)
    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    train_log = open(f'./snapshots/{args.prefix}_train.log', "w")
    test_log = open(f'./snapshots/{args.prefix}_test.log', "w") 

    # DEBUG
    print('arguments', args)
    print('')
    #raise Exception()

    train_loader, test_loader = pblm.mnist_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for X,y in train_loader: 
        break
    kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
    best_err = 1

    sampler_indices = []
    model = [select_model(args.model)]

    # Load model parameters if necessary!
    if args.model_snapshot is not None:
      print(f'loading snapshot {args.model_snapshot}')
      checkpoint = torch.load(args.model_snapshot)

      #for n, p in model[0].named_parameters():
      #  #print(p.size(), checkpoint[n].size())
      #  p.data.copy_(checkpoint[n])

      #for c in checkpoint.values():
      #  print(c.device)
      model[0].load_state_dict(checkpoint)
      del checkpoint
      torch.cuda.empty_cache()
      #print('inside loading snapshot')
      #input()
      #raise Exception()

    train_errors = []
    train_losses = []
    test_errors = []
    test_losses = []
    train_robust_errors = []
    train_robust_losses = []
    test_robust_errors = []
    test_robust_losses = []

    model[0].cuda()
    for _ in range(0,args.cascade): 
        if _ > 0: 
            # reduce dataset to just uncertified examples
            print("Reducing dataset...")
            train_loader = sampler_robust_cascade(train_loader, model, args.epsilon, **kwargs)
            if train_loader is None: 
                print('No more examples, terminating')
                break
            sampler_indices.append(train_loader.sampler.indices)

            print("Adding a new model")
            model.append(select_model(args.model))
        
        if args.opt == 'adam': 
            opt = optim.Adam(model[-1].parameters(), lr=args.lr)
        elif args.opt == 'sgd': 
            opt = optim.SGD(model[-1].parameters(), lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        else: 
            raise ValueError("Unknown optimizer")
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        eps_schedule = np.linspace(args.starting_epsilon, 
                                   args.epsilon, 
                                   args.schedule_length)

        for t in range(args.epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if t < len(eps_schedule) and args.starting_epsilon is not None: 
                epsilon = float(eps_schedule[t])
            else:
                epsilon = args.epsilon


            # standard training
            if args.method == 'baseline': # and t < 3: 
                print('Training baseline...')
                train_err, train_loss = train_baseline(train_loader, model[0], opt, t, train_log,
                                args.verbose)
                test_err, test_loss = evaluate_baseline(test_loader, model[0], t, test_log,
                                args.verbose)

                train_errors.append(train_err)
                train_losses.append(train_loss)
                test_errors.append(test_err)
                test_losses.append(test_loss)

            # madry training
            elif args.method=='madry':
                train_madry(train_loader, model[0], args.epsilon, 
                            opt, t, train_log, args.verbose)
                err = evaluate_madry(test_loader, model[0], args.epsilon, 
                                     t, test_log, args.verbose)

            # robust cascade training
            elif args.cascade > 1: 
                train_robust(train_loader, model[-1], opt, epsilon, t,
                                train_log, args.verbose, args.real_time,
                                l1_type=args.l1_train, bounded_input=True,
                                **kwargs)
                err = evaluate_robust_cascade(test_loader, model,
                   args.epsilon, t, test_log, args.verbose,
                   l1_type=args.l1_test, bounded_input=True,  **kwargs)

            # robust training
            else:
                print('Training robust version of model...')

                train_err, train_loss, train_robust_err, train_robust_loss = train_robust(train_loader, model[0], opt, epsilon, t,
                   train_log, args.verbose, args.real_time,
                   l1_type=args.l1_train, bounded_input=True, **kwargs)

                #raise Exception()

                test_err, test_loss, test_robust_err, test_robust_loss = evaluate_robust(test_loader, model[0], args.epsilon,
                   t, test_log, args.verbose, args.real_time,
                   l1_type=args.l1_test, bounded_input=True, **kwargs)

                train_errors.append(train_err)
                train_losses.append(train_loss)
                test_errors.append(test_err)
                test_losses.append(test_loss)
                train_robust_errors.append(train_robust_err)
                train_robust_losses.append(train_robust_loss)
                test_robust_errors.append(test_robust_err)
                test_robust_losses.append(test_robust_loss)
                
            model[0].cpu()
            torch.save(model[0].state_dict(), f'./snapshots/{args.prefix}_checkpoint_{t}.pth')
            model[0].cuda()

            r = {'train_errors': train_errors,
                'train_losses': train_losses,
                'test_errors': test_errors,
                'test_losses': test_losses,
                'train_robust_errors': train_robust_errors,
                'train_robust_losses': train_robust_losses,
                'test_robust_errors': test_robust_errors,
                'test_robust_losses': test_robust_losses,
                'epoch': t,
                'sampler_indices': sampler_indices
                }
            with open(f'./snapshots/{args.prefix}_results_{t}.pth', 'wb') as handle:
              pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)