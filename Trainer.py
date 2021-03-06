"""
Trainer class. Handles training and validation
"""
import gc
from helpers import get_gpu_memory_map
from Dataloader import Dataloader
# from Model import DeepVO
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from utils import se3qua

class Trainer():

    def __init__(self, args, epoch, model, train_set, val_set, loss_fn, optimizer, scheduler=None, \
                 gradClip=None):
        super(Trainer, self).__init__()
        # Commandline arguments
        self.args = args

        # Maximum number of epochs to train for
        self.maxEpochs = self.args.nepochs
        # Current epoch (initally set to -1)
        self.curEpoch = epoch

        # Model to train
        self.model = model

        # Train and validataion sets (Dataset objects)
        self.train_set = train_set
        self.val_set = val_set

        # Loss function
        self.loss_fn = nn.MSELoss(reduction='sum')

        # Variables to hold loss

        self.loss_r6 = torch.zeros(1, dtype=torch.float32).cuda()
        self.loss_xyzq = torch.zeros(1, dtype=torch.float32).cuda()

        self.loss = torch.zeros(1, dtype=torch.float32).cuda()

        # Optimizer
        self.optimizer = optimizer

        # Scheduler
        self.scheduler = scheduler

        # Flush gradient buffers before beginning training
        self.model.zero_grad()

        # Keep track of number of iters (useful for tensorboardX visualization)
        self.iters = 0
        self.abs_traj = None

    # Train for one epoch
    def train(self):

        # Switch model to train mode
        self.model.train()

        # Check if maxEpochs have elapsed
        if self.curEpoch >= self.maxEpochs:
            print('Max epochs elapsed! Returning ...')
            return

        # Increment iters
        self.iters += 1

        # Variables to store stats

        r6Losses = []
        poseLosses = []
        totalLosses = []
        r6Loss_seq = []
        poseLoss_seq = []
        totalLoss_seq = []

        # Handle debug mode here
        if self.args.debug is True:
            numTrainIters = self.args.debugIters
        else:
            numTrainIters = len(self.train_set)

        elapsedBatches = 0
        traj_pred = None
        gen = trange(numTrainIters)
        print("gen",gen)
        # assert False
        # Run a pass of the dataset
        for i in gen:

            if self.args.profileGPUUsage is True:
                gpu_memory_map = get_gpu_memory_map()
                tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file=sys.stdout)

            # Get the next frame
            inp, imu, r6, xyzq, _, _, _,timestamp, endOfSeq = self.train_set[i]
            pred_r6 = self.model.forward(inp, imu, xyzq)
            # del inp
            # del imu
            if self.abs_traj is None:
                self.abs_traj = xyzq.data.cpu().detach()[0][0]
                # Feed it through the model
            numarr = pred_r6.data.cpu().detach().numpy()[0][0]

            self.abs_traj = se3qua.accu(self.abs_traj,numarr)

            abs_traj_input = np.expand_dims(self.abs_traj, axis=0)
            abs_traj_input = np.expand_dims(abs_traj_input, axis=0)
            abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor)).cuda()

            curloss_r6= Variable(self.args.scf * (torch.dist(pred_r6, r6) ** 2), requires_grad=False)
            curloss_xyzq = Variable(self.args.scf * (torch.dist(abs_traj_input, xyzq) ** 2), requires_grad=False)
            # self.loss_r6 = curloss_r6.item()
            # self.loss_xyzq = curloss_xyzq.item()


            # if np.random.normal() < -0.9:
            #     tqdm.write('r6(pred,gt): ' + str(pred_r6.data)+' '+ str(r6.data) ,file=sys.stdout)
            #     tqdm.write('pose(pred,gt): ' + str(abs_traj_input.data) + ' '+str(xyzq.data), file=sys.stdout)

            self.loss += sum([(self.args.scf * self.loss_fn(pred_r6, r6)).item(), \
                              (self.loss_fn(abs_traj_input, xyzq)).item()])

            curloss_r6= curloss_r6.detach().cpu().numpy()
            curloss_xyzq = curloss_xyzq.detach().cpu().numpy()
            r6Losses.append(curloss_r6)
            r6Loss_seq.append(curloss_r6)
            poseLosses.append(curloss_xyzq)
            poseLoss_seq.append(curloss_xyzq)
            totalLosses.append(curloss_r6 + curloss_xyzq)
            totalLoss_seq.append(curloss_r6 + curloss_xyzq)
            del curloss_r6
            del curloss_xyzq

            # Handle debug mode here. Force execute the below if statement in the
            # last debug iteration
            if self.args.debug is True:
                if i == numTrainIters - 1:
                    endOfSeq = True

            elapsedBatches += 1

            # if endOfSeq is True:
            if endOfSeq is True:
                elapsedBatches = 0

                # if self.args.gamma > 0.0:
                #     paramsDict = self.model.state_dict()
                #     # print(paramsDict.keys())
                #
                #     if self.args.numLSTMCells == 1:
                #         reg_loss = None
                #         reg_loss = paramsDict['lstm1.weight_ih'].norm(2)
                #         reg_loss += paramsDict['lstm1.weight_hh'].norm(2)
                #         reg_loss += paramsDict['lstm1.bias_ih'].norm(2)
                #         reg_loss += paramsDict['lstm1.bias_hh'].norm(2)
                #     else:
                #         reg_loss = None
                #         # reg_loss = paramsDict['rnnIMU.weight_ih_l0'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.weight_hh_l0'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.bias_ih_l0'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.bias_hh_l0'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.weight_ih_l1'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.weight_Hh_l1'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.bias_ih_l1'].norm(2)
                #         # reg_loss += paramsDict['rnnIMU.bias_Hh_l1'].norm(2)
                #         reg_loss = paramsDict['rnn.weight_ih_l0'].norm(2)
                #         reg_loss += paramsDict['rnn.weight_hh_l0'].norm(2)
                #         reg_loss += paramsDict['rnn.bias_ih_l0'].norm(2)
                #         reg_loss += paramsDict['rnn.bias_hh_l0'].norm(2)
                #         reg_loss += paramsDict['rnn.weight_ih_l1'].norm(2)
                #         reg_loss += paramsDict['rnn.weight_Hh_l1'].norm(2)
                #         reg_loss += paramsDict['rnn.bias_ih_l1'].norm(2)
                #         reg_loss += paramsDict['rnn.bias_Hh_l1'].norm(2)
                #     self.loss = sum([self.args.gamma * reg_loss, self.loss])
                tqdm.write('r6 Loss: ' + str(np.mean(r6Loss_seq)) + 'pose Loss' + str(np.mean(poseLoss_seq)), file=sys.stdout)
                r6Loss_seq = []
                poseLoss_seq = []
                totalLoss_seq = []

                # Compute gradients
                self.loss.backward()

                paramList = list(filter(lambda p: p.grad is not None, [param for param in self.model.parameters()]))
                totalNorm = sum([(p.grad.data.norm(2.) ** 2.) for p in paramList]) ** (1. / 2)
                tqdm.write('gradNorm: ' + str(totalNorm.item()))

                # Perform gradient clipping, if enabled
                if self.args.gradClip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradClip)

                # Update parameters
                self.optimizer.step()

                # If it's the end of sequence, reset hidden states
                # if endOfSeq is True:
                #     self.model.reset_LSTM_hidden()
                # self.model.detach_LSTM_hidden()  # ???

                # Reset loss variables
                self.loss_r6 = torch.zeros(1, dtype=torch.float32).cuda()
                self.loss_xyzq = torch.zeros(1, dtype=torch.float32).cuda()
                self.loss = torch.zeros(1, dtype=torch.float32).cuda()

                # Flush gradient buffers for next forward pass
                self.model.zero_grad()
                self.abs_traj = None

        return r6Losses, poseLosses, totalLosses

    # Run one epoch of validation
    def validate(self):

        # Switch model to eval mode
        self.model.eval()

        # Run a pass of the dataset
        traj_pred = None
        self.abs_traj = None

        # Variables to store stats
        Losses_seq= []
        TotalLoss_seq = []

        # Handle debug switch here
        if self.args.debug is True:
            numValIters = self.args.debugIters
        else:
            numValIters = len(self.val_set)

        # Choose a generator (for iterating over the dataset, based on whether or not the
        # sbatch flag is set to True). If sbatch is True, we're probably running on a cluster
        # and do not want an interactive output. So, could suppress tqdm and print statements
        if self.args.sbatch is True:
            gen = range(numValIters)
        else:
            gen = trange(numValIters)

        for i in gen:

            if self.args.profileGPUUsage is True:
                gpu_memory_map = get_gpu_memory_map()
                tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file=sys.stdout)

            # Get the next frame
            inp, imu, r6, xyzq, seq, frame1, frame2, timestamp, endOfSeq = self.val_set[i]

            metadata = np.asarray([timestamp])

            # Feed it through the model
            pred_r6 = self.model.forward(inp, imu, xyzq)
            numarr = pred_r6.data.cpu().detach().numpy()
            if self.abs_traj is None:
                self.abs_traj = xyzq.data.cpu().detach()[0][0]
            if traj_pred is None:
                traj_pred = np.concatenate((metadata, self.abs_traj.numpy()), axis=0)
                traj_pred = np.resize(traj_pred, (1, -1))

            self.abs_traj = se3qua.accu(self.abs_traj, numarr)

            cur_pred = np.concatenate((metadata, self.abs_traj), axis=0)
            traj_pred = np.append(traj_pred, np.resize(cur_pred, (1, -1)), axis=0)

            # Store losses (for further analysis)

            curloss_r6 = (self.args.scf * self.loss_fn(pred_r6, r6)).detach().cpu().numpy()
            curloss_xyzq = (self.loss_fn(self.abs_traj, xyzq)).detach().cpu().numpy()
            Losses_seq.append(curloss_r6 + curloss_xyzq)
            TotalLoss_seq.append(curloss_r6 + curloss_xyzq)

            # Detach hidden states and outputs of LSTM
            # self.model.detach_LSTM_hidden()

            if endOfSeq is True:

                # Print stats

                tqdm.write('Total Loss: ' + str(np.mean(TotalLoss_seq)), file=sys.stdout)

                # Write predicted trajectory to file
                saveFile = os.path.join(self.args.expDir, 'plots', 'traj', str(seq).zfill(2), \
                                        'traj_' + str(self.curEpoch).zfill(3) + '.txt')
                np.savetxt(saveFile, traj_pred, newline='\n')

                # Reset variable, to store new trajectory later on
                traj_pred = None

                # Detach LSTM hidden states
                # self.model.detach_LSTM_hidden()

                # Reset LSTM hidden states
                # self.model.reset_LSTM_hidden()
                self.abs_traj = None
                self.model.zero_grad()

        return curloss_r6, curloss_xyzq, Losses_seq
