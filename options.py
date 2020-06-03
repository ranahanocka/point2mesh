import argparse
import os
import numpy as np
import torch

MANIFOLD_DIR = r'~/code/Manifold/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)


class Options:
    def __init__(self):
        self.args = None
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Point2Mesh options')
        parser.add_argument('--save-path', type=str, default='./checkpoints/guitar', help='path to save results to')
        parser.add_argument('--input-pc', type=str, default='./data/guitar.ply', help='input point cloud')
        parser.add_argument('--initial-mesh', type=str, default='./data/guitar_initmesh.obj', help='initial mesh')
        # HYPER PARAMETERS - RECONSTRUCTION
        parser.add_argument('--torch-seed', type=int, metavar='N', default=5, help='torch random seed')
        parser.add_argument('--samples', type=int, metavar='N', default=25000,
                            help='number of points to sample reconstruction with')
        parser.add_argument('--begin-samples', type=int, metavar='N', default=15000, help='num pts to start with')
        parser.add_argument('--iterations', type=int, metavar='N', default=10000, help='number of iterations to do')
        parser.add_argument('--upsamp', type=int, metavar='N', default=1000, help='upsample each {upsamp} iteration')
        parser.add_argument('--max-faces', type=int, metavar='N', default=10000,
                            help='maximum number of faces to upsample to')
        parser.add_argument('--faces-to-part', nargs='+', default=[8000, 16000, 20000], type=int,
                            help='after how many faces to split')
        # HYPER PARAMETERS - NETWORK
        parser.add_argument('--lr', type=float, metavar='1eN', default=1.1e-4, help='learning rate')
        parser.add_argument('--ang-wt', type=float, metavar='1eN', default=1e-1,
                            help='weight of the cosine loss for normals')
        parser.add_argument('--res-blocks', type=int, metavar='N', default=3, help='')
        parser.add_argument('--leaky-relu', type=float, metavar='1eN', default=0.01, help='slope for leaky relu')
        parser.add_argument('--local-non-uniform', type=float, metavar='1eN', default=0.1,
                            help='weight of local non uniform loss')
        parser.add_argument('--gpu', type=int, metavar='N', default=0, help='gpu to use')
        parser.add_argument('--convs', nargs='+', default=[16, 32, 64, 64, 128], type=int, help='convs to do')
        parser.add_argument('--pools', nargs='+', default=[0.0, 0.0, 0.0, 0.0], type=float,
                            help='percent to pool from original resolution in each layer')
        parser.add_argument('--transfer-data', action='store_true', help='')
        parser.add_argument('--overlap', type=int, default=0, help='overlap for bfs')
        parser.add_argument('--global-step', action='store_true',
                            help='perform the optimization step after all the parts are forwarded (only matters if nparts > 2)')
        parser.add_argument('--manifold-res', default=100000, type=int, help='resolution for manifold upsampling')
        parser.add_argument('--unoriented', action='store_true',
                            help='take the normals loss term without any preferred orientation')
        parser.add_argument('--init-weights', type=float, default=0.002, help='initialize NN with this size')
        #
        parser.add_argument('--export-interval', type=int, metavar='N', default=100, help='export interval')
        parser.add_argument('--beamgap-iterations', type=int, default=0,
                            help='the # iters to which the beamgap loss will be calculated')
        parser.add_argument('--beamgap-modulo', type=int, default=1, help='skip iterations with beamgap loss'
                                                                          '; calc beamgap when:'
                                                                          ' iter % (--beamgap-modulo) == 0')
        parser.add_argument('--manifold-always', action='store_true',
                            help='always run manifold even when the maximum number of faces is reached')

        self.args = parser.parse_args()

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        with open(f'{self.args.save_path}/opt.txt', '+w') as f:
            for k, v in sorted(vars(self.args).items()):
                f.write('%s: %s\n' % (str(k), str(v)))

    def get_num_parts(self, num_faces):
        lookup_num_parts = [1, 2, 4, 8]
        num_parts = lookup_num_parts[np.digitize(num_faces, self.args.faces_to_part, right=True)]
        return num_parts

    def dtype(self):
        return torch.float32

    def get_num_samples(self, cur_iter):
        slope = (self.args.samples - self.args.begin_samples) / int(0.8 * self.args.upsamp)
        return int(slope * min(cur_iter, 0.8 * self.args.upsamp)) + self.args.begin_samples
