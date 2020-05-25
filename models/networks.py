import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
from models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
from typing import List


###############################################################################
# Helper Functions
###############################################################################


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def populate_e(meshes, verts=None):
    mesh = meshes[0]
    if verts is None:
        verts = torch.rand(len(meshes), mesh.vs.shape[0], 3).to(mesh.vs.device)
    x = verts[:, mesh.edges, :]
    return x.view(len(meshes), mesh.edges_count, -1).permute(0, 2, 1).type(torch.float32)


def build_v(x, meshes):
    # mesh.edges[mesh.ve[2], mesh.vei[2]]
    mesh = meshes[0]  # b/c all meshes in batch are same
    x = x.reshape(len(meshes), 2, 3, -1)
    vs_to_sum = torch.zeros([len(meshes), len(mesh.vs_in), mesh.max_nvs, 3], dtype=x.dtype, device=x.device)
    x = x[:, mesh.vei, :, mesh.ve_in].transpose(0, 1)
    vs_to_sum[:, mesh.nvsi, mesh.nvsin, :] = x
    vs_sum = torch.sum(vs_to_sum, dim=2)
    nvs = mesh.nvs
    vs = vs_sum / nvs[None, :, None]
    return vs


def local_nonuniform_penalty(mesh):
    # non-uniform penalty
    area = mesh_area(mesh)
    diff = area[mesh.gfmm][:, 0:1] - area[mesh.gfmm][:, 1:]
    penalty = torch.norm(diff, dim=1, p=1)
    loss = penalty.sum() / penalty.numel()
    return loss

def mesh_area(mesh):
    vs = mesh.vs
    faces = mesh.faces
    v1 = vs[faces[:, 1]] - vs[faces[:, 0]]
    v2 = vs[faces[:, 2]] - vs[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area


##############################################################################
# Classes For Networks
##############################################################################

#TODO : skip pool and unpool if all zeros..
class PriorNet(nn.Module):
    """
    network for
    """
    def __init__(self, n_edges, in_ch=6, convs=[32, 64], pool=[], res_blocks=0,
                 init_verts=None, transfer_data=False, leaky=0, init_weights_size=0.002):
        super(PriorNet, self).__init__()
        # check that the number of pools and convs match such that there is a pool between each conv
        down_convs = [in_ch] + convs
        up_convs = convs[::-1] + [in_ch]
        pool_res = [n_edges] + pool
        self.encoder_decoder = MeshEncoderDecoder(pools=pool_res, down_convs=down_convs,
                                                  up_convs=up_convs, blocks=res_blocks,
                                                  transfer_data=transfer_data, leaky=leaky)
        self.last_conv = MeshConv(6, 6)
        init_weights(self, 'normal', init_weights_size)
        eps = 1e-8
        self.last_conv.conv.weight.data.uniform_(-1*eps, eps)
        self.last_conv.conv.bias.data.uniform_(-1*eps, eps)
        self.init_verts = init_verts

    def forward(self, x, meshes):
        meshes_new = [i.deep_copy() for i in meshes]
        x, _ = self.encoder_decoder(x, meshes_new)
        x = x.squeeze(-1)
        x = self.last_conv(x, meshes_new).squeeze(-1)
        est_verts = build_v(x.unsqueeze(0), meshes)
        # assert not torch.isnan(est_verts).any()
        return est_verts.float() + self.init_verts.expand_as(est_verts).to(est_verts.device)


class PartNet(PriorNet):
    def __init__(self, init_part_mesh, in_ch=6, convs=[32, 64], pool=[], res_blocks=0,
                 init_verts=None, transfer_data=False, leaky=0,
                 init_weights_size=0.002):
        temp = torch.linspace(len(convs), 1, len(convs)).long().tolist()
        super().__init__(temp[0], in_ch=in_ch, convs=convs, pool=temp[1:], res_blocks=res_blocks,
                 init_verts=init_verts, transfer_data=transfer_data, leaky=leaky, init_weights_size=init_weights_size)
        self.mesh_pools = []
        self.mesh_unpools = []
        self.factor_pools = pool
        for i in self.modules():
            if isinstance(i, MeshPool):
                self.mesh_pools.append(i)
            if isinstance(i, MeshUnpool):
                self.mesh_unpools.append(i)
        self.mesh_pools = sorted(self.mesh_pools, key=lambda x: x._MeshPool__out_target, reverse=True)
        self.mesh_unpools = sorted(self.mesh_unpools, key=lambda x: x.unroll_target, reverse=False)
        self.init_part_verts = nn.ParameterList([torch.nn.Parameter(i) for i in init_part_mesh.init_verts])
        for i in self.init_part_verts:
            i.requires_grad = False

    def __set_pools(self, n_edges: int, new_pools: List[int]):
        for i, l in enumerate(self.mesh_pools):
            l._MeshPool__out_target = new_pools[i]
        new_pools = [n_edges] + new_pools
        new_pools = new_pools[:-1]
        new_pools.reverse()
        for i, l in enumerate(self.mesh_unpools):
            l.unroll_target = new_pools[i]

    def forward(self, x, partmesh):
        """
        forward PartNet
        :param x: BXfXn_edges
        :param partmesh:
        :return:
        """

        for i, p in enumerate(partmesh):
            n_edges = p.edges_count
            self.init_verts = self.init_part_verts[i]
            temp_pools = [int(n_edges - i) for i in self.make3(PartNet.array_times(n_edges, self.factor_pools))]
            self.__set_pools(n_edges, temp_pools)
            relevant_edges = x[:, :, partmesh.sub_mesh_edge_index[i]]
            results = super().forward(relevant_edges, [p])
            yield results

    @staticmethod
    def array_times(num: int, iterable):
        return [i * num for i in iterable]

    @staticmethod
    def make3(array):
        diff = [i % 3 for i in array]
        return [array[i] - diff[i] for i in range(len(array))]


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True, leaky=0):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks, leaky=leaky)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data, leaky=leaky)
        self.bn = nn.InstanceNorm2d(up_convs[-1])

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        # mid_mesh = meshes[0].deep_copy()
        fe = self.decoder((fe, meshes), before_pool)
        fe = self.bn(fe.unsqueeze(-1))
        return fe, None

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, leaky=0):
        super(DownConv, self).__init__()
        self.leaky = leaky
        self.bn = []
        self.pool = None
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def forward(self, x):
        fe, meshes = x[0], x[1]
        x1 = self.conv1(fe, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        if self.bn:
            x1 = self.bn[0](x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k=1):
        super(ConvBlock, self).__init__()
        self.lst = [MeshConv(in_feat, out_feat)]
        for i in range(k - 1):
            self.lst.append(MeshConv(out_feat, out_feat))
        self.lst = nn.ModuleList(self.lst)

    def forward(self, input, meshes):
        for c in self.lst:
            input = c(input, meshes)
        return input



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True, leaky=0):
        super(UpConv, self).__init__()
        self.leaky = leaky
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = ConvBlock(in_channels, out_channels)
        if transfer_data:
            self.conv1 = ConvBlock(2 * out_channels, out_channels)
        else:
            self.conv1 = ConvBlock(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def forward(self, x, from_down=None):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        if self.bn:
            x1 = self.bn[0](x1)

        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, blocks=0, leaky=0):
        super(MeshEncoder, self).__init__()
        self.leaky = leaky
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool, leaky=leaky))
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        return fe, encoder_outs


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, leaky=0):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data, leaky=leaky))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False, leaky=leaky)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def face_areas_normals(faces, vs):
    face_normals = torch.cross(vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
                               vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5*face_areas
    return face_areas, face_normals

def sample_surface(faces, vs, count):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices
    faces: triangle faces (torch.long)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    normals: (count, 3) corresponding face normals for points
    """
    bsize, nvs, _ = vs.shape
    weights, normal = face_areas_normals(faces, vs)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples, normals


def get_scheduler(iters, optim):
    lr_lambda = lambda x : 1 - min((0.1*x / float(iters), 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return scheduler

def init_net(mesh, part_mesh, device, opts):
    # initialize network, weights, and random input tensor
    init_verts = mesh.vs.clone().detach()
    net = PartNet(init_part_mesh=part_mesh, convs=opts.convs,
                  pool=opts.pools, res_blocks=opts.res_blocks,
                  init_verts=init_verts, transfer_data=opts.transfer_data,
                  leaky=opts.leaky_relu, init_weights_size=opts.init_weights).to(device)

    optimizer = optim.Adam(net.parameters(), lr=opts.lr)
    scheduler = get_scheduler(opts.iterations, optimizer)
    rand_verts = populate_e([mesh])
    return net, optimizer, rand_verts, scheduler