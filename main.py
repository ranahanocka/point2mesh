import torch
from models.layers.mesh import Mesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss
from options import Options
import time
import os

options = Options()
opts = options.args

torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

# initial mesh
mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)

# input point cloud
input_xyz, input_normals = utils.read_pts(opts.input_pc)
# normalize point cloud based on initial mesh
input_xyz /= mesh.scale
input_xyz += mesh.translations[None, :]
input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
print(f'number of parts {part_mesh.n_submeshes}')
net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)

beamgap_loss = BeamGapLoss(device)

if opts.beamgap_iterations > 0:
    print('beamgap on')
    beamgap_loss.update_pm(part_mesh, torch.cat([input_xyz, input_normals], dim=-1))

for i in range(opts.iterations):
    num_samples = options.get_num_samples(i % opts.upsamp)
    if opts.global_step:
        optimizer.zero_grad()
    start_time = time.time()
    for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
        if not opts.global_step:
            optimizer.zero_grad()
        part_mesh.update_verts(est_verts[0], part_i)
        num_samples = options.get_num_samples(i % opts.upsamp)
        recon_xyz, recon_normals = sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0), num_samples)
        # calc chamfer loss w/ normals
        recon_xyz, recon_normals = recon_xyz.type(options.dtype()), recon_normals.type(options.dtype())
        xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, x_normals=recon_normals, y_normals=input_normals,
                                              unoriented=opts.unoriented)

        if (i < opts.beamgap_iterations) and (i % opts.beamgap_modulo == 0):
            loss = beamgap_loss(part_mesh, part_i)
        else:
            loss = (xyz_chamfer_loss + (opts.ang_wt * normals_chamfer_loss))
        if opts.local_non_uniform > 0:
            loss += opts.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()
        loss.backward()
        if not opts.global_step:
            optimizer.step()
            scheduler.step()
        part_mesh.main_mesh.vs.detach_()
    if opts.global_step:
        optimizer.step()
        scheduler.step()
    end_time = time.time()

    if i % 1 == 0:
        print(f'{os.path.basename(opts.input_pc)}; iter: {i} out of: {opts.iterations}; loss: {loss.item():.4f};'
              f' sample count: {num_samples}; time: {end_time - start_time:.2f}')
    if i % opts.export_interval == 0 and i > 0:
        print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            part_mesh.export(os.path.join(opts.save_path, f'recon_iter_{i}.obj'))

    if (i > 0 and (i + 1) % opts.upsamp == 0):
        mesh = part_mesh.main_mesh
        num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))

        if num_faces > len(mesh.faces) or opts.manifold_always:
            # up-sample mesh
            mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh,
                                           num_faces=min(num_faces, opts.max_faces),
                                           res=opts.manifold_res, simplify=True)

            part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
            print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
            net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
            if i < opts.beamgap_iterations:
                print('beamgap updated')
                beamgap_loss.update_pm(part_mesh, input_xyz)

with torch.no_grad():
    mesh.export(os.path.join(opts.save_path, 'last_recon.obj'))