import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root) 
import vtk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor



import torch
from models.layers.mesh import Mesh, vtkMesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss
from options import Options
import time
import os

app = QApplication([])

options = Options()
opts = options.args

torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))






iren = QVTKRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

#Initilaize Renderer
ren = vtk.vtkRenderer()
renWin = iren.GetRenderWindow()
# renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)



def UpdateGT(polydata, vs):
    for idx, pos in enumerate(vs):
        polydata.GetPoints().SetPoint(idx, pos[0], pos[1], pos[2])

    polydata.GetPoints().Modified()


def MakeInitMeshActor(polydata):

    mapper = vtk.vtkPolyDataMapper()    
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.1, .4, .9)
    

    return actor

def MakePointCloudActor(polydata):

    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetRadius(.05)    
    mapper.SetInputData(polydata)


    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.9, .4, .1)

    return actor


class Worker(QThread):

    backwarded = pyqtSignal(object)

    def __init__(self, initPoly, targetPoly):
        super().__init__()

        self.initPoly = initPoly
        self.targetPoly = targetPoly

    
    def MakeInputData(self, pointcloud):
        xyz = []
        normal = []

        nPoints = pointcloud.GetNumberOfPoints()

        for pid in range(nPoints):
            xyz.append(pointcloud.GetPoint(pid))
            normal.append(pointcloud.GetPointData().GetNormals().GetTuple(pid))

        return np.array(xyz), np.array(normal)


    def run(self):
        # mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)
        mesh = vtkMesh(self.initPoly, device=device, hold_history=True)
        


        # input point cloud
        input_xyz, input_normals = self.MakeInputData(self.targetPoly)
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
                xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, x_normals=recon_normals, y_normals=input_normals,unoriented=opts.unoriented)

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
            
            # mesh.export(os.path.join("./", f'recon_iter_{i}.obj'))
            self.backwarded.emit(mesh.vs)


            # if (i > 0 and (i + 1) % opts.upsamp == 0):
            #     mesh = part_mesh.main_mesh
            #     num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))

            #     if num_faces > len(mesh.faces) or opts.manifold_always:
            #         # up-sample mesh
            #         mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh,
            #                                     num_faces=min(num_faces, opts.max_faces),
            #                                     res=opts.manifold_res, simplify=True)

            #         part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
            #         print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
            #         net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
            #         if i < opts.beamgap_iterations:
            #             print('beamgap updated')
            #             beamgap_loss.update_pm(part_mesh, input_xyz)

        with torch.no_grad():
            mesh.export(os.path.join(opts.save_path, 'last_recon.obj'))


if __name__ == "__main__":

    initReader = vtk.vtkOBJReader()
    initReader.SetFileName(os.path.join(root, "data", opts.initial_mesh))
    initReader.Update()
    initMesh = initReader.GetOutput()

    cleanPoly = vtk.vtkCleanPolyData()
    cleanPoly.SetInputData(initMesh)
    cleanPoly.Update()
    initMesh = cleanPoly.GetOutput()

    initMesh.GetPointData().RemoveArray("Normals")

    targetReader = vtk.vtkPLYReader()
    targetReader.SetFileName(os.path.join(root, "data", opts.input_pc))
    targetReader.Update()


    targetpoly = targetReader.GetOutput()
    
    

    initMeshActor =  MakeInitMeshActor(initMesh)
    pcActor = MakePointCloudActor(targetpoly)

    ren.AddActor(initMeshActor)
    # ren.AddActor(pcActor)

    renWin.Render()
    

    #Run Thread

    def test(vs):
        UpdateGT(initMesh, vs)
        ren.ResetCamera()
        renWin.Render()

    trainingWorker = Worker(initMesh, targetpoly)
    trainingWorker.backwarded.connect(test)
    trainingWorker.start()

    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.centralWidget().setLayout(QVBoxLayout())
    window.centralWidget().layout().addWidget(iren)
    window.show()
    sys.exit(app.exec_())

    exit()