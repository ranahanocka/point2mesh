import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root) 
import vtk


import torch
from models.layers.mesh import Mesh, vtkMesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss

#Initilaize Torch
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

#Initilaize Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(1000, 1000)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())


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

def MakeInputData(pointcloud):
    xyz = []
    normal = []

    nPoints = pointcloud.GetNumberOfPoints()

    for pid in range(nPoints):
        xyz.append(pointcloud.GetPoint(pid))
        normal.append(pointcloud.GetPointData().GetNormals().GetTuple(pid))

    return np.array(xyz), np.array(normal)


if __name__ == "__main__":
    
    

    targetpoly = None
    initMesh = None


    
    
        

    initReader = vtk.vtkOBJReader()
    initReader.SetFileName(os.path.join(root, "data", "triceratops_initmesh.obj"))
    initReader.Update()

    targetReader = vtk.vtkPLYReader()
    targetReader.SetFileName(os.path.join(root, "data", "triceratops.ply"))
    targetReader.Update()

    

    

    targetpoly = targetReader.GetOutput()
    initMesh = initReader.GetOutput()


    #Convert to Mesh 
    mesh = vtkMesh(initMesh, device=device, hold_history=True)

    #does ply has novrmla?
    input_xyz, input_normals = MakeInputData(targetpoly)

    print(input_xyz.shape, input_normals.shape)
    

    input_xyz /= mesh.scale
    input_xyz += mesh.translations[None, :]
    input_xyz = torch.Tensor(input_xyz).type(torch.float32).to(device)[None, :, :]
    input_normals = torch.Tensor(input_normals).type(torch.float32).to(device)[None, :, :]


    print(input_xyz.shape, input_normals.shape)

    exit()




    
    targetActor = MakePointCloudActor(targetpoly)
    initActor = MakeInitMeshActor(initMesh)

    

    ren.AddActor(targetActor)
    ren.AddActor(initActor)
    renWin.Render()

    iren.Initialize()
    iren.Start()
