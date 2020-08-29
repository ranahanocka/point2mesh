import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root) 
import vtk

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
    mapper.SetRadius(.01)    
    mapper.SetInputData(polydata)
    

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.9, .4, .1)

    return actor

if __name__ == "__main__":
    
    
    data = sorted(os.listdir(os.path.join(root, "data")))


    targetpoly = None
    initMesh = None


    for filename in data:
        ext = filename[-3:]
        
        if ext == "obj":
            reader = vtk.vtkOBJReader()
        elif ext == "ply":
            reader = vtk.vtkPLYReader()
    
        reader.SetFileName(os.path.join(root, "data", filename))
        reader.Update()

        polydata = reader.GetOutput()

        if filename == "g.ply" : targetpoly = polydata
        if filename == "g.obj" : initMesh = polydata




    
    targetActor = MakePointCloudActor(targetpoly)
    initActor = MakeInitMeshActor(initMesh)

    

    ren.AddActor(targetActor)
    ren.AddActor(initActor)
    renWin.Render()

    iren.Initialize()
    iren.Start()
