# basic libraries
import numpy as np

# computational geometry libraries
from compas.datastructures import Mesh as CompasMesh
from compas_plotters.meshplotter import MeshPlotter
from numpy.lib.function_base import insert

# libraries for connection to grasshopper (speckle)
from specklepy.objects.geometry import Mesh as SpeckleMesh
from specklepy.objects.geometry import Base, Box, Interval, Vector, Point, Plane
import pyvista as pv

def speckle_to_vertices_and_faces(speckle_mesh, return_array=False):
    # unwrap the mesh data
    base_dict = speckle_mesh.dict()
    if "@data" in base_dict.keys():
        speckle_mesh_dict = base_dict["@data"][0][0]
    else:
        speckle_mesh_dict = base_dict


    V = np.array(speckle_mesh_dict["vertices"]).reshape((-1, 3)).tolist()
    # F = np.array(speckle_mesh_dict["faces"]).reshape((-1, 4))[:, 1:]
    # extract faces
    F = []
    face = []
    i = 0 
    for v in speckle_mesh_dict["faces"]:
        if len(face) == 0 and i == 0:
            i = v + 3
            continue
        else:
            face.append(v)
            i -= 1
            if i == 0:
                F.append(face)
                face = []
    if return_array:
        Va = np.array(V)
        F_quad = []
        for f in F:
            if len(f) == 3:
                f.append(f[0])
                F_quad.append(f)

        Fa = np.array(F_quad)
        return (Va, Fa)

    else:
        return (V, F)


def vertices_and_faces_to_speckle(V, F):
    V_arr = np.array(V)
    F_arr = np.hstack([[len(f) - 3 ] + f for f in F])
    # (V_arr, F_arr) = (np.array(V), np.array(F))
    # F_arr = np.pad(F_arr, ((0,0),(1,0)), mode='constant', constant_values=0)

    base_plane = Plane(
        origin=Point(x=0, y=0, z=0), 
        xdir= Vector(x=1, y=0, z=0), 
        ydir= Vector(x=0, y=1, z=0),
        normal= Vector(x=0, y=0, z=1))
    
    bounding_box = Box(
        basePlane = base_plane,
        area = 1,
        volume = 1,
        xSize = Interval(start=V_arr[:,0].min(), end=V_arr[:,0].max()), 
        ySize = Interval(start=V_arr[:,1].min(), end=V_arr[:,1].max()), 
        zSize = Interval(start=V_arr[:,2].min(), end=V_arr[:,2].max()))

    speckle_mesh = SpeckleMesh(
        vertices = V_arr.flatten().tolist(), 
        faces = F_arr.flatten().tolist(),
        # colors = [],
        # area = 0,
        # volume = 1,
        bbox = bounding_box
        )

    speckle_wrapped_mesh = Base(
        data = [[speckle_mesh]]
    )
    return speckle_wrapped_mesh
    
def vertices_and_faces_to_pyvista(V, F):
    F_pv = np.hstack([[len(f)] + f for f in F])
    pv_mesh = pv.PolyData(V, F_pv)

    return pv_mesh

def compas_to_pyvista(compas_mesh):
    (V, F) = compas_mesh.to_vertices_and_faces()
    pv_mesh = vertices_and_faces_to_pyvista(V, F)
    return pv_mesh

def speckle_to_pyvista(speckle_mesh):
    (V, F) = speckle_to_vertices_and_faces(speckle_mesh, return_array=False)
    pv_mesh = vertices_and_faces_to_pyvista(V, F)
    return pv_mesh

def speckle_to_compas(speckle_mesh):
    (V, F) = speckle_to_vertices_and_faces(speckle_mesh)
    compas_mesh = CompasMesh.from_vertices_and_faces(V,F)
    return compas_mesh

def compas_to_speckle(compas_mesh):
    (V, F) = compas_mesh.to_vertices_and_faces()
    speckle_mesh = vertices_and_faces_to_speckle(V, F)
    return speckle_mesh
