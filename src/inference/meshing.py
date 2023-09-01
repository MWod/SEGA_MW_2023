### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
import torchio as tio
import skimage.measure as measure
import scipy.ndimage as nd
# import pygalmesh

import vtk

from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteFlyingEdges3D
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtkmodules.vtkIOGeometry import vtkOBJWriter
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

### Internal Imports ###
from paths import paths as p
from inference import inference_sega
from evaluation import evaluation_functions as ev

########################


def sitk2vtk(img, debugOn=False):
    """
    Code from: https://github.com/dave3d/dicom2stl/blob/main/utils/sitk2vtk.py
    """
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))
    vtk_image = vtk.vtkImageData()
    if len(size) == 2:
        size.append(1)
    if len(origin) == 2:
        origin.append(0.0)
    if len(spacing) == 2:
        spacing.append(spacing[0])
    if len(direction) == 4:
        direction = [ direction[0], direction[1], 0.0,
                      direction[2], direction[3], 0.0,
                               0.0,          0.0, 1.0 ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion()<9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    depth_array = numpy_support.numpy_to_vtk(i2.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))
    return vtk_image

def parse_to_obj(image, output_path, show=False, name=None, smoothing_iterations=25, pass_band=0.001, feature_angle=120.0):
    vtk_obj = sitk2vtk(image, debugOn=False)

    discrete = vtkDiscreteMarchingCubes()
    discrete.SetInputData(vtk_obj)

    smoothing_iterations = smoothing_iterations
    pass_band = pass_band
    feature_angle = feature_angle

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    if output_path is not None:
        stlWriter = vtkOBJWriter()
        stlWriter.SetFileName(output_path)
        stlWriter.SetInputConnection(smoother.GetOutputPort())
        stlWriter.Write()

    if show:
        colors = vtkNamedColors()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        lut = vtkLookupTable()
        lut.SetNumberOfColors(2)
        lut.SetTableRange(0, 1)
        lut.SetScaleToLinear()
        lut.Build()
        lut.SetTableValue(0, 0.4, 0.4, 0.4, 1.0)
        lut.SetTableValue(1, 0.5, 0.8, 0.7, 1.0)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)

        ren = vtkRenderer()
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(ren)
        if name is None:
            ren_win.SetWindowName('STL Visualization')
        else:
            ren_win.SetWindowName(name)

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        ren.AddActor(actor)
        ren.SetBackground(colors.GetColor3d('Black'))

        ren_win.Render()
        iren.Start()
        
        
def parse_to_obj2(image, output_path, show=False, name=None, smoothing_iterations=35, pass_band=0.001, feature_angle=120.0):
    vtk_obj = sitk2vtk(image, debugOn=False)

    discrete = vtkDiscreteMarchingCubes()
    # discrete = vtkDiscreteFlyingEdges3D()
    discrete.SetInputData(vtk_obj)
    
    # print("Flying Edges..")

    smoothing_iterations = smoothing_iterations
    pass_band = pass_band
    feature_angle = feature_angle

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    
    filler = vtkFillHolesFilter()
    filler.SetHoleSize(100.0)
    filler.SetInputConnection(smoother.GetOutputPort())
    filler.Update()

    if output_path is not None:
        stlWriter = vtkOBJWriter()
        stlWriter.SetFileName(output_path)
        stlWriter.SetInputConnection(filler.GetOutputPort())
        stlWriter.Write()

    if show:
        colors = vtkNamedColors()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(filler.GetOutputPort())
        lut = vtkLookupTable()
        lut.SetNumberOfColors(2)
        lut.SetTableRange(0, 1)
        lut.SetScaleToLinear()
        lut.Build()
        lut.SetTableValue(0, 0.4, 0.4, 0.4, 1.0)
        lut.SetTableValue(1, 0.5, 0.8, 0.7, 1.0)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)

        ren = vtkRenderer()
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(ren)
        if name is None:
            ren_win.SetWindowName('STL Visualization')
        else:
            ren_win.SetWindowName(name)

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        ren.AddActor(actor)
        ren.SetBackground(colors.GetColor3d('Black'))

        ren_win.Render()
        iren.Start()