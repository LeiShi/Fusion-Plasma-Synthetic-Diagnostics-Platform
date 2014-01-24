"""Script for VisIt-Python interface to create time slides for reflectometry movie

Before running this script, one needs to run VisIt GUI, load a test dataset, plot all the data, adjust and save the following parameters:
    1. 3D View: drag with mouse left button, find the best view angle. 
    2. Plots' attributes: double click on the plot name in the plot list. choose the best color table and opacity
    3. Annotation attributes: click the "Control" menu and choose "Annotation". Adjust the parameters in "General","3D" and "Objects".
    
    run "Controls" -> "Launch CLI", and in the poped python command line, run the Save_Visit_Attribute.py script, by typing in:
        execfile('Save_Visit_Attribute.py')
    
    double check if the following files appeared:
        ../attrs/anno_attr.xml
        ../attrs/legend_attr.xml
        ../attrs/plasma_attr.xml
        ../attrs/wave_attr.xml
        
If all these saved attributes files are succesfully created, now one can proceed. 
Of course, all the data set files (*.vtk) for all time steps need to be ready.     
"""

import os
import visit as vi

view_attr_file = '../attrs/view_attr.xml'
annotation_attr_file = '../attrs/anno_attr.xml'
plasma_attr_file = '../attrs/plasma_attr.xml'
wave_attr_file = '../attrs/wave_attr.xml'


movie_file_name = 'FWR_760_movie_'
continued = 'B'


time_start = 275+267
time_end = 632
time_step = 1

#vtk_path = 'test_data/'  #path on Windows
vtk_path = '/p/gkp/lshi/FWR_Movie/vtk_small_files/' # path on Cluster
plasma_file_head = 'fluctuation'
paraxial_file_head = 'para_out'
fullwave_file_head = 'fullw_out'
 

def make_slide(time):
    plasma_file = vtk_path + plasma_file_head + str(time) +'.vtk'
    paraxial_file = vtk_path + paraxial_file_head + str(time) + '.vtk'
    fullwave_file = vtk_path + fullwave_file_head + str(time) + '.vtk'
    
    print 'opening plamsa file:' + plasma_file
    sts = vi.OpenDatabase(plasma_file,0,'VTK')
    if (sts != 1):
        print 'open file failed! error code:' + str(sts)
        return
    
    sts = vi.AddPlot('Pseudocolor','cutoff')
    vi.DrawPlots()
    po = vi.GetPlotOptions()
    vi.LoadAttribute(plasma_attr_file,po)
    vi.SetPlotOptions(po)
    
    print 'opening paraxial file:' + paraxial_file
    sts = vi.OpenDatabase(paraxial_file,0,'VTK')
    if (sts != 1):
        print 'open file failed! error code:' + str(sts)
        return
    
    sts = vi.AddPlot('Pseudocolor','Er_para')
    vi.DrawPlots()
    po = vi.GetPlotOptions()
    vi.LoadAttribute(wave_attr_file,po)
    vi.SetPlotOptions(po)
    
    print 'opening fullwave file:' + fullwave_file
    sts = vi.OpenDatabase(fullwave_file,0,'VTK')
    if (sts != 1):
        print 'open file failed! error code:' + str(sts)
        return
    
    sts = vi.AddPlot('Pseudocolor','Er_fullw')
    vi.DrawPlots()
    po = vi.GetPlotOptions()
    vi.LoadAttribute(wave_attr_file,po)
    vi.SetPlotOptions(po)
    
    view_attr = vi.GetView3D()
    vi.LoadAttribute(view_attr_file,view_attr)
    vi.SetView3D(view_attr)
    
    anno_attr = vi.GetAnnotationAttributes()
    vi.LoadAttribute(annotation_attr_file,anno_attr)
    vi.SetAnnotationAttributes(anno_attr)
    
    anno_names = vi.GetAnnotationObjectNames()
    for name in anno_names:
        legend = vi.GetAnnotationObject(name)
        legend.numberFormat = "%# -9.2g"
        legend.drawMinMax = 0
        legend.controlTicks = 1
        legend.numTicks = 3
         
        
    vi.SaveWindow()
def clear(time):
    plasma_file = vtk_path + plasma_file_head + str(time) +'.vtk'
    paraxial_file = vtk_path + paraxial_file_head + str(time) + '.vtk'
    fullwave_file = vtk_path + fullwave_file_head + str(time) + '.vtk'
    
    vi.DeleteAllPlots()
    vi.CloseDatabase(plasma_file)
    vi.CloseDatabase(paraxial_file)
    vi.CloseDatabase(fullwave_file)

windowA = vi.SaveWindowAttributes()
windowA.format = windowA.JPEG
windowA.progressive = 1
windowA.width = 840
windowA.height = 1080
windowA.fileName = movie_file_name + str(continued)+'_'		# VisIt will append image number to base name
windowA.outputDirectory = "/p/gkp/lshi/FWR_Movie/Time_Slides/"
windowA.outputToCurrentDirectory = 0
vi.SetSaveWindowAttributes(windowA)
   
for time in range(time_start,time_end+1,time_step):    
    make_slide(time)    
    clear(time)
    
        
        
