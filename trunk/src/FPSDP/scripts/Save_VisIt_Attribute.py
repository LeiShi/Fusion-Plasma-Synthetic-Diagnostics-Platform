"""Script for saving the changed attribute files
"""

import visit as vi

view_attr_file = 'view_attr.xml'
annotation_attr_file = 'anno_attr.xml'
plasma_attr_file = 'plasma_attr.xml'
wave_attr_file = 'wave_attr.xml'

vi.SaveAttribute(view_attr_file, vi.GetView3D())
vi.SaveAttribute(annotation_attr_file, vi.GetAnnotationAttributes())
vi.SetActivePlots(0)
vi.SaveAttribute(plasma_attr_file, vi.GetPlotOptions())
vi.SetActivePlots(1)
vi.SaveAttribute(wave_attr_file,vi.GetPlotOptions())

