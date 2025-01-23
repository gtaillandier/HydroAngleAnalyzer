#from .processing import parallel_process_frames
from .parser.parser_lammps import DumpParser
from .sliced_method.angle_fitting import ContactAnglePredictor
from .sliced_method.surface_defined import SurfaceDefinition
from .processing import GPUFrameProcessor, FrameProcessor