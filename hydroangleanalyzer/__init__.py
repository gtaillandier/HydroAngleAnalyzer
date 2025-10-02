#from .processing import parallel_process_frames
from .parser.base_parser import BaseParser
from .parser.parser_dump import DumpParser, DumpParse_wall, Dump_WaterMoleculeFinder
from .parser.parser_ase import Ase_Parser , Ase_wallParser, ASE_WaterMoleculeFinder
from .parser.parser_xyz import XYZ_Parser, XYZ_WaterOxygenParser, XYZ_wallParser
from .sliced_method.angle_fitting import ContactAnglePredictor
from .sliced_method.surface_defined import SurfaceDefinition
#from .processing import GPUFrameProcessor, FrameProcessor, BatchFrameProcessor
from .binning_method.contact_angle_analyzer import ContactAngleAnalyzer
from .binning_method.surface_definition import HyperbolicTangentModel
from .angles_frames_analysis.graphs_circle_surfaces import SurfacePlotter
from .angles_frames_analysis.angles_analysis import  frames_angle_PostProcessor
from .new_processing import HighPerformanceFrameProcessor
from .multi_processing import BatchFrameProcessor