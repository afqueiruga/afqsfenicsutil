"""

This is a little library of helper functions for use with FEniCS 
by Alejandro F. Queiruga, with major contributions by B. Emek Abali.

"""


from .various import *
#from my_restriction_map import restriction_map
from .write_vtk import write_vtk_f
from .with_scipy import assemble_as_scipy, look_at_a_form
from .mesh_morph import *
