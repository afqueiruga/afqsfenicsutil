"""
by B. E. Abali and A. Queiruga
"""

from fenics import *
import numpy as np

#
# Helper methods for extracting submeshes
# and making mappings between them
#
def SubMesh2(mesh,marker,inds):
    mask = MeshFunction("size_t",mesh,marker.dim())
    o = marker.array()
    m = mask.array()
    for i in xrange(len(m)): m[i] = 1 if o[i] in inds else 0
    return SubMesh(mesh, mask,1)

def Mesh2Submesh_FacetMap(mesh, sub):
    vk = sub.data().array('parent_vertex_indices',0)
    vertex_to_facets_name = {}
    for f in facets(mesh):
        vertex_to_facets_name[tuple(sorted(f.entities(0)))] = f.global_index()
    sub.init_global(2)
    local_to_global_facets = np.empty(sub.num_facets(),dtype=np.intc)
    for i,f in enumerate(facets(sub)):
        local_to_global_facets[i] = \
          vertex_to_facets_name[tuple(sorted(vk[f.entities(0)]))]
    return local_to_global_facets

def Mesh2Submesh_CellMap(mesh, sub):
    vk = sub.data().array('parent_vertex_indices',0)
    vertex_to_cells_name = {}
    for f in cells(mesh):
        vertex_to_cells_name[tuple(sorted(f.entities(0)))] = f.global_index()
    sub.init_global(2)
    local_to_global_cells = np.empty(sub.num_cells(),dtype=np.intc)
    for i,f in enumerate(cells(sub)):
        local_to_global_cells[i] = \
          vertex_to_cells_name[tuple(sorted(vk[f.entities(0)]))]
    return local_to_global_cells

def AssignMaterialCoefficients(target_mesh, cells_list, coeffs, mat_marking):
    coeffs_list = np.zeros(max(mat_marking)+1)
    for i,coeff in enumerate(coeffs): coeffs_list[mat_marking[i]] = coeff
    coeff_func = Function(FunctionSpace(target_mesh, 'DG', 0))
    markers = np.asarray(cells_list.array(), dtype=np.int32)
    coeff_func.vector()[:] = np.choose(markers, coeffs_list)
    return coeff_func

#
# Voigt notation utilities
#
def C_IsotropicVoigt(lam, mu):
    return np.array([
        [lam+2.*mu , lam, lam, 0, 0, 0],
        [lam, lam+2.*mu, lam, 0, 0, 0],
        [lam, lam, lam+2.*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]  ])
#
# Building rank-4 tensors from Voight notation
#
def VoigtToTensorRank4(A11=0., A12=0., A13=0., A14=0., A15=0., A16=0., A22=0., A23=0., A24=0., A25=0., A26=0., A33=0., A34=0., A35=0., A36=0., A44=0., A45=0., A46=0., A55=0., A56=0., A66=0.):
	A21, A31, A41, A51, A61 = A12, A13, A14, A15, A16
	A32, A42, A52, A62 = A23, A24, A25, A26
	A43, A53, A63 = A34, A35, A36
	A54, A64 = A45, A46
	A65 = A56
	return as_tensor([ \
	[ \
	[ [A11,A16,A15], [A16,A12,A14], [A15,A14,A13]] , \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] \
	] , [ \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A21,A26,A25], [A26,A22,A24], [A25,A24,A23]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] \
	] , [ \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] , \
	[ [A31,A36,A35], [A36,A32,A34], [A35,A34,A33]] ] \
	])

def VoigtToTensorRank3(A11=0.,A12=0.,A13=0.,A14=0.,A15=0.,A16=0., A21=0.,A22=0.,A23=0.,A24=0.,A25=0.,A26=0., A31=0.,A32=0.,A33=0.,A34=0.,A35=0.,A36=0.):
	return as_tensor([ \
	[ \
	[ A11, A16, A15 ] , \
	[ A16, A12, A14 ] , \
	[ A15, A14, A13 ] \
	] , [ \
	[ A21, A26, A25 ] , \
	[ A26, A22, A24 ] , \
	[ A25, A24, A23 ] \
	] , [ \
	[ A31, A36, A35 ] , \
	[ A36, A32, A34 ] , \
	[ A35, A34, A33 ] ] \
	])

def VoigtToTensorRank2(A11=0.,A12=0.,A13=0., A21=0.,A22=0.,A23=0., A31=0.,A32=0.,A33=0.):
	A21, A31, A32 = A12, A13, A23
	return as_tensor([ \
	[ A11, A12, A13 ] , \
	[ A21, A22, A23 ] , \
	[ A31, A32, A33 ] \
	] )
