"""
A super-simple mesh-morpher designed for EMSI

Alejandro F Queiruga
UC Berkeley, 2014

"""
import numpy as np
from scipy.spatial import Delaunay
def do_tri_map(fix_ind,nodes,nodes_orig):
    global dela,dela_orig,points,points_orig
    gdim = nodes.shape[1]
    fix_ind = list(set(fix_ind))
    points = nodes[fix_ind,:]
    points_orig = nodes_orig[fix_ind,:]
    # dela = Delaunay(points)
    dela_orig = Delaunay(points_orig)
    nodes_new = np.empty(nodes.shape)
    for ptid in xrange(nodes_orig.shape[0]):
        if ptid in fix_ind:
            continue
        pt = nodes_orig[ptid,:]
        s = dela_orig.find_simplex(pt)
        tri = dela_orig.simplices[s,:]
        b = np.zeros(gdim+1)
        b[0:gdim] = dela_orig.transform[s,:gdim].dot( pt-dela_orig.transform[s,gdim] )
        b[gdim] = 1-b.sum()
        # nodes_new[ptid,:] = (nodes[tri,:].T.dot(b)).T
        for i in range(gdim):
            nodes_new[ptid,i] = points[tri,i].dot(b)
    return nodes_new

def morph_fenics(mesh, nodes, u, other_fix = []):
    X_orig = mesh.coordinates()
    X_defo = X_orig.copy()
    uN = u.compute_vertex_values().reshape(len(u),len(nodes)).T
    X_defo[list(nodes),:] += uN
    # Warp the mesh
    X_new = do_tri_map( nodes + other_fix, X_defo, X_orig)
    mesh.coordinates()[:,:] = X_new
