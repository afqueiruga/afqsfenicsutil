def write_vtk_f(fname, mesh, nodefunctions=None,cellfunctions=None):
    # S = FunctionSpace(mesh,'P',1)
    # V = FunctionSpace(mesh,'P',1)
    write_vtk(fname, mesh.cells(), mesh.coordinates(),
                  [(k,f.compute_vertex_values().reshape(-1,mesh.num_vertices()).T) for k,f in nodefunctions.iteritems()]
                  )
def write_vtk(fname, elems, X, nodefields=None,edgefields=None):
    celltypekey = {
        1:1, # Pt
        2:3, # Line
        3:5, # Tri
        # 4:9, # Quad
        4:10, # Tet
        8:12} # Hex
    vecformatdict = {
        1:"{0} 0 0\n",
        2:"{0} {1} 0\n",
        3:"{0} {1} {2}\n"
        }
    tenformatdict = {
        1:"{0} 0 0\n0 0 0\n0 0 0\n",
        2:"{0} {1} 0\n{2} {3} 0\n0 0 0\n",
        3:"{0} {1} {2}\n{3} {4} {5}\n{6} {7} {8}\n"
        }
    vecfmt = vecformatdict[X.shape[1]]
    tenfmt = tenformatdict[X.shape[1]]
    fh = open(fname,"w")
    fh.write("# vtk DataFile Version 2.0\nGraph connectivity\nASCII\n")
    fh.write("DATASET UNSTRUCTURED_GRID\n")
    
    fh.write("POINTS {0} double\n".format(X.shape[0]))
    for pt in X:
        fh.write(vecfmt.format(*pt))
    
    fh.write("\nCELLS {0} {1}\n".format(elems.shape[0],elems.shape[0]*(1+elems.shape[1]))) # I assume they're all the same
    for el in elems:
        fh.write("{0} ".format(len(el))+" ".join([str(x) for x in el])+"\n")
    fh.write("\nCELL_TYPES {0}\n".format(elems.shape[0]))
    for el in elems:
        fh.write("{0}\n".format(celltypekey[elems.shape[1]]))

    # Macro to write a data block
    def PUTFIELD(n,f):
        if len(f.shape)==1 or f.shape[1]==1:
            fh.write("SCALARS {0} double\n".format(n))
            fh.write("LOOKUP_TABLE default\n")
            for l in f.ravel():
                fh.write(str(l)+"\n")
        elif f.shape[1]==X.shape[1]:
            fh.write("VECTORS {0} double\n".format(n))
            for l in f:
                fh.write(vecfmt.format(*l))
        else:
            fh.write("TENSORS {0} double\n".format(n))
            for l in f:
                fh.write(tenfmt.format(*l))
    # Dump all of the node fields
    if nodefields:
        fh.write("POINT_DATA {0}\n".format(X.shape[0]))
        for n,f in nodefields:
            PUTFIELD(n,f)
    # Cell fields now
    if edgefields:
        fh.write("CELL_DATA {0}\n".format(elems.shape[0]))
        for n,f in edgefields:
            PUTFIELD(n,f)
            
    fh.close()
