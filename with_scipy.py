"""
These are utilities for interfacing FEniCS with Scipy.
This is how I look at sparsities or interface with LIS (see pylis by afq).
"""

def assemble_as_scipy(form):
    K = PETScMatrix()
    assemble(form, tensor=K)
    ki,kj,kv = K.mat().getValuesCSR()
    import scipy
    import scipy.sparse
    Ksp = scipy.sparse.csr_matrix((kv, kj, ki))
    return Ksp


def look_at_a_form(form,fname="foo.png",xlim=None,ylim=None):
    from matplotlib import pylab as plt
    Ksp = assemble_as_scipy(form)
    plt.spy(Ksp)
    if xlim != None: plt.xlim(*xlim)
    if ylim != None: plt.ylim(*ylim)
    plt.savefig(fname,dpi=150)
