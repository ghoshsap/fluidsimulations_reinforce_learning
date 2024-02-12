from mpi4py import MPI
import socket, pickle

from fenics import *
from dolfin import *
import numpy as np
import random 
from scipy.interpolate import griddata

comm = MPI.comm_world
rank = MPI.rank(comm)

set_log_level(50)

# Test to find appropriate solver
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


# Define parameters of the simulations

rho = 1.6 # Density
a_2 = (rho-1)
a_4 = -(1 + rho)/rho**2 # a_2, a_4 from the bulk free energy
lam = 1.0 # Flow alignment parameter
Smax = np.sqrt(2.0) # Maximum of scalar order parameter (Exceeds sometimes?)
mu = 0.01

seed = 55

nx, ny = 25, 25
length, width = 50, 50

N = 42
mesh = RectangleMesh(Point(0, 0), Point(length, width), nx, ny)


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], width)) or
                          (near(x[0], length) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], length) and near(x[1], width):
            y[0] = x[0] - length
            y[1] = x[1] - width
        elif near(x[0], length):
            y[0] = x[0] - length
            y[1] = x[1]
        else:   
            y[0] = x[0]
            y[1] = x[1] - width


class Subdomain(SubDomain):

    def __init__(self, H, width):
        SubDomain.__init__(self)
        self.H = H
        self.width = width


    def inside(self, x, on_boundary):
        return x[1] < self.width/2 + self.H/2 and x[1] > self.width/2 - self.H/2

bcp = PeriodicBoundary()


tol = 1E-14


FQ = VectorFunctionSpace(mesh, 'P', 1, constrained_domain = bcp)

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH, constrained_domain = bcp)


# 2D Mixed Function Space to store [Qxx, Qxy]:
F = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([F,F]), constrained_domain = bcp)
FS = FunctionSpace(mesh, F, constrained_domain = bcp)
# Define test function

r = TestFunction(MFS)
(rx, ry) = split(r)

y_new = Function(MFS) # Solution at the next time step
y_old = Function(MFS) # Solution at the current time step

(Qxx_new, Qxy_new) = split(y_new)
(Qxx_old, Qxy_old) = split(y_old)


Q_new = as_tensor([[Qxx_new, Qxy_new], [Qxy_new, -Qxx_new]])
Q_old = as_tensor([[Qxx_old, Qxy_old], [Qxy_old, -Qxx_old]])

R = as_tensor([[rx, ry], [ry, -rx]])

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

FA = FunctionSpace(mesh, "Lagrange", 1)

alpha = Function(FA)


class InitialConditions_U(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

U_init = InitialConditions_U(degree = 1)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define anti-symmetric gradient
def omega(u):
    return 0.5 * (nabla_grad(u) - nabla_grad(u).T)

# Define active force (with alpha being space-dependent)
Qxxx, Qxxy = grad(alpha*Qxx_new)
Qxyx, Qxyy = grad(alpha*Qxy_new)

divQ = as_vector([Qxxx+Qxyy, Qxyx-Qxxy]) # Divergence of the Q-tensor, computed manually

# Initialize
U = Function(W) # U --> (u0, p)
u0 = split(U)[0]

# Initial condition for state variable

NOISE_STRENGTH = 0.01
seed = 3002
class InitialConditionsQ(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        nx = 1.0 / sqrt(2.0)
        ny = 1.0 / sqrt(2.0)
        values[0] = (nx**2 - 0.5) + 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxx
        values[1] = (nx*ny) + 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxy

    def value_shape(self):
        return (2,)


y_init = InitialConditionsQ(degree = 1)


dt = 0.1
time_count = 0
H = 10
subdomain = Subdomain(H, width)
markers = MeshFunction("size_t", mesh, mesh.topology().dim())
markers.set_all(0)
subdomain.mark(markers, 1)

dx_subdomain = Measure("dx", domain=mesh, subdomain_data=markers)
Res_Q = inner((Q_new - Q_old), R) / dt * dx \
        + inner(dot(u0, nabla_grad(Q_new)), R) * dx \
        + inner((dot(omega(u0), Q_new) - dot(Q_new, omega(u0))), R) * dx \
        + (-lam * inner(epsilon(u0), R) * dx) \
        + (-a_2 * inner(Q_new, R)) * dx \
        + (- 2 * a_4 * (Qxx_old**2 + Qxy_old**2) * inner(Q_new, R)) * dx \
        + 2 * inner(grad(Q_new), grad(R)) * dx

def interpolate_up(original_grid):

    original_grid = np.asarray(original_grid)
    x_original, y_original = np.meshgrid(np.linspace(0, 1, ny + 1), np.linspace(0, 1, nx + 1))
    x_new, y_new = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    values_original = original_grid.flatten()
    values_new = griddata((x_original.flatten(), y_original.flatten()), values_original, (x_new, y_new), method='linear')
    interpolated_grid = values_new.reshape((N, N))

    return interpolated_grid


dof_coordinates = FA.tabulate_dof_coordinates()
comm.barrier()
x_coordinates = [ dof_coordinates[row][0] for row in range(len(dof_coordinates))]
y_coordinates = [ dof_coordinates[row][1] for row in range(len(dof_coordinates))]
dof_x = np.hstack(mesh.mpi_comm().allgather(x_coordinates))
dof_y = np.hstack(mesh.mpi_comm().allgather(y_coordinates))

comm.barrier()
y_new.interpolate(y_init)
y_old.interpolate(y_init)
U.interpolate(U_init)


def reset(arr):

    global time_count
    global y_old
    global y_new
    global U
    global alpha
    global Qxx_old, Qxy_old
    global Q_new
    global Q_old
    global alpha
    global f, divQ
    global time_count

    if rank == 0:
        print("resetting environment", flush=True)

    y_init = InitialConditionsQ(degree = 1)

    y_new.interpolate(y_init)
    y_old.interpolate(y_init)
    U.interpolate(U_init)
    
    time_count = 0



def time_step(arr):

    global time_count
    global y_old
    global y_new
    global U
    global alpha
    global Qxx_old, Qxy_old
    global Q_new
    global Q_old
    global alpha
    global f, divQ
    global time_count

    arr = np.asarray(arr)
    arr = arr.reshape(nx + 1, ny + 1)
    vec = alpha.vector()
    values = vec.get_local()    


    dofmap = FA.dofmap()                                                             
    my_first, my_last = dofmap.ownership_range()                # global
    visited = []
    # dof_coordinates = FA.tabulate_dof_coordinates()

    for cell in cells(mesh):                                                        
        dofs = dofmap.cell_dofs(cell.index())                  # local                                    
        for dof in dofs:                                                            
            if not dof in visited:
                global_dof = dofmap.local_to_global_index(dof)  # global
                if my_first <= global_dof < my_last:                  
                    visited.append(dof)   
                    dof_coord = dof_coordinates[dof]
                    px, py = int((nx/length)*dof_coord[0]), int((ny/width)*dof_coord[1])
                    values[dof] = arr[px, py]

    vec.set_local(values)
    vec.apply('insert')


    solve(Res_Q == 0, y_new)
    y_old.assign(y_new)
    

    f = divQ
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx + mu*inner(u, v)*dx
    L = inner(f, v)*dx

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx + mu*inner(u, v)*dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs = None)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs = None)

    # Create Krylov solver and AMG pre-conditioner
    solver = KrylovSolver(krylov_method, "amg")

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)
    solver.solve(U.vector(), bb)

    ux = project(u0[0], FA)
    uy = project(u0[1], FA)
    #Qxx = project(Qxx_old, FA)
    #Qxy = project(Qxy_old, FA)



    ux_vec = ux.vector().get_local()
    uy_vec = uy.vector().get_local()
    #Qxx_vec = Qxx.vector().get_local()
    #Qxy_vec = Qxy.vector().get_local()

    comm.barrier()
    ux__ = np.hstack(mesh.mpi_comm().allgather(ux_vec))
    uy__ = np.hstack(mesh.mpi_comm().allgather(uy_vec))
    #Qxx__ = np.hstack(mesh.mpi_comm().allgather(Qxx_vec))
    #Qxy__ = np.hstack(mesh.mpi_comm().allgather(Qxy_vec))
    comm.barrier()

    
    ux_arr = np.zeros([ny + 1, nx + 1])
    uy_arr = np.zeros([ny + 1, nx + 1])

    for dof in range(len(dof_x)):

        px, py = int((ny/width)*dof_y[dof]), int((nx/length)*dof_x[dof])
        ux_arr[px, py] = ux__[dof]
        uy_arr[px, py] = uy__[dof]
        #Qxx_arr[px, py] = Qxx__[dof]
        #Qxy_arr[px, py] = Qxy__[dof]


    ux_arr = interpolate_up(ux_arr).astype(np.float16)
    uy_arr = interpolate_up(uy_arr).astype(np.float16)
    #Qxx_arr = interpolate_up(Qxx_arr).astype(np.float16)
    #Qxy_arr = interpolate_up(Qxy_arr).astype(np.float16)
    #alpha_arr  = interpolate_up(arr).astype(np.float16)

    obs = np.stack((ux_arr, uy_arr), axis=0)

    # print(np.max(np.abs(obs)), flush = True)    

    if time_count % 10 == 0 and rank == 0:
        print(f"at time {time_count}", flush = True)

    time_count = time_count + 1
    
    return obs


#######################################################################################################################

count = 0

while True:
    
    # Master rank handles socket

    payload = None

    if rank == 0:
        if (count == 0):

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("Socket successfully created")

            port = 8888
            s.bind(('localhost', port))
            print("Socket binded to %s" % (port), flush = True)

            s.listen(1)
            print("Socket is listening", flush = True)

        c, addr = s.accept()

        if count == 0:
            print('Got connection from', addr)

        x = np.zeros([nx + 1, ny + 1]).astype(np.float16)
        x_bytes= pickle.dumps(x) 

        received_data = b""
        while len(received_data) < len(x_bytes):
            chunk = c.recv(len(x_bytes) - len(received_data))
            if not chunk:
                raise RuntimeError("socket connection broken")
            received_data += chunk

        payload = pickle.loads(received_data)

        # print(f"average at recv at {count}", np.mean(payload))

    payload = comm.bcast(payload, root=0)

    if np.sum(payload) == 0:
        reset(payload)

    else: 
        next_state = time_step(payload)


    if rank == 0 and np.sum(payload) != 0:
        
        data_to_send = pickle.dumps(next_state)
        while data_to_send:
            sent = c.send(data_to_send)
            data_to_send = data_to_send[sent:]


    count  = count + 1


#######################################################################################################################
