# Numerical schemes for 4DVar
# (c) 2021 D. Bereziat, A. Filoche

import torch

def addborder(I,mode):
    if mode == 'copy':
        I = torch.cat((I[:,0].reshape(-1,1),I,I[:,-1].reshape(-1,1)),dim=1)
        I = torch.cat((I[0,:].reshape(1,-1),I,I[-1,:].reshape(1,-1)),dim=0)
    else:
        n,m = I.shape
        I = torch.cat((torch.zeros(n,1),I,torch.zeros(n,1)),dim=1)
        n,m = I.shape
        I = torch.cat((torch.zeros(1,m),I,torch.zeros(1,m)),dim=0)
    return I

def rmborder(I):
    return I[1:-1,1:-1]

def interp2(M,ei,ej,pi1,pj1,pi2,pj2):
    return (1-ej)*((1-ei)*M[pj1,pi1] + ei*M[pj1,pi2]) + ej*((1-ei)*M[pj2,pi1] + ei*M[pj2,pi2])


class aimi_sl:
    """ Integration of the AIMI model (Advection Image Model Internal) using
        a semi-Lagrangian scheme.

        State vector is a 3-D vector containing 2-D velocity and an image tracer.
        Velocity is advected by itself using a semi-Lagrangian scheme.
        Tracer is advected by velocity using the same scheme.

        Semi-Lagrangian is an implicit scheme that is unconditionally stable.

        See ....
    """
    def __init__(self, dx, dy, dt, ItDepPt=0):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.ItDepPt = ItDepPt

    def step(self, X, Xm1=None, EpsM=None):
        """ If Xm1 == None a simplified (and faster)scheme is used. Otherwise
            step() computes: X(t) = M(X(t-1), X(t-2)).
        """
        dx = self.dx
        dy = self.dy
        dt = self.dt
    
        U = addborder(X[:,:,0], 'null')
        V = addborder(X[:,:,1], 'null')
        Q = addborder(X[:,:,2], 'copy')

        if Xm1 != None:
            Um1 = addborder(Xm1[:,:,0], 'null')
            Vm1 = addborder(Xm1[:,:,1], 'null')
        
            di = - (3*U - Um1) * dt/(2*dx)
            dj = - (3*V - Vm1) * dt/(2*dy)
        else:
            di = - U * dt/dx
            dj = - V * dt/dy

        pi = torch.floor(di)
        pj = torch.floor(dj)
        epsi = di - pi
        epsj = dj - pj

        n,m = pi.shape
        j,i = torch.meshgrid(torch.arange(n),torch.arange(m))
        pi = pi + i
        pj = pj + j

        zeros = torch.zeros(n,m)
        ones = torch.ones(n,m)
        pi1 = torch.max(zeros,torch.min(pi,(m-1)*ones)).long()
        pi2 = torch.max(zeros,torch.min(pi+1,(m-1)*ones)).long()
        pj1 = torch.max(zeros,torch.min(pj,(n-1)*ones)).long()
        pj2 = torch.max(zeros,torch.min(pj+1,(n-1)*ones)).long()

        for k in range(self.ItDepPt):
            di = -(2*interp2(U,   epsi, epsj, pi1, pj1, pi2, pj2)
                   - interp2(Um1, epsi, epsj, pi1, pj1, pi2, pj2)
                   + U) * dt/(2*dx)
            dj = -(2*interp2(V,   epsi, epsj, pi1, pj1, pi2, pj2)
                   - interp2(Vm1, epsi, epsj, pi1, pj1, pi2, pj2)
                   + V) * dt/(2*dy)
        
            pi = torch.floor(di)
            pj = torch.floor(dj)
            epsi = di - pi
            epsj = dj - pj

            pi = pi + i
            pj = pj + j

            pi1 = torch.max(zeros,torch.min(pi,(m-1)*ones)).long()
            pi2 = torch.max(zeros,torch.min(pi+1,(m-1)*ones)).long()
            pj1 = torch.max(zeros,torch.min(pj,(n-1)*ones)).long()
            pj2 = torch.max(zeros,torch.min(pj+1,(n-1)*ones)).long()

        U = rmborder(interp2(U, epsi, epsj, pi1, pj1, pi2, pj2))
        V = rmborder(interp2(V, epsi, epsj, pi1, pj1, pi2, pj2))
        Q = rmborder(interp2(Q, epsi, epsj, pi1, pj1, pi2, pj2))

        if EpsM != None:
            U += EpsM[:,:,0]
            V += EpsM[:,:,0]


        return torch.stack((U,V,Q),dim=2)


def NormGrad(X, dx, dy):
    U = X[:,:,0]
    V = X[:,:,1]
    lin0 = torch.zeros((1, U.shape[1]))
    col0 = torch.zeros((U.shape[0], 1))
    Uleft = torch.cat((U[:,1:], col0), dim=1) - U
    Uup   = torch.cat((U[1:,:], lin0), dim=0) - U
    Vleft = torch.cat((V[:,1:], col0), dim=1) - V
    Vup   = torch.cat((V[1:,:], lin0), dim=0) - V
    return (Uleft/dx)**2 + (Uup/dy)**2 + (Vleft/dx)**2 + (Vup/dy)**2

def NormDiv(X,dx,dy):
    U = X[:,:,0]
    V = X[:,:,1]
    n, m = U.shape
    Uleft = torch.cat((U[:,1:], torch.zeros((n, 1))), dim=1) - U
    Vup   = torch.cat((V[1:,:], torch.zeros((1, m))), dim=0) - V
    return (Uleft/dx + Vup/dy)**2
