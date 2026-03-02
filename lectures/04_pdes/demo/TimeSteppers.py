import math


def FE_Step(t, u, F, dt, info):
    """Perform one Forward Euler step
       - t current time
       - u current solution
       - F function that computes the right-hand-side 'F' in
            du/dt = F
         The caling sequence is F(t, u, info)
       - dt time-step
       - info extra information needed by the right-hand-side function
         (e.g. mass-matrices for DG)
    returns
       t_new, u_new
    """
    u_new = u + dt * F(t, u, info)
    return t + dt, u_new


def RK2_Step(t, u, F, dt, info):
    """ "Perform one Runge-Kutta 2 step
       - t current time
       - u current solution
       - F function that computes the right-hand-side 'F' in
            du/dt = F
         The caling sequence is F(t, u, info)
       - dt time-step
       - info extra information needed by the right-hand-side function
         (e.g. mass-matrices for DG)
    returns
       t_new, u_new
    """

    k1 = dt * F(t, u, info)
    k2 = dt * F(t + 0.5 * dt, u + 0.5 * k1, info)
    return (t + dt, u + k2)


def RK4_Step(t, u, F, dt, info):
    """ "Perform one Runge-Kutta 4 step
       - t current time
       - u current solution
       - F function that computes the right-hand-side 'F' in
            du/dt = F
         The caling sequence is F(t, u, info)
       - dt time-step
       - info extra information needed by the right-hand-side function
         (e.g. mass-matrices for DG)
    returns
       t_new, u_new
    """

    k1 = dt * F(t, u, info)
    k2 = dt * F(t + 0.5 * dt, u + 0.5 * k1, info)
    k3 = dt * F(t + 0.5 * dt, u + 0.5 * k2, info)
    k4 = dt * F(t + dt, u + k3, info)
    return (t + dt, u + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)


def Evolve(t, t_final, u, F, Tstepper, CF, info):
    """Evolve the evolution equations represented by right-hand-side 'F'
    with time-stepper Tstepper until final time 't_final'.
      - t current time
      - t_final final time
      - u solution at current time 't'
      - F right-hand-side of evolution equations, w/ calling sequence F(t, u, info)
      - Tstepper time-stepper with calling sequence Tstepper(t, u, F, dt, info)
      - CF courant-factor, i.e dt will be chosen such that dt < C dxmin
      - info class holding any additional information necessary.  It is passed into 'rhs'.
        It is also assumed that info.dxmin returns the minimal grid-spacing.
    returns
      t_final, u_final"""

    # time-step
    dt = CF * info.dxmin
    Nsteps = math.ceil((t_final - t) / dt)  # round up number of steps
    dt = (t_final - t) / Nsteps  # adjust dt to precisely reach t_final
    for i in range(Nsteps):
        t, u = Tstepper(t, u, F, dt, info)
    return t, u
