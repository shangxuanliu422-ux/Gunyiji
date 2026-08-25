# Architecture Notes

This project is organized as a staged validation stack:

1. Dynamics: state propagation and force/moment modeling.
2. Environment: 3D obstacles and collision queries.
3. Planning: global path generation with RRT*.
4. Control: MPC tracking and PID low-level stabilization.
5. Simulation: scenario orchestration and result logging.

