import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

#Internal Imports
from classes import *

if __name__ == "__main__":
    engine = SimulationEngine(grid_size=1000)

    # Add objects to the simulation
    cca = CCA(name="CCA1", initial_position=[100, 100, 100])
    foxtrot = Foxtrot(name="Foxtrot1", initial_position=[200, 200, 200])
    engine.add_object(cca)
    engine.add_object(foxtrot)

    # Simulate with specific movement functions
    movement_functions = {
        "CCA1": cca_movement_fn,
        "Foxtrot1": foxtrot_movement_fn
    }
    engine.simulate(steps=50, movement_fns=movement_functions)