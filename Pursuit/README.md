# Multi-Agent Pursuit Using Deep Reinforcement Learning

This project implements an intelligent pursuit scenario where multiple agents (CCAs - Cooperative Cognitive Agents) learn to chase and capture a moving target (Foxtrot) in 3D space using deep reinforcement learning.

## Project Overview

The system uses the PPO (Proximal Policy Optimization) algorithm to train agents to cooperatively pursue a target. The agents learn optimal pursuit strategies through trial and error, receiving rewards for getting closer to the target and penalties for inefficient movements.

### Key Components

1. **simulation.py**: The core environment and training logic
   - Defines the 3D environment where pursuit takes place
   - Implements the PPO training environment
   - Handles reward calculations and agent movements
   - Contains the main training loop with curriculum learning

2. **showcase.py**: Visualization and demonstration
   - Loads trained models
   - Provides 3D visualization of pursuit scenarios
   - Shows paths of both agents and target
   - Creates animated visualizations of pursuit behavior

3. **classes.py**: Neural network architectures
   - Implements different attention mechanisms:
     - LMA (Latent Meta Attention)
     - MHA (Multi-Head Attention)
     - MHA_Lite (Lightweight Multi-Head Attention)

## How It Works

### Environment
- 3D grid space (default 500x500x500)
- Multiple pursuing agents (CCAs)
- Moving target (Foxtrot)
- Configurable movement patterns for the target:
  - Stationary
  - Random position
  - Rectangular path movement

### Learning Process
1. **State Space**: Each agent observes:
   - Its own position
   - Target position
   - Historical positions (last 6 timesteps)
   - Previous actions

2. **Action Space**:
   - Continuous 3D movement vectors
   - Agents decide direction and speed

3. **Reward System**:
   Two reward modes available:
   - **Basic**: Simple distance-based rewards
   - **Complex**: Sophisticated reward shaping with:
     - Progress rewards for getting closer
     - Energy efficiency penalties
     - Collision penalties
     - Exploration bonuses
     - Capture radius bonuses

### Training Curriculum
The agents learn through progressive stages:
1. Stationary target with nearby spawn points
2. Stationary target with random spawn points
3. Moving target following complex paths

## Getting Started

### Prerequisites
```bash
pip install gymnasium numpy torch stable-baselines3 matplotlib
```

### Training a Model
1. Configure training parameters in `simulation.py`
2. Run the training:
```bash
python simulation.py
```

### Visualizing Results
1. Configure visualization settings in `showcase.py`
2. Run the visualization:
```bash
python showcase.py
```

## Configuration Options

### Target Movement Modes
- `STATIONARY_FOXTROT`: Fixed position target
- `RECTANGULAR_FOXTROT`: Target moves in rectangular path
- `RAND_POS`: Random positioning
- `FIXED_POS`: Fixed starting position

### Agent Spawn Modes
- `RAND_FIXED_CCA`: Random spawn near target
- `PROXIMITY_CCA`: Spawn at target location
- `RAND_CCA`: Completely random spawn

### Reward Systems
- `BASIC_REWARD`: Simple distance-based
- `COMPLEX_REWARD`: Advanced shaping rewards

### Neural Network Architectures
- LMA: Best for long sequences
- MHA: Standard transformer attention
- MHA_Lite: Resource-efficient version

## Understanding the Code Structure

### Key Classes and Functions

1. **PPOEnv Class** (`simulation.py`)
   - Inherits from `gymnasium.Env`
   - Manages environment state
   - Handles agent-environment interactions
   - Calculates rewards

2. **ShowcaseSimulation Class** (`showcase.py`)
   - Handles visualization
   - Loads trained models
   - Creates 3D animations

### Important Parameters

- `grid_size`: Environment dimensions
- `step_size`: Maximum movement per step
- `spawn_range`: Initial spawn distance
- `capture_radius`: Success threshold
- `max_steps`: Episode length limit

## Advanced Features

### Attention Mechanisms
The project implements three types of attention:

1. **LMA (Linear Memory Attention)**
   - Efficient for long sequences
   - Better memory handling
   - Suitable for complex trajectories

2. **MHA (Multi-Head Attention)**
   - Standard transformer attention
   - Good for pattern recognition
   - More computationally intensive

3. **MHA_Lite**
   - Lightweight version
   - Faster training
   - Lower memory usage

### Curriculum Learning
The training process uses curriculum learning to gradually increase task difficulty:

1. **Stage 1**: Simple stationary target
2. **Stage 2**: Random target positions
3. **Stage 3**: Moving target with complex paths

This approach helps agents learn basic skills before tackling more complex scenarios.

## Tips for Modifications

1. **Adjusting Difficulty**
   - Modify `spawn_range` for initial positioning
   - Change `step_size` for movement speed
   - Adjust `capture_radius` for task difficulty

2. **Reward Tuning**
   - Modify reward weights in `_calculate_reward()`
   - Balance exploration vs. exploitation
   - Adjust penalty factors

3. **Network Architecture**
   - Choose between LMA/MHA/MHA_Lite
   - Modify network parameters
   - Adjust learning rates

## Contributing

Feel free to contribute to this project by:
1. Testing different network architectures
2. Implementing new reward structures
3. Adding different movement patterns
4. Improving visualization capabilities
