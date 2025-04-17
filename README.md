# ObjectTrackingRL: Reinforcement Learning Scenarios and Simulation Engines

Welcome to **ObjectTrackingRL**, a learning-focused repository designed to provide a wide range of reinforcement learning (RL) scenarios—from the most basic, using only PyTorch modules, to advanced experiments leveraging Stable-Baselines3 and Gym environments. This project is ideal for students, educators, and researchers who want to understand, build, and extend RL environments and agents from the ground up.

---

## 🚩 **Project Goals**

- **Educational Resource:** Serve as a comprehensive, step-by-step RL learning platform.
- **Scenario Diversity:** Demonstrate RL in simple to complex environments, including custom digital simulation engines.
- **Modularity:** Show how to build, modify, and extend RL environments and agent architectures.
- **Community Growth:** Encourage contributions and collaborative learning.

---

## 📁 **Repository Structure**

- **OpenAI_Envs/**  
  Contains scripts (e.g., `train_v2.py`) for running RL experiments using Stable-Baselines3 and Gym environments. Includes support for custom feature extractors (MHA, LMA, etc.) and logging of evaluation results.

- **Pursuit/**  
  Implements a custom multi-agent pursuit environment with curriculum learning, advanced reward shaping, and flexible agent/target spawning. Demonstrates how to build a digital simulation engine from scratch using Gym's API.

- **Swarm/**  
  Contains environments and utilities for multi-agent swarm scenarios, including formation control and curriculum-based training.

- **EVALS/**  
  Stores evaluation results and logs for different experiments and feature extractors.

- **classes.py**  
  Defines custom feature extractors, neural network modules, and configuration dataclasses used throughout the codebase.

- **tqdm_utils.py**  
  Utilities for progress bar management and output suppression.

---

## 🏗️ **How to Use**

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run RL Experiments**
   - Navigate to `OpenAI_Envs/` and run:
     ```bash
     python train_v2.py
     ```
   - Select the experiment table (quick/full) and the script will iterate through environments and feature extractors, logging results to `results.log`.

3. **Explore Custom Environments**
   - Check the `Pursuit/` and `Swarm/` folders for custom Gym environments.
   - Modify environment hyperparameters (e.g., grid size, agent count, reward shaping) in the respective `simulation.py` or `globals.py` files.

4. **Extend and Experiment**
   - Add new feature extractors or neural architectures in `classes.py`.
   - Create new environments or modify existing ones to test different RL ideas.

---

## 🧑‍💻 **Contributing**

Contributions are **highly encouraged**! This repository aims to grow into a comprehensive RL learning hub for young researchers and practitioners. You can contribute by:

- Adding new RL scenarios or environments
- Improving documentation and tutorials
- Implementing new feature extractors or agent architectures
- Sharing evaluation results and benchmarks
- Suggesting curriculum improvements or new reward structures

---

## 📝 **Customization & Tips**

- **Environment Hyperparameters:**  
  Easily modify agent count, grid size, reward shaping, and more in each environment's config files.
- **Simulation Engine:**  
  Learn how to build your own Gym-compatible simulation engine by studying the `Pursuit/` and `Swarm/` modules.
- **Feature Extractors:**  
  Experiment with Baseline, MHA, LMA, and MHA_Lite extractors to see their impact on learning.
- **Logging & Evaluation:**  
  All experiments log results to `results.log` for easy comparison.

---

## 🎓 **Learning Path**

1. **Start Simple:**  
   Run basic experiments in `OpenAI_Envs/` to understand RL workflows.
2. **Dive Deeper:**  
   Explore custom environments in `Pursuit/` and `Swarm/`.
3. **Experiment:**  
   Modify hyperparameters, architectures, and reward functions.
4. **Contribute:**  
   Share your findings and help others learn!

---

## 📬 **Get Involved**

If you have questions, ideas, or want to contribute, please open an issue or submit a pull request. Let's build a great RL learning resource together!

---

**Happy Learning and Experimenting!**