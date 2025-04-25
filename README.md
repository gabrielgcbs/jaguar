# Jaguar: A Hierarchical Deep Reinforcement Learning approach with Transfer Learning for StarCraft II

## Table of Contents
- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Hierarchical Approach](#hierarchical-approach)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
``Jaguar`` is a thesis project with the objective of developing an hierarchical approach that is able to deal with the complexity of the StarCraft II RTS game. This README provides an overview of the project's structure, architecture, dependencies, and usage.

## Folder Structure
The project directory is organized as follows:
```
jaguar
├── config/
|   ├── avail_actions.py    # Script that creates the actions
|   ├── avail_maps.yaml     # Maps configuration file
|   ├── params.yaml         # Agent configuration file
├── data/                   # Data files and datasets
|   ├── models/             # Neural networks trained
|   ├── replays/            # Replays of the matches
|   ├── results/            # Results and metrics of the runs
├── hierarchical_agent/
│   ├── agents/         
|   |   ├── jaguar.py       # The hierarchical agent
|   |   ├── policy.py       # The high-level and low-level agents
│   ├── environment.py      # Wrapper for the PySC2 environment
│   └── models.py           # Deep Reinforcement Learning models
├── run.py                  # Main script to run the agent
├── utils.py                # Script containing utils functions
├── makefile                
├── README.md           
└── requirements.txt
```

## Hierarchical Approach
The project follows a hierarchical approach to ensure modularity and maintainability. The approach is as follows:

1. **Hierarchy**: A high-level agent to deal with macromanagement (strategic) decisions, and a lower-level agent to deal with micromanagement (tactical) decisions

2. **RL Models**: [DDQN](https://arxiv.org/abs/1509.06461) for macromanagement and [DQN](https://arxiv.org/abs/1312.5602) for micromanagement

3. **Spaces**: Maps the observation space to a Grid to reduce complexity and defines the strategy spaces to hold specific micro-actions

4. **Action masking**: Masking invalid actions from both the observation and the communication between the high-level agent and the lower-level agent

5. **Reward structure**: Immediate reward for the lower-level agent and a sparse reward for the high-level agent

6. **Transfer Learning**: Easily switch between normal training mode and transfer learning mode to apply the knowledge in a new scenario

7. **StarCraft II**: Simulation environment using [PySC2](https://github.com/google-deepmind/pysc2)

## Dependencies
The project relies on the following main dependencies:
- Python==3.10.5
- PySC2==4.0.0
- torch==2.3.1

> [!IMPORTANT]
> I hightly recommend to use a `virtual environment`:
> ```sh
> python -m venv .venv
> ```

You can install all dependencies using:

```sh
make install
```
or manually:

```sh
pip install -r requirements.txt
```

## Usage
To run the agent, simple use:
```sh
make run
```
Ensure that all dependencies are installed before running the agent.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the GNU GPLv3 License. See the LICENSE file for more details.

## Citing
If you use Jaguar in your research, please cite:

```bibtex
@mastersthesis{Jaguar,
  address={Natal, RN, Brazil},
  title={Jaguar: a Hierarchical Deep Reinforcement Learning Approach with Transfer Learning for StarCraft II},
  url={https://repositorio.ufrn.br/handle/123456789/63424},
  author={Gabriel Caldas Barros e Sá},
  year={2024},
  month={December},
  school={Universidade Federal do Rio Grande do Norte \(UFRN\)},
  type={Master's thesis}
}
```

---

You can read the thesis [here](https://repositorio.ufrn.br/handle/123456789/63424).

---
