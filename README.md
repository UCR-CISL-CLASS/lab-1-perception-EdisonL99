[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/w3FoW0fO)
# EE260C_LAB1: Perception

Please refer to the instructions [here](https://docs.google.com/document/d/1BvQ9ztEvxDwsHv-RWEy2EOA7kdAonzdkbJIuQSB1nJI/edit?usp=sharing)

# Getting Started

**Step 1** Environment Setup

Create a conda environment for CARLA 0.9.15 and MMDetection3D. 

Instruction to install CARLA [here](https://carla.readthedocs.io/en/latest/start_quickstart/).

Instruction to install MMDetection3D [here](https://github.com/open-mmlab/mmdetection3d).

**Step 2** Get the Repo

Git clone this repo to a local directory. 
```
/home/your directory/lab-1-perception-EdisonL99
```
**Step 3** Setup CARLA

Open a terminal under CARLA directory, run:
```
./CarlaUE4.sh
```

**Step 4** Run Python Scripts

Open two terminals in the repo folder and run the following commands respectively. 
```
python3 generate_traffic.py
python3 automatic_control.py
```
