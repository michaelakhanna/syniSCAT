```markdown
[![DOI](https://zenodo.org/badge/1130729031.svg)](https://doi.org/10.5281/zenodo.18617286)

SyniSCAT is a Python-based simulation pipeline designed to generate synthetic datasets for Interferometric Scattering (iSCAT) microscopy. It provides a framework for creating large-scale training data for deep learning models (such as Transformers) by simulating the optical scattering of non-spherical particles and complex background noise.

Key Features:

- **Scattering Approximation**: Implements a "rigid sphere cluster" model to approximate the scattering patterns of complex particle shapes using superposition.
- **Defect Modeling**: Explicitly simulates lithography artifacts, including nano-hole irregularities, edge roughness, and "double-dipping" effects.
- **Data Pipeline**: Automates the generation of full video sequences paired with ground-truth segmentation masks.

# iSCAT Video Simulation – Conda Setup

1. Install Conda (Miniconda or Anaconda), if you don’t already have it:
   - https://docs.conda.io/en/latest/miniconda.html

2. Clone this repository:
   ```bash
   git clone https://github.com/michaelakhanna/iscat-video-simulation.git
   cd iscat-video-simulation
   ```

3. Create a separate Conda environment:
   ```bash
   conda create -n iscat-video-sim python=3.12.11 -y
   ```

4. Activate the environment:
   ```bash
   conda activate iscat-video-sim
   ```

5. Install the exact Python libraries used by the simulation code:
   ```bash
   conda install -c conda-forge \
       numpy==1.26.4 \
       scipy==1.16.0 \
       opencv==4.10.0 \
       tqdm==4.67.1
   ```

6. Run the simulation (from inside the `iscat` folder where `main.py` is):
   ```bash
   cd iscat
   python main.py
   ```

Additional record on Zenodo:  
https://zenodo.org/records/18634740

Michael A Khanna
