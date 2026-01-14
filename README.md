```markdown
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

6. Run the simulation (from inside the `iscat` folder where `main.py` lives):
   ```bash
   cd iscat
   python main.py
   ```
```
