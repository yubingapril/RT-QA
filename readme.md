# Rhomboid tiling-based topological deep learning (RT-TDL) for quality assessment of protein complex structures

This repository provides the code implementation for the paper:
**Rhomboid tiling-based topological deep learning (RT-TDL) for quality assessment of protein complex structures**.
The code allows evaluating protein complex structures using higher order Delaunay graph representaion and rhomboid tiling-based message passing.

## Installation
Clone the repository and install dependencies.
This project requires **two separate environments**:

### Python environment (for prediction scripts)
```bash
cd RT-TDL
pip install -r requirements.txt
```
If you are using conda:
```bash
conda env create -f environment.yml
conda activate rttdl 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
### Rhomboid tiling environment 
* Python 3.5
* CGAL 4.9
* Requires compilation via CMake 
```bash
# create and activate RT enviroment 
conda create -n rt python=3.5 
conda activate rt

# install CGAL 
conda install -c conda-forge cgal=4.9

# Clone project repository 
cd rt
git clone https://github.com/geoo89/rhomboidtiling.git
cd rhomboidtiling
# In `./catch2/catch.hpp`, replace line 8164 with:
#    constexpr static std::size_t sigStackSize = 32768;

# Build project 
cmake .
make
```



## Input Data 
* complex_folder (-c): Directory containing input protein complex PDB files.
* rt_folder (-rt): Directory containing rhomboid tiling and Voronoi diagram computation code.
* work_dir (-w): Working directory to store intermediate files. 
* result_folder (-r): Directory to save evaluation results. 
Example folder structure:
```bash
project/
│── pdb/              # Input protein complexes (.pdb files)
│── rt/               # rhmboid tiling computation code
│── work/             # Temporary working directory
│── result/           # Output results
```

## Usage 
Run the prediction script:
```bash
python inference.py \
    --complex_folder ./pdb \
    --rt_folder ./rt \
    --work_dir ./work \
    --result_folder ./result \
    --delete_tmp False
```

## Output 
* CSV file in result_folder summarizing the predicted quality scores. 
* Optionally intermediate graphs and Voronoi partitioning in work_dir (if --delete_tmp False).

## Example
```bash
python inference.py -c ./pdb -rt ./rt -w ./work -r ./result
```
