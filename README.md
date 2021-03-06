# Laplacian_Surface_Editing_reproduce

A reproduce repository to the classic computer graphics paper: "Laplacian Surface Editing" (https://doi.org/10.1145/1057432.1057456) 

## Demo

+ Select boundary control points

<img src="imgs/1.png" alt="1" style="zoom: 25%;" />

+ Calculate the boundary points

<img src="imgs/2.png" alt="2" style="zoom:25%;" />

+ Select handle point and input displacement

<img src="imgs/3.png" alt="3" style="zoom:25%;" />

+ Calculate the edit point set

<img src="imgs/4.png" alt="4" style="zoom:25%;" />

Calculate the result

<img src="imgs/5.png" alt="5" style="zoom:25%;" />

## Requirements

- Programming Language: `python 3.x`
- Package Required: `numpy`, `open3d`, `scipy`
- No other graph base library (`networkx`, etc) needed

 All packages above can be installed with `pip install <package name>`.

## Usage

See `main.py`. Pass the mesh and relative offset to the core function `LSE()`. Interactively select the boundary control points and handle point of the graph. And it will calculates the deformation automatically. 

## Reference

The code refers to https://github.com/luost26/laplacian-surface-editing

The point cloud model is from The Stanford 3D Scanning Repository http://graphics.stanford.edu/data/3Dscanrep/#uses
