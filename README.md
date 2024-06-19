# gpu_svl_lattice


This software package helps you to create lattice structures that can rotate, bend and change its size and shape spatially. This software build inorder to solve the problems facing in 'Deisgn for Additive Manufacturing' (DfAM) process. Knowledge sharing through source code could help the young minds to learn and gain confidence in handling DfAM applications and thus enhancing creativity thereby boosting skill development.

This software had complied and ran in Ubuntu 22.04.4 LTS (jammy). 


Users should have a GPU device with atleast 8 gb memory. It may run on lower end GPU's for low grid resolution but have not tested.


## Prerequisite
1. Install CUDA Toolkit (Cuda - GPU programming)
2. Install GLFW (Window Manager)
3. Install Vulkan (New generation Cross platform API for 3D graphics)

For more details please check DEPENDENCIES.md file.

## Compilation 

1. In the 'Makefile' provide the location of 'nvcc' compiler in CUDA_PATH. (here '/usr/local/cuda-12.3')
2. Before compiling the cuda codes, compile the shaders using the command 'make compile_shader' in current working directory terminal.This is necessary for the first run and also if shader files changes.
3. Then type 'make all'
4. Check whether output you get int the terminal  message is 'Compilation Done!'.
5. After successful compilation run the executable './generate_lattice' with following argument values for each case.

    **./(Executable_name) (grid_size) (grid_spacing) (n or b or r) (u or v) (false or true)**
   
    Example : ***./generate_lattice 100 1.0 n u false***
   
    Explanation :
    * argument value 1 - execuatble with name 'generate_lattice'
    * argument value 2 - grid size (values in between 16 to 150) 
    * argument value 3 - grid spacing (value is 1.0, not changeable)
    * argument value 4 - 

        * n - normal lattice

        * b - bend lattice

        * r - round lattice

    * argument value 5 -

        * u - uniform lattice

        * v - varying lattice

    * argument value 6 - 

        * false - no obj file output
    
        * true - two file output in obj format namely 'Unit_Lattice.obj' for the unit cell and 'Spatial_Lattice.obj' for spatial lattice structure.
     
    

