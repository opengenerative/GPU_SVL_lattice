# gpu_svl_lattice


This software package helps you to create lattice structures that can rotate, bend and change its size and shape spatially. This software build inorder to solve the problems facing in 'Deisgn for Additive Manufacturing' (DfAM) process. Knowledge sharing through source code could help the young minds to learn and gain confidence in handling DfAM applications and thus enhancing creativity thereby boosting skill development.

This software had complied and ran in Ubuntu 22.04.4 LTS (jammy). 


Users should have a GPU device with atleast 8 gb memory. It may run on lower end GPU's for low grid resolution but have not tested.


## Prerequisite
1. Install CUDA Toolkit (Cuda - GPU programming)
2. Install GLFW (Window Manager)
3. Install Vulkan (New generation Cross platform API for 3D graphics)

For more details please check DEPENDENCIES.md file.

## Download and Compilation 
1. Select a folder or directory
2. Use ' git clone https://github.com/opengenerative/GPU_SVL_lattice.git ' or download the "zip" folder and extract it in the folder(directory) selected.
3. Change directory to 'GPU_SVL_lattice' by using the command ' cd GPU_SVL_lattice/ ' or clicking on the ' GPU_SVL_lattice ' directory.
4. In the ' Makefile ' provide the location of ' nvcc ' compiler in CUDA_PATH. (here ' /usr/local/cuda-12.3 ')
5. Before compiling the cuda codes, in the command terminal type the command ' make compile_shader '. It will compile the shaders.This is necessary for the 
   first run and also if shader files changes.
6. Next type ' make all '
7. Check whether output you get int the terminal  message is ' Compilation Done! '.
8. After successful compilation run the executable ' ./generate_lattice ' with following argument values for each case.

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
     
    

