# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-12.3

TARGET_SIZE := $(shell getconf LONG_BIT)

# operating system
HOST_OS   := Linux
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),Linux))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
ALL_CCFLAGS := -m${TARGET_SIZE}


INCLUDES  := -I./commons
LIBRARIES :=

################################################################################

# find Vulkan SDK and dependencies

VULKAN_HEADER := /usr/include
VULKAN_SDK_LIB := /usr/lib/x86_64-linux-gnu 
# Vulkan specific libraries
ifeq ($(TARGET_OS),Linux)

 LIBRARIES += -L$(VULKAN_SDK_LIB)
 LIBRARIES += `pkg-config --static --libs glfw3` -lvulkan
 INCLUDES  += `pkg-config --static --cflags glfw3` -I$(VULKAN_HEADER)
 
endif
CURRENT_SM := 86
HIGHEST_SM := 90
GENCODE_FLAGS += -gencode arch=compute_$(CURRENT_SM),code=compute_$(CURRENT_SM)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
ALL_CCFLAGS += --std=c++14 --threads 0
LIBRARIES += -lcufft

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
################################################################################
msg := "Compiling Shaders"
compile_shader: 
	@echo $(msg)
	./compile.sh

	


msg1 := "Compiling Spatial Grating file 'Gratings.cu' - Step 1 of 7"

msg2 := "Compiling Unit Lattice Generation FIle 'Fft_lattice.cu' - Step 2 of 7"

msg3 := "Compiling Obj writer File 'File_output.cu' - Step 3 of 7"

msg4 := "Compiling Marching Cube File 'MarchingCubes_kernel.cu' - Step 4 of 7"

msg5 := "Compiling Vulkan application file - 'VulkanApp.cu' - Step 5 of 7"

msg6 := "Compiling Main file 'main.cu' - Step 6 of 7"

msg7 := "Building the executable './generate_lattice' - Step 7 of 7"

msg8 := "Compilation Done!"

# Target rules
all: build


build: generate_lattice


	
Gratings.o:Gratings.cu
	@echo $(msg1)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $< 

Fft_lattice.o:Fft_lattice.cu
	@echo $(msg2)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

File_output.o:File_output.cu
	@echo $(msg3)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

MarchingCubes_kernel.o:MarchingCubes_kernel.cu
	@echo $(msg4)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


VulkanApp.o:VulkanApp.cpp
	@echo $(msg5)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cu
	@echo $(msg6)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


generate_lattice: Gratings.o Fft_lattice.o File_output.o MarchingCubes_kernel.o VulkanApp.o main.o
	@echo $(msg7)
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	@echo $(msg8)
 
run: build

	./generate_lattice 30 1.0 n u false

clean:
	rm -f Gratings.o Fft_lattice.o File_output.o MarchingCubes_kernel.o VulkanApp.o main.o generate_lattice

