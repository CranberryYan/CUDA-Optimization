# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build

# Include any dependencies generated for this target.
include CMakeFiles/index_v0.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/index_v0.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/index_v0.dir/flags.make

CMakeFiles/index_v0.dir/index_v0.cu.o: CMakeFiles/index_v0.dir/flags.make
CMakeFiles/index_v0.dir/index_v0.cu.o: ../index_v0.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/index_v0.dir/index_v0.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/index_v0.cu -o CMakeFiles/index_v0.dir/index_v0.cu.o

CMakeFiles/index_v0.dir/index_v0.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/index_v0.dir/index_v0.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/index_v0.dir/index_v0.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/index_v0.dir/index_v0.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target index_v0
index_v0_OBJECTS = \
"CMakeFiles/index_v0.dir/index_v0.cu.o"

# External object files for target index_v0
index_v0_EXTERNAL_OBJECTS =

app/index_v0: CMakeFiles/index_v0.dir/index_v0.cu.o
app/index_v0: CMakeFiles/index_v0.dir/build.make
app/index_v0: CMakeFiles/index_v0.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable app/index_v0"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/index_v0.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/index_v0.dir/build: app/index_v0

.PHONY : CMakeFiles/index_v0.dir/build

CMakeFiles/index_v0.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/index_v0.dir/cmake_clean.cmake
.PHONY : CMakeFiles/index_v0.dir/clean

CMakeFiles/index_v0.dir/depend:
	cd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Index/build/CMakeFiles/index_v0.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/index_v0.dir/depend
