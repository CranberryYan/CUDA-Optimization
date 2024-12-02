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
CMAKE_SOURCE_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build

# Include any dependencies generated for this target.
include CMakeFiles/fastatomicadd_baseline.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fastatomicadd_baseline.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fastatomicadd_baseline.dir/flags.make

CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o: CMakeFiles/fastatomicadd_baseline.dir/flags.make
CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o: ../fastatomicadd_baseline.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/fastatomicadd_baseline.cu -o CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o

CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target fastatomicadd_baseline
fastatomicadd_baseline_OBJECTS = \
"CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o"

# External object files for target fastatomicadd_baseline
fastatomicadd_baseline_EXTERNAL_OBJECTS =

app/fastatomicadd_baseline: CMakeFiles/fastatomicadd_baseline.dir/fastatomicadd_baseline.cu.o
app/fastatomicadd_baseline: CMakeFiles/fastatomicadd_baseline.dir/build.make
app/fastatomicadd_baseline: CMakeFiles/fastatomicadd_baseline.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable app/fastatomicadd_baseline"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fastatomicadd_baseline.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fastatomicadd_baseline.dir/build: app/fastatomicadd_baseline

.PHONY : CMakeFiles/fastatomicadd_baseline.dir/build

CMakeFiles/fastatomicadd_baseline.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fastatomicadd_baseline.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fastatomicadd_baseline.dir/clean

CMakeFiles/fastatomicadd_baseline.dir/depend:
	cd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/FastAtomicAdd/build/CMakeFiles/fastatomicadd_baseline.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fastatomicadd_baseline.dir/depend

