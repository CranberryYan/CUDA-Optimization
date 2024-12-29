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
CMAKE_SOURCE_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build

# Include any dependencies generated for this target.
include CMakeFiles/Sgemm_v2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Sgemm_v2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Sgemm_v2.dir/flags.make

CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o: CMakeFiles/Sgemm_v2.dir/flags.make
CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o: ../Sgemm_v2.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/Sgemm_v2.cu -o CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o

CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Sgemm_v2
Sgemm_v2_OBJECTS = \
"CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o"

# External object files for target Sgemm_v2
Sgemm_v2_EXTERNAL_OBJECTS =

app/Sgemm_v2: CMakeFiles/Sgemm_v2.dir/Sgemm_v2.cu.o
app/Sgemm_v2: CMakeFiles/Sgemm_v2.dir/build.make
app/Sgemm_v2: CMakeFiles/Sgemm_v2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable app/Sgemm_v2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Sgemm_v2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Sgemm_v2.dir/build: app/Sgemm_v2

.PHONY : CMakeFiles/Sgemm_v2.dir/build

CMakeFiles/Sgemm_v2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Sgemm_v2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Sgemm_v2.dir/clean

CMakeFiles/Sgemm_v2.dir/depend:
	cd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Sgemm/build/CMakeFiles/Sgemm_v2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Sgemm_v2.dir/depend

