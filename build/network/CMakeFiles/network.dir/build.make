# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/tigranda1809/RTL AI"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/tigranda1809/RTL AI/build"

# Include any dependencies generated for this target.
include network/CMakeFiles/network.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include network/CMakeFiles/network.dir/compiler_depend.make

# Include the progress variables for this target.
include network/CMakeFiles/network.dir/progress.make

# Include the compile flags for this target's objects.
include network/CMakeFiles/network.dir/flags.make

network/CMakeFiles/network.dir/network.cpp.o: network/CMakeFiles/network.dir/flags.make
network/CMakeFiles/network.dir/network.cpp.o: /home/tigranda1809/RTL\ AI/network/network.cpp
network/CMakeFiles/network.dir/network.cpp.o: network/CMakeFiles/network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/tigranda1809/RTL AI/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object network/CMakeFiles/network.dir/network.cpp.o"
	cd "/home/tigranda1809/RTL AI/build/network" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT network/CMakeFiles/network.dir/network.cpp.o -MF CMakeFiles/network.dir/network.cpp.o.d -o CMakeFiles/network.dir/network.cpp.o -c "/home/tigranda1809/RTL AI/network/network.cpp"

network/CMakeFiles/network.dir/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/network.dir/network.cpp.i"
	cd "/home/tigranda1809/RTL AI/build/network" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/tigranda1809/RTL AI/network/network.cpp" > CMakeFiles/network.dir/network.cpp.i

network/CMakeFiles/network.dir/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/network.dir/network.cpp.s"
	cd "/home/tigranda1809/RTL AI/build/network" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/tigranda1809/RTL AI/network/network.cpp" -o CMakeFiles/network.dir/network.cpp.s

# Object files for target network
network_OBJECTS = \
"CMakeFiles/network.dir/network.cpp.o"

# External object files for target network
network_EXTERNAL_OBJECTS =

network/libnetwork.a: network/CMakeFiles/network.dir/network.cpp.o
network/libnetwork.a: network/CMakeFiles/network.dir/build.make
network/libnetwork.a: network/CMakeFiles/network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/tigranda1809/RTL AI/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libnetwork.a"
	cd "/home/tigranda1809/RTL AI/build/network" && $(CMAKE_COMMAND) -P CMakeFiles/network.dir/cmake_clean_target.cmake
	cd "/home/tigranda1809/RTL AI/build/network" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
network/CMakeFiles/network.dir/build: network/libnetwork.a
.PHONY : network/CMakeFiles/network.dir/build

network/CMakeFiles/network.dir/clean:
	cd "/home/tigranda1809/RTL AI/build/network" && $(CMAKE_COMMAND) -P CMakeFiles/network.dir/cmake_clean.cmake
.PHONY : network/CMakeFiles/network.dir/clean

network/CMakeFiles/network.dir/depend:
	cd "/home/tigranda1809/RTL AI/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/tigranda1809/RTL AI" "/home/tigranda1809/RTL AI/network" "/home/tigranda1809/RTL AI/build" "/home/tigranda1809/RTL AI/build/network" "/home/tigranda1809/RTL AI/build/network/CMakeFiles/network.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : network/CMakeFiles/network.dir/depend

