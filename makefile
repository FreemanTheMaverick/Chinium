# --- Compiler ---
CXX = __CXX__
# Note: MAKE is implicitly defined, no need to export unless overriding

# --- Paths to Dependencies ---
EIGEN3_PATH    = __EIGEN3__
# Eigen3: The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
LIBINT2_PATH   = __LIBINT2__
# LIBINT2: The path where you can find "include/", "lib/" and "share/".
LIBXC_PATH     = __LIBXC__
# LIBXC: The path where you can find "bin/", "include/" and "lib/".
LIBMWFN_PATH   = __LIBMWFN__
MANIVERSE_PATH = __MANIVERSE__

# --- Project Structure ---
TARGET      = Chinium
SRCDIR      = src
OBJDIR      = obj
SOURCES     = $(shell find $(SRCDIR) -name '*.cpp')
# Generate corresponding object file paths in OBJDIR, preserving subdirectory structure
OBJECTS     = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
# Generate dependency files (optional but good practice for header changes)
DEPS        = $(OBJECTS:.o=.d)

# --- Build Flags ---
# General Compiler Flags (apply to compilation steps)
# Using -isystem for external libraries suppresses warnings from their headers
# Added -MMD -MP to generate dependency files (.d)
CPPFLAGS    = -isystem $(EIGEN3_PATH) \
              -isystem $(LIBINT2_PATH)/include \
              -isystem $(LIBXC_PATH)/include \
              -isystem $(LIBMWFN_PATH)/include \
              -isystem $(MANIVERSE_PATH)/include \
              -DEIGEN_INITIALIZE_MATRICES_BY_ZERO \
              -DEIGEN_MAX_ALIGN_BYTES=64 \
              -MMD -MP # Generate dependency files

CXXFLAGS    = -std=c++2a \
			  -O3 -fopenmp -march=native \
			  -Wall -Wextra -Wpedantic -Wno-array-bounds

# Linker Flags (apply to the final linking step)
# -L flags specify paths for the *linker* to search during the build
# -Wl,-rpath, flags embed paths into the executable for the *runtime dynamic linker*
LDFLAGS     = -L$(LIBINT2_PATH)/lib \
              -L$(LIBXC_PATH)/lib \
              -L$(LIBXC_PATH)/lib64 \
              -L$(LIBMWFN_PATH)/lib \
              -L$(MANIVERSE_PATH)/lib \
              -Wl,-rpath,$(LIBINT2_PATH)/lib \
              -Wl,-rpath,$(LIBXC_PATH)/lib \
              -Wl,-rpath,$(LIBXC_PATH)/lib64 \
              -Wl,-rpath,$(LIBMWFN_PATH)/lib \
              -Wl,-rpath,$(MANIVERSE_PATH)/lib \
              -fopenmp # Often needed for linking OpenMP code too

# Libraries to Link
LDLIBS      = -lint2 \
              -lxc \
              -l:libmwfn.a \
              -l:libmaniverse.a # Explicitly link the static archive

# --- Main Rules ---

# Default target: build the executable
.PHONY: all
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)
# $^ expands to all prerequisites (the object files)
# CXXFLAGS are included in the link step (good practice for LTO, -fopenmp etc.)

# --- Compilation Rule ---

# Pattern rule to compile .cpp files from SRCDIR into .o files in OBJDIR
# This handles source files in subdirectories of SRCDIR as well.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR) # Use order-only prerequisite for OBJDIR
	@echo "Compiling $< -> $@ ..."
	@mkdir -p $(@D) # Create subdirectory in obj/ if it doesn't exist
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
# $< expands to the first prerequisite (the .cpp file)
# $@ expands to the target (the .o file)
# $(@D) expands to the directory part of the target

# --- Utility Rules ---

# Rule to create the object directory
# This is triggered by the order-only prerequisite in the compilation rule
$(OBJDIR):
	@echo "Creating directory $@"
	@mkdir -p $@

# Rule to clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)   # Remove executable
	rm -rf $(OBJDIR)  # Remove object directory and all its contents (.o, .d files)

# Include dependency files, if they exist
# This makes Make automatically recompile files if included headers change
-include $(DEPS)
