NVCC      ?= nvcc
ARCH      ?=
CXXFLAGS  ?= -O3 --std=c++17

ifeq ($(ARCH),)
ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d .)
endif
ifeq ($(ARCH),)
ARCH := 80
endif

# Single-architecture build for the local GPU by default. Pass a list like
# ARCHS="80 90 100" for a fat binary; PTX for the newest entry is embedded
# so later GPU generations load it too.
ARCHS     ?=
ifeq ($(ARCHS),)
GENCODE   := -arch=sm_$(ARCH)
else
GENCODE   := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
GENCODE   += -gencode arch=compute_$(lastword $(ARCHS)),code=compute_$(lastword $(ARCHS))
endif

ifeq ($(ARCHS)$(ARCH),80)
GENCODE   += -maxrregcount=96
endif
ifeq ($(strip $(ARCHS)),80)
GENCODE   += -maxrregcount=96
endif

INCLUDE   := -Iinclude -Isrc
SRC       := src/szgpu.cu
HDRS      := include/szgpu/szgpu.h include/szgpu/szgpu.hpp src/szgpu_internal.cuh src/szgpu_kernels.cuh

BUILD     ?= build
LIB       := $(BUILD)/libszgpu.a
SHLIB     := $(BUILD)/libszgpu.so
TOOL      := $(BUILD)/szgpu
EXAMPLES  := $(BUILD)/example_roundtrip $(BUILD)/example_hostptr
TESTS     := $(BUILD)/test_roundtrip $(BUILD)/test_hostptr

.PHONY: all lib shared tool examples tests fortran clean install

all: lib tool examples tests

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/szgpu.o: $(SRC) $(HDRS) | $(BUILD)
	$(NVCC) $(CXXFLAGS) $(GENCODE) -c $(SRC) -o $@ $(INCLUDE) -Xcompiler -fPIC

$(LIB): $(BUILD)/szgpu.o
	ar rcs $@ $<

$(SHLIB): $(SRC) $(HDRS) | $(BUILD)
	$(NVCC) $(CXXFLAGS) $(GENCODE) -shared $(SRC) -o $@ $(INCLUDE) -Xcompiler -fPIC

lib: $(LIB)
shared: $(SHLIB)

$(TOOL): tools/szgpu.cu tools/szgpu_file_format.hpp $(LIB)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $< -o $@ $(INCLUDE) -Itools $(LIB)

tool: $(TOOL)

$(BUILD)/example_%: examples/example_%.cu $(LIB)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $< -o $@ $(INCLUDE) $(LIB)

$(BUILD)/test_%: tests/test_%.cu $(LIB)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $< -o $@ $(INCLUDE) $(LIB)

examples: $(EXAMPLES)
tests:    $(TESTS)

# Fortran interface (optional): builds the szgpu module plus its example and
# test against the static library. Requires gfortran or a compatible FC.
ifeq ($(origin FC),default)
FC        := gfortran
endif
CUDA_HOME ?= $(shell dirname $$(dirname $$(command -v nvcc)))

fortran: $(LIB) | $(BUILD)
	$(FC) -O2 -J$(BUILD) -c fortran/szgpu.f90 -o $(BUILD)/szgpu_mod.o
	$(FC) -O2 -I$(BUILD) fortran/example_szgpu.f90 $(BUILD)/szgpu_mod.o $(LIB) \
	    -o $(BUILD)/example_fortran -L$(CUDA_HOME)/lib64 -lcudart -lstdc++
	$(FC) -O2 -I$(BUILD) fortran/test_szgpu.f90 $(BUILD)/szgpu_mod.o $(LIB) \
	    -o $(BUILD)/test_fortran -L$(CUDA_HOME)/lib64 -lcudart -lstdc++

PREFIX ?= /usr/local
install: $(LIB) $(TOOL)
	install -d $(PREFIX)/bin $(PREFIX)/lib $(PREFIX)/include/szgpu
	install -m 755 $(TOOL) $(PREFIX)/bin/
	install -m 644 $(LIB) $(PREFIX)/lib/
	install -m 644 include/szgpu/szgpu.h include/szgpu/szgpu.hpp $(PREFIX)/include/szgpu/

clean:
	rm -rf $(BUILD)
