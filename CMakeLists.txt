cmake_minimum_required(VERSION 3.18)

# Default GPU architectures. Users can override with
# -DCMAKE_CUDA_ARCHITECTURES=... ; sm_100 (Blackwell) is appended
# automatically when the CUDA toolkit is new enough to know it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
    set(_szgpu_default_archs ON)
endif()

project(szgpu LANGUAGES CXX CUDA VERSION 1.0.1)

if(_szgpu_default_archs AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100)
endif()

if(CMAKE_CUDA_ARCHITECTURES MATCHES "^80(-real)?$")
    string(APPEND CMAKE_CUDA_FLAGS " -maxrregcount=96")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

option(SZGPU_BUILD_SHARED   "Also build the shared library (libszgpu.so)" ON)
option(SZGPU_BUILD_TOOLS    "Build the szgpu command-line tool"            ON)
option(SZGPU_BUILD_EXAMPLES "Build SZ-GPU examples"                         ON)
option(SZGPU_BUILD_TESTS    "Build SZ-GPU tests"                            ON)

set(SZGPU_SOURCES src/szgpu.cu)

# Static library (the default link target, exported as szgpu::szgpu).
add_library(szgpu STATIC ${SZGPU_SOURCES})
add_library(szgpu::szgpu ALIAS szgpu)

# Shared library for dynamic linking and the Python bindings.
if(SZGPU_BUILD_SHARED)
    add_library(szgpu_shared SHARED ${SZGPU_SOURCES})
    set_target_properties(szgpu_shared PROPERTIES OUTPUT_NAME szgpu)
endif()

foreach(_szgpu_lib IN ITEMS szgpu szgpu_shared)
    if(NOT TARGET ${_szgpu_lib})
        continue()
    endif()
    target_include_directories(${_szgpu_lib}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${_szgpu_lib} PUBLIC CUDA::cudart)
    set_target_properties(${_szgpu_lib} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE  ON)
    target_compile_options(${_szgpu_lib} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# Command-line tool: the binary is named `szgpu`.
if(SZGPU_BUILD_TOOLS)
    add_executable(szgpu_cli tools/szgpu.cu)
    set_target_properties(szgpu_cli PROPERTIES OUTPUT_NAME szgpu)
    target_include_directories(szgpu_cli PRIVATE tools)
    target_link_libraries(szgpu_cli PRIVATE szgpu::szgpu)
endif()

if(SZGPU_BUILD_EXAMPLES)
    add_executable(szgpu_example_roundtrip examples/example_roundtrip.cu)
    target_link_libraries(szgpu_example_roundtrip PRIVATE szgpu::szgpu)

    add_executable(szgpu_example_hostptr examples/example_hostptr.cu)
    target_link_libraries(szgpu_example_hostptr PRIVATE szgpu::szgpu)
endif()

if(SZGPU_BUILD_TESTS)
    enable_testing()
    add_executable(szgpu_test_roundtrip tests/test_roundtrip.cu)
    target_link_libraries(szgpu_test_roundtrip PRIVATE szgpu::szgpu)
    add_test(NAME roundtrip COMMAND szgpu_test_roundtrip)

    add_executable(szgpu_test_hostptr tests/test_hostptr.cu)
    target_include_directories(szgpu_test_hostptr PRIVATE src)
    target_link_libraries(szgpu_test_hostptr PRIVATE szgpu::szgpu)
    add_test(NAME hostptr COMMAND szgpu_test_hostptr)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Fortran interface: built whenever a Fortran compiler is available.
option(SZGPU_BUILD_FORTRAN "Build the Fortran interface" ON)
if(SZGPU_BUILD_FORTRAN)
    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        add_library(szgpu_fortran STATIC fortran/szgpu.f90)
        target_link_libraries(szgpu_fortran PUBLIC szgpu::szgpu stdc++)
        set_target_properties(szgpu_fortran PROPERTIES
            Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules)
        target_include_directories(szgpu_fortran INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>)
        if(SZGPU_BUILD_EXAMPLES)
            add_executable(szgpu_example_fortran fortran/example_szgpu.f90)
            target_link_libraries(szgpu_example_fortran PRIVATE szgpu_fortran)
        endif()
        if(SZGPU_BUILD_TESTS)
            add_executable(szgpu_test_fortran fortran/test_szgpu.f90)
            target_link_libraries(szgpu_test_fortran PRIVATE szgpu_fortran)
            add_test(NAME fortran COMMAND szgpu_test_fortran)
        endif()
        install(TARGETS szgpu_fortran
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/szgpu.mod
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/szgpu)
    else()
        message(STATUS "SZ-GPU: no Fortran compiler found; skipping the Fortran interface")
    endif()
endif()

install(TARGETS szgpu EXPORT szgpuTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(SZGPU_BUILD_SHARED)
    install(TARGETS szgpu_shared EXPORT szgpuTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(SZGPU_BUILD_TOOLS)
    install(TARGETS szgpu_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT szgpuTargets
    FILE szgpuTargets.cmake
    NAMESPACE szgpu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/szgpu)

configure_package_config_file(cmake/szgpuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/szgpuConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/szgpu)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/szgpuConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/szgpuConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/szgpuConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/szgpu)
