string(APPEND CONFIG_ARGS " --host=cray")
if (COMP_NAME STREQUAL gptl)
  string(APPEND CPPDEFS " -DHAVE_NANOTIME -DBIT64 -DHAVE_SLASHPROC -DHAVE_GETTIMEOFDAY")
  # Workaround GCC 12+ warnings that break gptl C build
  string(APPEND CMAKE_C_FLAGS " -Wno-error -Wno-implicit-function-declaration -Wno-implicit-int")
endif()
string(APPEND CMAKE_C_FLAGS_RELEASE " -O2 -g")
string(APPEND CMAKE_Fortran_FLAGS_RELEASE " -O2 -g")
# FTorch include paths and library for emulator
string(APPEND CMAKE_Fortran_FLAGS " -I/global/cfs/cdirs/m4549/code/FTorch_gnu_cpu/include")
string(APPEND CMAKE_Fortran_FLAGS " -I/global/cfs/cdirs/m4549/code/FTorch_gnu_cpu/build/modules")

set(MPICC "cc")
set(MPICXX "CC")
set(MPIFC "ftn")
set(SCC "gcc")
set(SCXX "g++")
set(SFC "gfortran")
