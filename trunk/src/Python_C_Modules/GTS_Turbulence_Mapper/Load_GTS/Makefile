#############################################################################
#             Makefile to build the GTS code
#           ==================================
#
# You only need to type "gmake" to build the code on the platforms
# defined below. The makefile runs the "uname -s" command to detect
# the operating system automatically.
#
# Other options are:
#
#  % gmake OPENMP=y       Builds the code with OpenMP support
#
#  % gmake DEBUG=y        Compiles the files with debug option (-g)
#
#  % gmake ESSL=y         On AIX, uses the FFT routine from ESSL library
#                         The default is to use the NAG library routine
#
# You can combine more than one option on the command line:
#
#  % gmake OPENMP=y ESSL=y
#
# Special targets:
#
#  % gmake clean      Removes the executable and all object files (*.o)
#
#  % gmake cleanomp   Removes the executable and the object files
#                     containing OpenMP directives
#
#############################################################################
OPENMP=y
DEBUG=y

# In the next declaration, "uname" is a standard Unix command which prints
# the name of the current system.
os:= $(shell uname -s)

# Common file for fft routine using NAG except on the CRAY T3E
FFT:=fft_gl.o

# By default we don't use SUPERLU
SUPERLU_FILES:=

# We now try to determine which compiler to use on the current system
ifeq ($(shell which ftn 2> /dev/null | sed 's/.*\///'),ftn)
   # Cray systems, XT or X1
   CMP:=ftn
   F90C:=ftn
else
   ifeq ($(shell which mpif90 2> /dev/null | sed 's/.*\///'),mpif90)
      CMP:=mpif90
      ifeq ($(shell which lf95 2> /dev/null | sed 's/.*\///'),lf95)
         # Lahey compiler
         F90C:=lf95
      endif
      ifeq ($(shell which lfc 2> /dev/null | sed 's/.*\///'),lfc)
         # New Lahey compiler
         F90C:=lfc
      endif
      ifeq ($(shell which pgf90 2> /dev/null | sed 's/.*\///'),pgf90)
         # Portland Group (PGI) compiler
         F90C:=pgf90
      endif
   endif
   ifeq ($(shell which mpxlf90 2> /dev/null | sed 's/.*\///'),mpxlf90)
      # IBM compiler on AIX system
      CMP:=mpxlf90_r
      F90C:=xlf90_r
   endif
endif

ifeq ($(F90C),xlf90_r)
    # Flags for the IBM AIX compiler
    # We also define the "__NERSC" symbol to be passed to the preprocessor
    # through the compiler option "-WF,-D__NERSC"
    # -g -pg
    OPT:=-q64 -qsuffix=cpp=F90 -WF,-D__NERSC \
         $(SPRNG) $(NETCDF) -qsuffix=f=f90 -qfree=f90 -qinitauto \
        -qarch=auto -qtune=auto -qcache=auto -qstrict -qflttrap -O3 -Q -u \
        -qautodbl=dbl4 -I ~u4144/ntcc64/RS6000/mod $(OPT) $(HDF5_INCLUDE)

    # -O3 is recommended by IBM;  without -qstrict the compiler 
    # might bend some IEEE rules and one has to check carefully the result;
    # -qhot  improves nested loops 
    HOME_ETHIER:=/usr/common/homes/e/ethier
    PSPLINE:=-L~u4144/ntcc64/RS6000/lib -lpspline
    # Assume that the NAG library is available by default
    ifeq ($(HAVE_NAG),n)
       NAG_FILES:=fake_nag.o nag.o
       NAG:=-L./cmlib -lcmlib
       NAG_REPLACEMENT:=libcm
    else
       NAG_FILES:=
       NAG:=-L${NAG_ROOT}/lib -lnag
       NAG_REPLACEMENT:=
    endif
    ###LIB:=$(HOME_ETHIER)/work/timers/system_stats.o $(NETCDF) $(NAG) $(MASS) 
    LIB:=$(HOME_ETHIER)/work/timers/system_stats.o $(NETCDF) $(NAG_SMP) $(MASS) \
	$(LIB) ${HDF5} ${PSPLINE}
    #   $(LIB) $(HDF5_LIB) -lhdf5_fortran
    # If we use "gmake OPENMP=y" then we compile with mpxlf90_r and add
    # the -qsmp=omp option to take into account the OpenMP directives.
    CC:= cc_r
    CFLAGS:=-q64 -D_F_TO_C_NO_UNDERSCORE \
         $(NETCDF) \
        -qarch=pwr3 -qtune=pwr3 -qcache=auto -qstrict -qflttrap -O3 -Q
    SUPERLU_FILES:=c_superlu_pdgssv.o
    ifdef OPENMP
       OPT:=-qsmp=omp -qnosave $(OPT)
       CC:= mpcc_r -qsmp=omp
    endif
    ifdef DEBUG
       OPT:=-C -g $(OPT)
       CFLAGS:=-g $(CFLAGS)
    endif
    ifneq ($(ESSL),n)
       #####LIB=/usr/common/usg/timers/system_stats.o $(NETCDF) -lessl_r $(MASS)
       LIB:=$(LIB) -lessl_r
       FFT=fft_essl.o
    endif
endif

ifeq ($(F90C),lf95)
    # Flags for the Lahey compiler under Linux
    OPT:=-O --dbl --ap --pca --trace -Dflush_=flush
    CC:=mpicc
    CFLAGS:=-O -DAdd_
    NETCDF:=-L${NETCDFHOME}/lib -lnetcdf
    # We are probably running at PPPL so NAG is available and is the default
    ifeq ($(HAVE_NAG),n)
       NAG_FILES:=fake_nag.o nag.o
       NAG:=-L./cmlib -lcmlib
       NAG_REPLACEMENT:=libcm
    else
       NAG_FILES:=
       NAG:=-L${NAG_ROOT}/lib -lnag
       NAG_REPLACEMENT:=
    endif
    SPRNG:=-L${SPRNG_HOME}/lib -lsprng
    PSPLINE:=-L${NTCCHOME}/lib -lpspline -lportlib -lezcdf
    ifeq ($(DEBUG),y)
       OPT:=-g $(OPT) --warn --chk[aesu] --chkglobal
       CFLAGS:=-g $(CFLAGS)
    endif
    LIB:=$(NETCDF) $(NAG) $(PSPLINE) $(SPRNG)
    OPT:=$(OPT) -I${NETCDFHOME}/include -I${SPRNG_HOME}/include \
         -I${NTCCHOME}/mod
endif

ifeq ($(F90C),lfc)
    # Flags for the Lahey compiler under Linux
    OPT:=-O --dbl --ap --pca --trace -Dflush_=flush
    CC:=mpicc
    CFLAGS:=-O -DAdd_
    NETCDF:=-L${NETCDFHOME}/lib -lnetcdf
    # We are probably running at PPPL so NAG is available and is the default
    ifeq ($(HAVE_NAG),n)
       NAG_FILES:=fake_nag.o nag.o
       NAG:=-L./cmlib -lcmlib
       NAG_REPLACEMENT:=libcm
    else
       NAG_FILES:=
       NAG:=-L${NAG_ROOT} -lnag
       NAG_REPLACEMENT:=
    endif
    SPRNG:=-L${SPRNG_HOME}/lib -lsprng
    PSPLINE:=-L${NTCCHOME}/lib -lpspline -lportlib -lezcdf
    ifeq ($(DEBUG),y)
       OPT:=-g --warn --chk[aesu] --chkglobal $(OPT)
       CFLAGS:=-g $(CFLAGS)
    endif
    LIB:=$(NETCDF) $(NAG) $(PSPLINE) $(SPRNG)
    OPT:=$(OPT) -I${NETCDFHOME}/include -I${SPRNG_HOME}/include \
         -I${NTCCHOME}/mod
endif

ifeq ($(F90C),pgf90)
    # Flags for the PGI compilers under Linux
    OPT:=-O -D__PGF90 -Dflush_=flush -Kieee
    CC:=mpicc
    CFLAGS:=-O -c9x -DAdd_
    NETCDF:=-L${NETCDFHOME}/lib -lnetcdf
    SPRNG:=-L${SPRNG_HOME}/lib -lsprng
    PSPLINE:=-L${NTCCHOME}/lib -lpspline -lportlib -lezcdf
    ifeq ($(DEBUG),y)
       OPT:=-g $(OPT)
       CFLAGS:=-g $(CFLAGS)
    endif
    ifeq ($(HAVE_NAG),y)
       NAG_FILES:=
       NAG:=-L${NAG_ROOT}/lib -lnag_nag
       NAG_REPLACEMENT:=
    else
       NAG_FILES:=fake_nag.o nag.o
       NAG:=-L./cmlib -lcmlib
       NAG_REPLACEMENT:=libcm
    endif
    LIB:=$(NETCDF) $(NAG) $(PSPLINE) $(SPRNG)
    OPT:=$(OPT) -I${NETCDFHOME}/include -I${SPRNG_HOME}/include \
         -I${NTCCHOME}/mod
endif

ifeq ($(F90C),ftn)
   # Flags for Cray XT system
   #####ifeq ($(shell uname -n | cut -c-3),hop)
   #####ifeq ($(NERSC_HOST),hopper)
       ifeq ($(PE_ENV),PGI)   # Using PGI compiler on Hopper
          FFLAGS:= -r8 -Kieee -Dflush_=flush -module .
          OPT:=-fast
          COPT:=-fast
          CFLAGS:=
          OMPFLAG:=-mp=nonuma
          NOOMPFLAG:=
          SPRNG_LIB:=-L${SPRNG_DIR}/lib -lsprng
          SPRNG_INC:=-I${SPRNG_DIR}/include
          PSPLINE_INC:=-I${PSPLINE_DIR}/mod
          PSPLINE:=-L${PSPLINE_DIR}/lib -lpspline -lezcdf -L${NETCDF_DIR}/lib -lnetcdf
       else   # Using CRAY compiler
          #PSPLINE_DIR:=/project/projectdirs/mp19/software/cray
          FFLAGS:=-s real64 -Dflush_=flush -I. -static -e F -e m -rm
          OPT:=
          CFLAGS:=
          COPT:=
          OMPFLAG:=
          NOOMPFLAG:=-hnoomp
          SPRNG_LIB:=-L${SPRNG_DIR}/lib -lsprng
          SPRNG_INC:=-I${SPRNG_DIR}/include
          PSPLINE_INC:=-I${PSPLINE_DIR}/mod
          PSPLINE:=-L${PSPLINE_DIR}/lib -lpspline -lezcdf -L${NETCDF_DIR}/lib -lnetcdf
       endif
####   else
####     # Jaguarpf system at ORNL
####       FFLAGS:= -r8 -Kieee -Dflush_=flush -module .
####       OPT:=-fast
####       COPT:=-fast
####       CFLAGS:=
####       PSPLINE:=-module /apps/pspline/LINUX_pg62/mod \
####               -L/apps/pspline/LINUX_pg62/lib -lpspline -lezcdf -lportlib
####       SPRNG:=${SPRNG_LIB}
####       NETCDF:=${NETCDF_FLIB}
####       HDF5:=${HDF5_FLIB}
####   endif
   PETSC_FLIB:=-L${PETSC_DIR}/${PETSC_ARCH}/lib -lcraypetsc_cray_real
   FFTLIB:=
   NAG_FILES:=fake_nag.o nag.o
   NAG:=-L./cmlib -lcmlib
   NAG_REPLACEMENT:=libcm
   CC:=cc
   CFLAGS:= -DAdd_ ${SPRNG_INC}
###     -I$(SUPERLU_SRC)  ${SPRNG_LIB} ${NETCDF_C_LIB} ${HDF5_CLIB}

   FFLAGS:=$(FFLAGS) $(SPRNG_INC) $(NETCDF) $(PSPLINE_INC) $(HDF5)
endif

   ifeq ($(DEBUG),y)
     OPT:=-g  $(FFLAGS)
     COPT:= -g $(CFLAGS)
   else
     OPT:=$(OPT) $(FFLAGS)
     COPT:=$(COPT) $(CFLAGS)
   endif

   ifeq ($(OPENMP),y)
     CMD:=GTS_OMP
     OPT:=$(OMPFLAG) $(OPT)
     COPT:=$(OMPFLAG) $(COPT)
   else
     CMD:=GTS
     OPT:=$(NOOMPFLAG) $(OPT)
     COPT:=$(NOOMPFLAG) $(COPT) 
   endif
   CMD:=Coord_Map
   GSLLIB:= ${GSL} -lgslcblas
   LIB:=$(GSLLIB)  $(NETCDF) $(SPRNG_LIB) $(PSPLINE) $(HDF5) $(FFTLIB)
###     LIB:=$(NAG)  $(SUPERLU) $(NETCDF) $(SPRNG) $(PSPLINE) $(HDF5) $(FFTLIB) -lacml

##################################################################
# We add ".F90" to the list of suffixes to allow source files on which the
# co-processor will be run automatically.
.SUFFIXES: .o .f90 .F90 .F .c

#all : libcm $(CMD)

#export PETSC_ARCH=cray-xt4_O
#export PETSC_DIR=/project/projectdirs/mp19/petsc-3.0.0-p7

#petsc3.09 include ${PETSC_DIR}/conf/base
#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules

#PETSCFINCLUDE:= -I${PETSC_DIR}/include -I${PETSC_DIR} -I${PETSC_DIR}/${PETSC_ARCH}/include $(MPI_INCLUDE)
#SOLVER:=solver_petsc.o 

# List of all the object files needed to build the code
#OBJ:=module.o spl_profile.o ${SOLVER} main.o function_gg.o setup_v2.o Er_provider.o Theory_jb.o \
     load.o restart.o diagnosis.o snapshot.o charge_ion.o current_ion.o \
     charge_eon.o current_eon.o solve_dphidt.o poisson.o smooth.o \
     field.o pushi.o pushe.o shifti.o shifte.o collision.o collision_eon.o CTPandCELL.o \
     $(FFT) output.o escZ_march03.o check_eq.o view_distribution.o \
     view_orbit.o $(NAG_FILES) check_temperature.o get_fluxes.o get_3D_flow.o get_fluxes_e.o get_ZF_shear.o \
     open_files.o csr2css.o check_tracers.o create_directory.o weight_damping.o weight_damping_eon.o
#output3d.o
# selectmode.o volume.o 

OBJ:=mainC.o esiZ120813.o reflectometer.o supplementary.o
all: $(CMD)
$(CMD): $(OBJ)
	$(CC) -o $(CMD) $(OBJ) $(COPT) $(LIB)
#	$(CMP) -o $(CMD) $(OBJ) $(OPT) $(LIB)
###$(CMP) -o $(CMD) $(OBJ) $(OPT) $(LIB) ${PETSC_FORTRAN_LIB} ${PETSC_KSP_LIB}

#libcm :
#	(cd cmlib; $(MAKE) FC=$(F90C) FFLAGS="$(OPT)" ; cd ..)

#$(OBJ): module.o
$(OBJ):
#.f90.o :
#	$(CMP) $(OPT) -c $<

#.F90.o :
#	$(CMP) $(OPT) $(PETSCFINCLUDE) -c $<

#.F.o :
#	$(CMP) $(OPT) -c $<

.c.o   :
	$(CC) -c $(COPT) $(LIB) $<

# The following tag is meant to "clean" the directory by removing the
# executable along with all the object files created by the compilation 
# One only has to run:  gmake clean

cleangts:
	-rm -f $(CMD) $(OBJ) *.mod *.lst
#	(cd cmlib; $(MAKE) clean ; cd ..)
