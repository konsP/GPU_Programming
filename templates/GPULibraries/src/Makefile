#
# Makefile for SPD Linear Algebra benchmark
#
# see matinv.c for more info
#
# Alan Gray, EPCC, August 2014
#


CC=	icc
CFLAGS= -O3 -fopenmp 

MKLROOT=/opt/intel/composer_xe_2013_sp1.2.144/mkl
MAGMAROOT=/home/h018/agray3/packages/magma/magma-1.5.0-install
CUDAROOT=/usr/local/cuda

MKLFLAGS=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm 

NVCC=	nvcc
NVCFLAGS= -O3 -arch=sm_21 -DHAVE_CUBLAS -DNOCHANGE -I $(MAGMAROOT)/include
LFLAGS =   -L $(MAGMAROOT)/lib -lmagma $(MKLFLAGS) -L $(CUDAROOT)/lib64 -lcublas -lcudart -lgomp

MF= Makefile

EXE= matinv

SRC= matinv.c 
NVSRC= matinv_gpu.cu 

.SUFFIXES:
.SUFFIXES: .c .cu .o

OBJ=	$(SRC:.c=.o)

NVOBJ=	$(NVSRC:.cu=.o)

.c.o: 
	$(CC) $(CFLAGS) -c $<

.cu.o: 
	$(NVCC) $(NVCFLAGS) -c $<


all:	$(EXE)

$(EXE):	$(OBJ) $(NVOBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(NVOBJ) $(LFLAGS)

$(OBJ):	$(MF) 

$(NVOBJ): $(MF) 

clean:
	rm -f $(OBJ) $(NVOBJ) $(EXE) core

