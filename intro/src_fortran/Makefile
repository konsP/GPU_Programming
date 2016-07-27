# 
#Makefile
#

MF= Makefile

EXE= intro

SRC= array_negate.cuf  gpu_utils.cuf  intro.cuf  


CC=	pgfortran
CFLAGS= -Mcuda=cc20 -O3 


.SUFFIXES:
.SUFFIXES: .cuf  .o

OBJ=	$(SRC:.cuf=.o)


.cuf.o:
	$(CC) $(CFLAGS) -c $<

default: $(EXE)

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) *.mod core
