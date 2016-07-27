MF=	Makefile

CC=	pgcc
CFLAGS=  -O3 -ta=nvidia -Minfo=all 

LFLAGS=	-lm

EXE=	image

SRC= \
	image.c \
	cio.c

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .c .o

OBJ=	$(SRC:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

clean:
	rm -f $(OBJ) $(EXE) core
