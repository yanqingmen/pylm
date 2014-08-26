CC = g++
WEIGHTTYPE = float
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -lm -O2 -Wall -funroll-loops -ffast-math
TFLAGS = -g
#CFLAGS = -lm -O2 -Wall

all: rnnlmlib.o rnnlm
test: rnnlmtest

rnnlmlib.o : rnnlmlib.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlmlib.cpp

rnnlm : rnnlm.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlm.cpp rnnlmlib.o -o rnnlm

rnnlmtest: rnnlmtest.cpp rnnlmlib.o
	$(CC) $(CFLAGS) $(TFLAGS) $(OPT_DEF) rnnlmtest.cpp rnnlmlib.o -o rnnlmtest

clean:
	rm -rf *.o rnnlm rnnlmtest
