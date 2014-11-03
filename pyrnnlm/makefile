CC = g++
WEIGHTTYPE = float
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -lm -O2 -Wall -funroll-loops -ffast-math -fPIC
TFLAGS = -g
INCFLAGS = -I/usr/include/python2.7 -I./ext_codes -I./rnnlm
LINKFLAGS = -lpython2.7
OBJS = ./obj
#CFLAGS = -lm -O2 -Wall

all: ./python_ext/_pyrnnlm.so

./python_ext/_pyrnnlm.so : $(OBJS)/rnnlmlib.o $(OBJS)/sentbuffer.o $(OBJS)/rnnlmlib_ext.o $(OBJS)/pyrnnlm_wrap.o
	$(CC) $(CFLAGS) $(INCFLAGS) $(OPT_DEF) -shared $^ -o $@ $(LINKFLAGS)

$(OBJS)/rnnlmlib.o : ./rnnlm/rnnlmlib.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) $(OPT_DEF) -c $^ -o $@

$(OBJS)/sentbuffer.o : ./ext_codes/sentbuffer.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) $(OPT_DEF) -c $^ -o $@

$(OBJS)/rnnlmlib_ext.o : ./ext_codes/rnnlmlib_ext.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) $(OPT_DEF) -c $^ -o $@

$(OBJS)/pyrnnlm_wrap.o : ./swig/pyrnnlm_wrap.cxx
	$(CC) $(CFLAGS) $(INCFLAGS) $(OPT_DEF) -c $^ -o $@



clean:
	rm -rf *.o rnnlm rnnlmtest
