app=pg_learning

SOURCES=pg_learning.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=-g
INCLUDES=-I./ -I../ -I/usr/include/ -I/usr/include/torch/csrc/api/include/
OPT=-O3
LIBS=-lc10 -lc10_cuda -ltorch -lcaffe2_nvrtc -lcaffe2_observers -lcaffe2_detectron_ops_gpu -lcaffe2_module_test_dynamic -lshm
DEFINES=

CXXFLAGS=-std=c++17 -MD -pedantic -O3 -Wall -Wextra $(DEFINES) $(INCLUDES) $(OPT) $(DEBUG)
CXXLINKS=$(CXXFLAGS) $(LIBS)

COMPILER=clang++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXLINKS) $^ -o $@

%.o: %.cpp
	$(COMPILER) $(CXXFLAGS) -c $^

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
