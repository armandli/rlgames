app=test_discrete_dist

SOURCES=test_discrete_dist.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

#NOTE: tests are using the official torch library install, not the nightly

DEBUG=-g
INCLUDES=-I./
OPT=-O3
LIBS=

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
