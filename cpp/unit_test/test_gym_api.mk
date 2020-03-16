app=test_gym_api

SOURCES=test_gym_api.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=-g
INCLUDES=-I./ -I../ -I/usr/include/
OPT=-O3
LIBS=-lcurl -lgtest -lgtest_main
DEFINES=

CXXFLAGS=-std=c++17 -MD -pedantic -pedantic-errors -O3 -Wall -Wextra $(DEFINES) $(INCLUDES) $(OPT) $(DEBUG)
CXXLINKS=$(CXXFLAGS) $(LIBS)

COMPILER=clang++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXLINKS) $^ -o $@

%.o: %.cpp
	$(COMPILER) $(CXXFLAGS) -c $^

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
