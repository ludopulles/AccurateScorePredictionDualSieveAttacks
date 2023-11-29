# This Makefile compiles the C-parts of the python program dual_utils.py
.POSIX:

all: bdd_sample.so unif_sample.so

clean:
	-rm -f bdd_sample.so unif_sample.so

CXX=g++
CFLAGS=
CXXFLAGS=-Wall --std=c++17

bdd_sample.so: bdd_sample.cpp
	$(CXX) $(CXXFLAGS) -fPIC -Ofast -march=native -shared -o bdd_sample.so bdd_sample.cpp $(LIBS)

unif_sample.so: unif_sample.c
	$(CC) $(CFLAGS) -fPIC -Ofast -march=native -shared -o unif_sample.so unif_sample.c $(LIBS)
