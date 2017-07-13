M : main.o global.o
	g++ -g -pg -o M main.cpp global.cpp global.h MySVM.h `pkg-config opencv2413 --libs --cflags`
