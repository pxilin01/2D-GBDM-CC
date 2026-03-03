oper : maingray-cstaper.o alloc.o complex.o franuni.o fft.o sinc.o
	mpicc -o oper maingray-cstaper.o alloc.o complex.o franuni.o fft.o sinc.o -lm -fopenmp
maingray-cstaper.o : maingray-cstaper.c
	mpicc -c maingray-cstaper.c -fopenmp
alloc.o : alloc.c
	gcc -c alloc.c
complex.o : complex.c
	gcc -c complex.c
franuni.o : franuni.c
	gcc -c franuni.c
fft.o : fft.c
	gcc -c fft.c
sinc.o : sinc.c
	gcc -c sinc.c

clean:
	rm oper maingray-cstaper.o alloc.o complex.o franuni.o fft.o sinc.o
