CC = /usr/local/cuda-7.5/bin/nvcc

test: main.cu utils.o impl_utils.o impl1.o impl2.o impl3.o
	$(CC) main.cu utils.o impl_utils.o impl1.o impl2.o impl3.o -O3 -arch=sm_30 -o test

utils.o: utils.cu utils.h
	$(CC) utils.cu -dc

impl_utils.o: impl_utils.cu implementation.h
	$(CC) impl_utils.cu -dc

impl1.o: impl1.cu implementation.h 
	$(CC) impl1.cu -dc

impl2.o: impl2.cu implementation.h
	$(CC) impl2.cu -dc

impl3.o: impl3.cu implementation.h
	$(CC) impl3.cu -dc

run: test
	./test --input in.txt --output out.txt --bsize 1024 --bcount 2 --method 1

run2: test
	./test --input in.txt --output out.txt --bsize 1024 --bcount 2 --method 2

run3: test
	./test --input input/words.txt --output out.txt --bsize 1024 --bcount 2 --method 1

clean:
	rm -f *.o test
