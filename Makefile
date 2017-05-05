CC = /usr/local/cuda-7.5/bin/nvcc

test: *.cu
	$(CC) -std=c++11 utils.cu impl1.cu impl2.cu impl3.cu main.cu -O3 -arch=sm_30 -o test

run: test
	./test --input in.txt --output out.txt --bsize 1024 --bcount 2 --method 1

clean:
	rm -f *.o test
