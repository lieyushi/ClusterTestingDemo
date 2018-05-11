CC = g++

CFLAGS = 

TARGET = main

RM = rm

RM_F = rm -f

STD = -std=c++11

MP = -fopenmp -O3

DICTORY = -I/usr/local/include

SOURCE = Evrot.cpp Initialization.cpp SpectralClustering.cpp main.cpp

HEADER = Evrot.h Initialization.h SpectralClustering.h

OBJECT = $(SOURCE:.cpp = .o)

all: $(SOURCE) $(TARGET)

clean: 
	$(RM) $(TARGET)

$(TARGET): $(OBJECT)
	$(CC) $(OBJECT) $(STD) $(MP) -o $@

.cpp.o: $(HEADER)
	$(CC) $(STD) $(CFLAGS) $(DICTORY) $(MP) $< -c -o $@
