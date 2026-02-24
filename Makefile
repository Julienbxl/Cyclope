TARGET     := Cyclope
SRC        := Cyclope.cu Hash.cu
OBJ        := $(SRC:.cu=.o)
CC         := nvcc
GENCODE    := -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17
LDFLAGS    := -lcudadevrt -cudart=static -lssl -lcrypto
all: $(TARGET)
$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)
%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@
clean:
	rm -f $(TARGET) $(OBJ)