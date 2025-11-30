CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Ilib -I/usr/include/eigen3
SRC = src/main.cpp
TARGET = main.o

all: $(TARGET)
format:
	clang-format -i src/*.cpp src/*.hpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)