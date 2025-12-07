CXX = g++
CXXFLAGS = -O3 -fPIC -shared -Wall
TARGET = libnlms_clutter_canceller.so
SRC = nlms_clutter_canceller.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -DLIBRARY_BUILD $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
