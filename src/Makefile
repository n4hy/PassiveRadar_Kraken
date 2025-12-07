CXX = g++
CXXFLAGS = -O3 -fPIC -shared -Wall
TARGET_NLMS = libnlms_clutter_canceller.so
TARGET_DOPPLER = libdoppler_processing.so
SRC_NLMS = nlms_clutter_canceller.cpp
SRC_DOPPLER = doppler_processing.cpp

all: $(TARGET_NLMS) $(TARGET_DOPPLER)

$(TARGET_NLMS): $(SRC_NLMS)
	$(CXX) $(CXXFLAGS) -DLIBRARY_BUILD $(SRC_NLMS) -o $(TARGET_NLMS)

$(TARGET_DOPPLER): $(SRC_DOPPLER)
	$(CXX) $(CXXFLAGS) $(SRC_DOPPLER) -o $(TARGET_DOPPLER)

clean:
	rm -f $(TARGET_NLMS) $(TARGET_DOPPLER)
