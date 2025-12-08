CXX = g++
CXXFLAGS = -O3 -fPIC -shared -Wall
TARGET_ECA = libeca_b_clutter_canceller.so
TARGET_DOPPLER = libdoppler_processing.so
TARGET_AOA = libaoa_processing.so
TARGET_RESAMPLER = libresampler.so
SRC_ECA = eca_b_clutter_canceller.cpp
SRC_DOPPLER = doppler_processing.cpp
SRC_AOA = aoa_processing.cpp
SRC_RESAMPLER = resampler.cpp

all: $(TARGET_ECA) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)

$(TARGET_ECA): $(SRC_ECA)
	$(CXX) $(CXXFLAGS) $(SRC_ECA) -o $(TARGET_ECA)

$(TARGET_DOPPLER): $(SRC_DOPPLER)
	$(CXX) $(CXXFLAGS) $(SRC_DOPPLER) -lfftw3f -o $(TARGET_DOPPLER)

$(TARGET_AOA): $(SRC_AOA)
	$(CXX) $(CXXFLAGS) $(SRC_AOA) -o $(TARGET_AOA)

$(TARGET_RESAMPLER): $(SRC_RESAMPLER)
	$(CXX) $(CXXFLAGS) $(SRC_RESAMPLER) -o $(TARGET_RESAMPLER)

clean:
	rm -f $(TARGET_ECA) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)
