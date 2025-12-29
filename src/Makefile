CXX = g++
CXXFLAGS = -O3 -fPIC -shared -Wall -march=native
INCLUDES = -I. -Ioptmath

TARGET_ECA = libkraken_eca_b_clutter_canceller.so
TARGET_DOPPLER = libkraken_doppler_processing.so
TARGET_AOA = libkraken_aoa_processing.so
TARGET_RESAMPLER = libkraken_resampler.so

SRC_ECA = eca_b_clutter_canceller.cpp
SRC_DOPPLER = doppler_processing.cpp
SRC_AOA = aoa_processing.cpp
SRC_RESAMPLER = resampler.cpp
SRC_NEON = optmath/neon_kernels.cpp

all: $(TARGET_ECA) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)

$(TARGET_ECA): $(SRC_ECA) $(SRC_NEON)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_ECA) $(SRC_NEON) -o $(TARGET_ECA)

$(TARGET_DOPPLER): $(SRC_DOPPLER)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_DOPPLER) -lfftw3f -o $(TARGET_DOPPLER)

$(TARGET_AOA): $(SRC_AOA)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_AOA) -o $(TARGET_AOA)

$(TARGET_RESAMPLER): $(SRC_RESAMPLER) $(SRC_NEON)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_RESAMPLER) $(SRC_NEON) -o $(TARGET_RESAMPLER)

clean:
	rm -f $(TARGET_ECA) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)
