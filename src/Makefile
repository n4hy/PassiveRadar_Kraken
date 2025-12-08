CXX = g++
CXXFLAGS = -O3 -fPIC -shared -Wall
TARGET_NLMS = libnlms_clutter_canceller.so
TARGET_DOPPLER = libdoppler_processing.so
TARGET_AOA = libaoa_processing.so
TARGET_RESAMPLER = libresampler.so
SRC_NLMS = nlms_clutter_canceller.cpp
SRC_DOPPLER = doppler_processing.cpp
SRC_AOA = aoa_processing.cpp
SRC_RESAMPLER = resampler.cpp

all: $(TARGET_NLMS) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)

$(TARGET_NLMS): $(SRC_NLMS)
	$(CXX) $(CXXFLAGS) -DLIBRARY_BUILD $(SRC_NLMS) -o $(TARGET_NLMS)

$(TARGET_DOPPLER): $(SRC_DOPPLER)
	$(CXX) $(CXXFLAGS) $(SRC_DOPPLER) -o $(TARGET_DOPPLER)

$(TARGET_AOA): $(SRC_AOA)
	$(CXX) $(CXXFLAGS) $(SRC_AOA) -o $(TARGET_AOA)

$(TARGET_RESAMPLER): $(SRC_RESAMPLER)
	$(CXX) $(CXXFLAGS) $(SRC_RESAMPLER) -o $(TARGET_RESAMPLER)

clean:
	rm -f $(TARGET_NLMS) $(TARGET_DOPPLER) $(TARGET_AOA) $(TARGET_RESAMPLER)
