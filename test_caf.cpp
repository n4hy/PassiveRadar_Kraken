#include <iostream>
#include <complex>
#include <vector>

extern "C" {
    void* caf_create(int n);
    void caf_destroy(void* p);
    void caf_process(void* p, const float* r, const float* s, float* o);
}

int main() {
    std::cout << "Creating CAF..." << std::endl;
    void* obj = caf_create(4096);
    std::cout << "Created: " << obj << std::endl;

    std::vector<std::complex<float>> ref(4096);
    std::vector<std::complex<float>> surv(4096);
    std::vector<std::complex<float>> out(4096);

    std::cout << "Processing..." << std::endl;
    caf_process(obj, (float*)ref.data(), (float*)surv.data(), (float*)out.data());
    std::cout << "Done." << std::endl;

    caf_destroy(obj);
    return 0;
}
