#include "kernel_registry.h"
#include <stdexcept>

void KernelRegistry::registerKernel(const std::string& name, KernelFactory factory) {
    factories_[name] = factory;
}

SpMMKernelPtr KernelRegistry::createKernel(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        throw std::runtime_error("Kernel not found: " + name);
    }
    return it->second();
}

std::vector<std::string> KernelRegistry::getAvailableKernels() const {
    std::vector<std::string> names;
    names.reserve(factories_.size());
    for (const auto& pair : factories_) {
        names.push_back(pair.first);
    }
    return names;
}

bool KernelRegistry::hasKernel(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}