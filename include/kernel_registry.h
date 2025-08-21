#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>
#include "kernel_interface.h"

class KernelRegistry {
public:
    using KernelFactory = std::function<SpMMKernelPtr()>;
    
    static KernelRegistry& getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    
    void registerKernel(const std::string& name, KernelFactory factory);
    
    SpMMKernelPtr createKernel(const std::string& name) const;
    
    std::vector<std::string> getAvailableKernels() const;
    
    bool hasKernel(const std::string& name) const;

private:
    KernelRegistry() = default;
    std::unordered_map<std::string, KernelFactory> factories_;
};

#define REGISTER_KERNEL(name, KernelClass) \
    static bool registered_##KernelClass = []() { \
        KernelRegistry::getInstance().registerKernel( \
            name, \
            []() -> SpMMKernelPtr { return std::make_unique<KernelClass>(); } \
        ); \
        return true; \
    }()

#endif // KERNEL_REGISTRY_H