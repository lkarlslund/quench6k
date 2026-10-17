// Runtime kernel variant table (see variants.cuh).

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "variants.cuh"

namespace {

std::mutex mutex;
std::map<std::string, int>* table_ptr = nullptr;

// The table, filled from QUENCH6K_VARIANTS on first use. Callers hold the mutex.
std::map<std::string, int>& table() {
    if (table_ptr == nullptr) {
        table_ptr = new std::map<std::string, int>();
        if (const char* env = std::getenv("QUENCH6K_VARIANTS")) {
            std::string s(env);
            size_t pos = 0;
            while (pos < s.size()) {
                size_t end = s.find(',', pos);
                if (end == std::string::npos) end = s.size();
                const std::string item = s.substr(pos, end - pos);
                const size_t eq = item.find('=');
                if (eq != std::string::npos) (*table_ptr)[item.substr(0, eq)] = std::atoi(item.c_str() + eq + 1);
                pos = end + 1;
            }
        }
    }
    return *table_ptr;
}

}  // namespace

int sq_variant(const char* key, int fallback) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto& t = table();
    const auto it = t.find(key);
    return it == t.end() ? fallback : it->second;
}

extern "C" void quench6k_set_variant(const char* key, int value) {
    std::lock_guard<std::mutex> lock(mutex);
    table()[key] = value;
}

extern "C" int quench6k_variant_get(const char* key, int fallback) { return sq_variant(key, fallback); }

extern "C" void quench6k_clear_variants() {
    std::lock_guard<std::mutex> lock(mutex);
    table().clear();
}
