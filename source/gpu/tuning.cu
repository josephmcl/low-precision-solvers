#include "gpu/tuning.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tuning {

table &current() {

    static table instance;
    return instance;
}

std::string file_name_for(std::string const &device_name) {

    std::string out;
    bool underscore = false;

    for (std::size_t i = 0; i != device_name.size(); ++i) {

        unsigned char const c = static_cast<unsigned char>(device_name[i]);
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
            underscore = false;
        }
        else if (!underscore && !out.empty()) {
            out.push_back('_');
            underscore = true;
        }
    }

    while (!out.empty() && out.back() == '_')
        out.pop_back();

    return out + ".conf";
}

/*  LPS_<KEY>, dots to underscores, uppercased. */
static std::string environment_name(std::string const &key) {

    std::string out = "LPS_";
    for (std::size_t i = 0; i != key.size(); ++i) {
        char const c = key[i];
        out.push_back((c == '.')? '_'
                     : static_cast<char>(std::toupper(
                           static_cast<unsigned char>(c))));
    }
    return out;
}

bool table::_load(std::string const &path) {

    std::ifstream in(path.c_str());
    if (!in)
        return false;

    std::string line;
    while (std::getline(in, line)) {

        std::size_t const hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);

        std::size_t const eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key   = line.substr(0, eq);
        std::string value = line.substr(eq + 1);

        auto trim = [](std::string &s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
                s.erase(s.begin());
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
                s.pop_back();
        };
        trim(key);
        trim(value);

        if (key.empty() || value.empty())
            continue;

        entry e;
        e.key   = key;
        e.value = std::atoi(value.c_str());
        _entries.push_back(e);
    }

    _source = path;
    return true;
}

void table::load(std::string const &device_name) {

    if (char const *explicit_path = std::getenv("LPS_TUNING")) {
        if (_load(explicit_path))
            return;
        std::cout << "[tuning] LPS_TUNING=" << explicit_path
                  << " could not be opened; falling back\n";
    }

    std::string const path = "tuning/" + file_name_for(device_name);
    if (_load(path))
        return;

    std::cout << "[tuning] no " << path
              << "; using built-in defaults, which were tuned for a different"
                 " device. Run the sweep and write that file before quoting"
                 " timings from this machine.\n";
}

int table::get(std::string const &key, int const fallback) const {

    /*  Environment first, so a sweep can override without editing the file. */
    if (char const *v = std::getenv(environment_name(key).c_str()))
        return std::atoi(v);

    for (std::size_t i = 0; i != _entries.size(); ++i) {
        if (_entries[i].key == key) {
            _entries[i].read = true;
            return _entries[i].value;
        }
    }

    return fallback;
}

std::vector<std::string> table::unused() const {

    std::vector<std::string> out;
    for (std::size_t i = 0; i != _entries.size(); ++i)
        if (!_entries[i].read)
            out.push_back(_entries[i].key);

    return out;
}

} /* namespace tuning */
