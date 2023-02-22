#include "utils.h"

bool file_exists(std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}


Utils::Utils()
{

}
