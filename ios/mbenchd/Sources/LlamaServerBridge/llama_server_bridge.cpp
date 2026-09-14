#include "llama_server_bridge.h"

#include <vector>

int  llama_server(int argc, char ** argv);
void llama_server_terminate();

int          llama_build_number(void);
const char * llama_commit(void);

extern "C" int32_t mb_server_run(int32_t argc, char * const * argv) {
    std::vector<char *> args(argv, argv + argc);
    return llama_server(static_cast<int>(argc), args.data());
}

extern "C" void mb_server_stop(void) {
    llama_server_terminate();
}

extern "C" const char * mb_llama_commit(void) {
    return llama_commit();
}

extern "C" int32_t mb_llama_build_number(void) {
    return llama_build_number();
}
