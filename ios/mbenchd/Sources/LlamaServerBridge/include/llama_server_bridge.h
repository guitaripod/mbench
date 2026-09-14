#ifndef LLAMA_SERVER_BRIDGE_H
#define LLAMA_SERVER_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t mb_server_run(int32_t argc, char * const * argv);
void mb_server_stop(void);
const char * mb_llama_commit(void);
int32_t mb_llama_build_number(void);

#ifdef __cplusplus
}
#endif

#endif
