#pragma once

// Do not declare new variables/functions in this header.
// Only add internal headers to this to collect global variables/functions
// that the users might want to take a look

#include "librpa_enums.h"
#include "../interface/mpi/mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Only for internal usage of the Fortran binding
void librpa_init_global_fortran(MPI_Fint *f_comm, LibrpaSwitch switch_redirect_stdout = LIBRPA_SWITCH_OFF, const char *redirect_path = "stdout",
                                LibrpaSwitch switch_process_output = LIBRPA_SWITCH_ON);

#ifdef __cplusplus
}
#endif
