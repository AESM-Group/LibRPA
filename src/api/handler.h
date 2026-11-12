#pragma once
#include "../../include/librpa_handler.h"
#include "../interface/mpi/mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Only for internal usage in the Fortran binding
LibrpaHandler* librpa_create_handler_fortran(MPI_Fint *f_comm);

#ifdef __cplusplus
}
#endif

