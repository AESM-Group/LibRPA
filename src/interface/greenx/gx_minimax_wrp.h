#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void gx_minimax_grid_frequency_wrp(int *num_points, double *e_min, double *e_max,
                                   double *omega_points, double *omega_weights,
                                   int *ierr);

void gx_minimax_grid_wrp(int *num_points, double *e_min, double *e_max,
                         double *tau_points, double *tau_weights,
                         double *omega_points, double *omega_weights,
                         double *cosft_wt, double *cosft_tw, double *sinft_wt,
                         double max_errors[3], double *cosft_duality_error, int *ierr,
                         int *bare_cos_sin_weights, double *regularization,
                         double *sinft_tw, double *sinft_duality_error,
                         double *max_error_sin_wt);

#ifdef __cplusplus
}
#endif
