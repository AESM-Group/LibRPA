#pragma once

/**
 * @file librpa_enums.h
 * @brief Enumeration types and constants for LibRPA.
 *
 * This file defines the enumeration types used throughout LibRPA for
 * parallel routing, time/frequency grids, and runtime controls.
 */

// C enums
#ifdef __cplusplus
extern "C" {
#endif

/** @brief Undefined or unset value for integer parameters. */
#define LIBRPA_UNSET -101

/** @brief Automatic selection value. LibRPA will choose appropriate setting. */
#define LIBRPA_AUTO -51

/** @brief Switch value for disabled/off state (equivalent to false). */
#define LIBRPA_SWITCH_OFF 0

/** @brief Switch value for enabled/on state (equivalent to true). */
#define LIBRPA_SWITCH_ON 1

/** @brief Verbose level for debug output. */
#define LIBRPA_VERBOSE_DEBUG 4

/** @brief Verbose level for informational messages. */
#define LIBRPA_VERBOSE_INFO 3

/** @brief Verbose level for warning messages. */
#define LIBRPA_VERBOSE_WARN 2

/** @brief Verbose level for essential results and critical/error messages only. */
#define LIBRPA_VERBOSE_CRITICAL 1

/** @brief Silent mode - no output. */
#define LIBRPA_VERBOSE_SILENT 0

// Reserved for future DFT code interfaces
// #define LIBRPA_KIND_AIMS 100
// #define LIBRPA_KIND_ABACUS 101
// #define LIBRPA_KIND_OPENMX 102
// #define LIBRPA_KIND_PYSCF 103

/**
 * @brief Ordering of real spherical harmonics inside one angular-momentum shell.
 */
typedef enum
{
    LIBRPA_ANGULAR_ORDER_UNSET = LIBRPA_UNSET,  ///< Unknown or not specified
    LIBRPA_ANGULAR_ORDER_NATURAL = 0,           ///< -l, -l+1, ..., l-1, l
    LIBRPA_ANGULAR_ORDER_ABS_PM = 1,            ///< 0, 1, -1, 2, -2, ...
    LIBRPA_ANGULAR_ORDER_OPENMX = 2,            ///< OpenMX ordering
    LIBRPA_ANGULAR_ORDER_PYSCF = 3,             ///< PySCF ordering
} LibrpaAngularOrder;

/**
 * @brief Coefficient-pair convention for one nonzero real spherical harmonic branch.
 *
 * For m < 0 this selects {a, b} in \f$i/\sqrt{2} [a Y_l^m - b Y_l^{-m}]\f$.
 * For m > 0 this selects {c, d} in \f$1/\sqrt{2} [c Y_l^m + d Y_l^{-m}]\f$.
 */
typedef enum
{
    LIBRPA_RSH_COEFF_UNSET = LIBRPA_UNSET,  ///< Unknown or not specified
    LIBRPA_RSH_COEFF_1_M = 0,               ///< {1, (-1)^m}
    LIBRPA_RSH_COEFF_M_1 = 1,               ///< {(-1)^m, 1}
} LibrpaRshCoeff;

/** Number of parallel routing types available. */
#define LIBRPA_ROUTING_COUNT 5

/**
 * @brief Parallel routing strategy for distributed memory calculations.
 *
 * Specifies how the computation is distributed across MPI processes.
 */
typedef enum
{
    LIBRPA_ROUTING_UNSET = LIBRPA_UNSET,  ///< Use default routing (unset)
    LIBRPA_ROUTING_AUTO = LIBRPA_AUTO,    ///< Automatically select optimal routing
    LIBRPA_ROUTING_RTAU = 0,              ///< Real-space tau (time) decomposition
    LIBRPA_ROUTING_ATOMPAIR = 1,          ///< Atom-pair parallelization
    LIBRPA_ROUTING_LIBRI = 2,             ///< Use LibRI for RI basis operations
} LibrpaParallelRouting;

/** Number of time/frequency grid types available. */
#define LIBRPA_TFGRID_COUNT 7

/**
 * @brief Type of time or frequency grid for integration.
 *
 * Different grid types offer different convergence properties
 * for RPA and GW calculations.
 */
typedef enum
{
    LIBRPA_TFGRID_UNSET = LIBRPA_UNSET,   ///< Use default grid (unset)
    LIBRPA_TFGRID_GAUSS_LEGENDRE = 0,      ///< Gauss-Legendre quadrature
    LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I = 1,   ///< Gauss-Chebyshev type I
    LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II = 2,  ///< Gauss-Chebyshev type II
    LIBRPA_TFGRID_MINIMAX = 3,             ///< Minimax grid
    LIBRPA_TFGRID_EVEN_SPACED = 4,         ///< Evenly spaced grid
    LIBRPA_TFGRID_EVEN_SPACED_TF = 5,      ///< Evenly spaced in time-frequency
    LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE = 6, ///< Finite GL interval plus transformed tail
} LibrpaTimeFreqGrid;

/** @brief Analytic continuation method for GW and QSGW. */
typedef enum
{
    LIBRPA_ANACON_PADE = 0,   ///< Thiele Pade with barycentric fallback
    LIBRPA_ANACON_AAA = 1,    ///< Adaptive barycentric rational approximation
    LIBRPA_ANACON_RIDGE = 2,  ///< Regularized rational coefficient fit (default)
} LibrpaAnalyContMethod;

/**
 * @brief Method to compute a quantity on the time/frequency grid.
 *
 * Several options use this type, one for each stage of a calculation.
 */
typedef enum
{
    LIBRPA_METHOD_UNSET = LIBRPA_UNSET,  ///< Use default method (unset)
    LIBRPA_METHOD_AUTO = LIBRPA_AUTO,    ///< Select from the grid type
    LIBRPA_METHOD_SPACE_TIME = 0,        ///< Real space and imaginary time
    LIBRPA_METHOD_CONVENTIONAL = 1,      ///< Reciprocal space and imaginary frequency
} LibrpaMethod;

/**
 * @brief Boolean switch type.
 *
 * Use LIBRPA_SWITCH_ON (1) or LIBRPA_SWITCH_OFF (0) to set.
 */
typedef int LibrpaSwitch;

/**
 * @brief Type of DFT code (reserved for future use).
 *
 * Currently reserved for identifying the source of wavefunction data.
 * Not yet implemented.
 */
typedef int LibrpaKind;

/**
 * @brief Verbosity level for runtime output.
 *
 * Controls the amount of information printed during computation.
 * Use one of LIBRPA_VERBOSE_* constants.
 */
typedef int LibrpaVerbose;

#ifdef __cplusplus
}
#endif
