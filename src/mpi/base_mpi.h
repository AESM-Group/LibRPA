#pragma once

#include <algorithm>
#include <cstddef>
#include <complex>
#include <string>
#include <vector>

#include "../interface/mpi/mpi.h"
#include "librpa_enums.h"

namespace librpa_int
{

// traits to decide MPI_Datatype for communication
template <typename T>
struct mpi_datatype;
template <>
struct mpi_datatype<int>
{
    static inline const MPI_Datatype value = MPI_INT;
};
template <>
struct mpi_datatype<float>
{
    static inline const MPI_Datatype value = MPI_FLOAT;
};
template <>
struct mpi_datatype<double>
{
    static inline const MPI_Datatype value = MPI_DOUBLE;
};
template <>
struct mpi_datatype<long>
{
    static inline const MPI_Datatype value = MPI_LONG;
};
template <>
struct mpi_datatype<unsigned long>
{
    static inline const MPI_Datatype value = MPI_UNSIGNED_LONG;
};
template <>
struct mpi_datatype<std::complex<float>>
{
    static inline const MPI_Datatype value = MPI_C_FLOAT_COMPLEX;
};
template <>
struct mpi_datatype<std::complex<double>>
{
    static inline const MPI_Datatype value = MPI_C_DOUBLE_COMPLEX;
};

extern const char *mpi_procname_uninit;

class MpiCommHandler
{
public:
private:
    bool initialized_;
    bool comm_set_;

public:
    MPI_Comm comm;
    int myid;
    int nprocs;
    std::string procname;

public:
    MpiCommHandler();
    MpiCommHandler(MPI_Comm comm_in, bool init_on_construct = false);
    ~MpiCommHandler(){};
    void init();
    void reset_comm();
    void reset_comm(MPI_Comm comm_in, bool init_on_reset = false);
    // Free the wrapped communicator.
    // It should not be called within the destructor,
    // since the MPI communicator may be still used by other handlers.
    void free_comm();
    bool is_root() const { return initialized_ && this->myid == 0; }
    void barrier() const;
    std::string str() const;
    bool is_initialized() const { return initialized_; }
    void check_initialized() const;

    // T1 is for cases where MPI_IN_PLACE is used at sendbuf
    template <typename T1, typename T2>
    inline void allreduce(const T1 *sendbuf, T2 *recvbuf, int count, MPI_Op op) const
    {
        MPI_Allreduce(sendbuf, recvbuf, count, mpi_datatype<T2>::value, op, this->comm);
    }

    template <typename T1, typename T2>
    inline void allgather(const T1 *sendbuf, int sendcount, T2 *recvbuf, int recvcount) const
    {
        MPI_Allgather(sendbuf, sendcount, mpi_datatype<T1>::value, recvbuf, recvcount,
                      mpi_datatype<T2>::value, this->comm);
    }

    template <typename T1, typename T2>
    inline void reduce(const T1 *sendbuf, T2 *recvbuf, int count, int root, MPI_Op op) const
    {
        if (sendbuf == MPI_IN_PLACE && myid != root)
            MPI_Reduce(recvbuf, recvbuf, count, mpi_datatype<T2>::value, op, root, this->comm);
        else
            MPI_Reduce(sendbuf, recvbuf, count, mpi_datatype<T2>::value, op, root, this->comm);
    }

    template <typename T>
    inline void bcast(T *buf, int count, int root) const
    {
        MPI_Bcast(buf, count, mpi_datatype<T>::value, root, this->comm);
    }

    //! Broadcast a buffer that may exceed the MPI int count limit in slices.
    template <typename T>
    void bcast_large(T *buf, std::size_t count, int root) const
    {
        constexpr std::size_t chunk = static_cast<std::size_t>(1) << 28;
        for (std::size_t offset = 0; offset < count; offset += chunk)
        {
            const int slice = static_cast<int>(std::min(chunk, count - offset));
            MPI_Bcast(buf + offset, slice, mpi_datatype<T>::value, root, this->comm);
        }
    }

    // void allreduce_matrix(matrix &mat_send, matrix &mat_recv) const;
    // void allreduce_ComplexMatrix(ComplexMatrix &cmat_send, ComplexMatrix & cmat_recv) const;
    // void reduce_matrix(matrix &mat_send, matrix & cmat_recv, int root) const;
    // void reduce_ComplexMatrix(ComplexMatrix &cmat_send, ComplexMatrix & cmat_recv, int root)
    // const;
};

// extern const std::string parallel_routing_notes[LIBRPA_ROUTING_COUNT];

// extern ParallelRouting parallel_routing;

// void set_parallel_routing(const std::string &option, const int &atpais_num, const int &Rt_num,
// ParallelRouting &routing);

//! Return the actual routing inside LibRPA when auto is selected
LibrpaParallelRouting decide_auto_routing(const int n_atoms, const int Rt_num);

//! Wrapper of MPI_Comm_rank
int get_mpi_rank(const MPI_Comm &comm);

//! Wrapper of MPI_Comm_size
int get_mpi_size(const MPI_Comm &comm);

//! Return the lowest rank that owns a distributed object, or -1 if none owns it.
int find_mpi_owner_rank(bool owns_object, const MPI_Comm &comm);

//! task dispatchers, implemented single index and double indices versions. Ending indices are
//! excluded
std::vector<int> dispatcher(int ist, int ied, unsigned myid, unsigned size, bool sequential);
std::vector<std::pair<int, int>> dispatcher(int i1st, int i1ed, int i2st, int i2ed, unsigned myid,
                                            unsigned size, bool sequential, bool favor_1st);

std::vector<unsigned> dispatcher_balanced_counts(unsigned dist, const std::vector<int> &weights);

//! Task dispatchers with load balance, implemented single index version. Ending indices are
//! excluded Larger weight, smaller size of returned container
std::vector<int> dispatcher_balanced(int ist, int ied, const std::vector<int> &weights,
                                     unsigned myid, bool sequential);
//! This overload gathers weights on comm before dispatching.
std::vector<int> dispatcher_balanced(int ist, int ied, int weight, bool sequential,
                                     const MPI_Comm comm);

std::vector<std::pair<int, int>> pick_upper_triangular_tasks(std::vector<int> list_row,
                                                             std::vector<int> list_col);
std::vector<std::pair<int, int>> dispatch_upper_triangular_tasks(const int &natoms, const int &myid,
                                                                 const int &nprows,
                                                                 const int &npcols,
                                                                 const int &myprow,
                                                                 const int &mypcol);

/*!
 * @brief find duplicate pairs across all processes
 * @param n: the maximum index in the ordered pair
 * @param ordered_pairs: the pairs on each process
 * @return the pairs to be removed in each process to ensure that one pair appears only once across
 * the processes in comm
 */
std::vector<std::pair<int, int>> find_duplicate_ordered_pair(
    int n, const std::vector<std::pair<int, int>> &ordered_pairs, const MPI_Comm &comm);

template <typename T>
std::vector<T> dispatch_vector(std::vector<T> world_vec, unsigned myid, unsigned size,
                               bool sequential)
{
    std::vector<int> ids = dispatcher(0, world_vec.size(), myid, size, sequential);
    std::vector<T> local_vec;
    for (auto id : ids) local_vec.push_back(world_vec[id]);
    return local_vec;
}

template <typename T1, typename T2>
std::vector<std::pair<T1, T2>> dispatch_vector_prod(const std::vector<T1> &vec1,
                                                    const std::vector<T2> &vec2, unsigned myid,
                                                    unsigned size, bool sequential, bool favor_1st)
{
    std::vector<std::pair<int, int>> ids =
        dispatcher(0, int(vec1.size()), 0, int(vec2.size()), myid, size, sequential, favor_1st);
    std::vector<std::pair<T1, T2>> local_vec;
    for (auto id : ids) local_vec.push_back({vec1[id.first], vec2[id.second]});
    return local_vec;
}

} /* end of namespace librpa_int */
