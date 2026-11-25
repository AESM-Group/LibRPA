#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "../interface/mpi/mpi.h"
#include "../utils/error.h"

namespace librpa_int
{

template <typename T>
class ShmMpiHandler
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "MPI shared-memory windows store trivially copyable data only");

public:
    ShmMpiHandler() = default;

    ShmMpiHandler(MPI_Comm comm, std::size_t local_count) { allocate(comm, local_count); }

    ShmMpiHandler(const ShmMpiHandler &) = delete;
    ShmMpiHandler &operator=(const ShmMpiHandler &) = delete;

    ShmMpiHandler(ShmMpiHandler &&other) noexcept { move_from(other); }

    ShmMpiHandler &operator=(ShmMpiHandler &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            move_from(other);
        }
        return *this;
    }

    ~ShmMpiHandler() { reset(); }

    void allocate(MPI_Comm comm, std::size_t local_count)
    {
        reset();

        if (comm == MPI_COMM_NULL)
            throw LIBRPA_RUNTIME_ERROR("shared-memory window needs a valid communicator");
        if (local_count >
            static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(T))
            throw LIBRPA_RUNTIME_ERROR("shared-memory window is too large for MPI_Aint");

        comm_ = comm;
        local_count_ = local_count;
        MPI_Comm_rank(comm_, &myid_);
        MPI_Comm_size(comm_, &nprocs_);

        void *base = nullptr;
        const auto nbytes = static_cast<MPI_Aint>(local_count * sizeof(T));
        int ierr = MPI_Win_allocate_shared(nbytes, static_cast<int>(sizeof(T)), MPI_INFO_NULL,
                                           comm_, &base, &win_);
        if (ierr != MPI_SUCCESS) throw LIBRPA_RUNTIME_ERROR("MPI_Win_allocate_shared failed");

        local_data_ = static_cast<T *>(base);
        ptrs_.assign(nprocs_, nullptr);
        counts_.assign(nprocs_, 0);

        for (int rank = 0; rank != nprocs_; ++rank)
        {
            MPI_Aint rank_nbytes = 0;
            int disp_unit = 0;
            void *rank_base = nullptr;
            ierr = MPI_Win_shared_query(win_, rank, &rank_nbytes, &disp_unit, &rank_base);
            if (ierr != MPI_SUCCESS) throw LIBRPA_RUNTIME_ERROR("MPI_Win_shared_query failed");
            if (disp_unit != static_cast<int>(sizeof(T)) || rank_nbytes % sizeof(T) != 0)
                throw LIBRPA_RUNTIME_ERROR("unexpected shared-memory segment layout");
            ptrs_[rank] = static_cast<T *>(rank_base);
            counts_[rank] = static_cast<std::size_t>(rank_nbytes / sizeof(T));
        }

        ierr = MPI_Win_lock_all(0, win_);
        if (ierr != MPI_SUCCESS) throw LIBRPA_RUNTIME_ERROR("MPI_Win_lock_all failed");
        locked_ = true;
    }

    void reset() noexcept
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && win_ != MPI_WIN_NULL)
        {
            if (locked_) MPI_Win_unlock_all(win_);
            MPI_Win_free(&win_);
        }
        win_ = MPI_WIN_NULL;
        comm_ = MPI_COMM_NULL;
        local_data_ = nullptr;
        local_count_ = 0;
        myid_ = 0;
        nprocs_ = 0;
        locked_ = false;
        ptrs_.clear();
        counts_.clear();
    }

    // One barrier sync; split acquire/release if tensor kernels need overlap.
    void sync() const
    {
        MPI_Win_sync(win_);
        MPI_Barrier(comm_);
    }

    T *local_data() noexcept { return local_data_; }
    const T *local_data() const noexcept { return local_data_; }
    std::size_t local_count() const noexcept { return local_count_; }

    T *data(int rank) noexcept { return ptrs_[rank]; }
    const T *data(int rank) const noexcept { return ptrs_[rank]; }
    std::size_t count(int rank) const noexcept { return counts_[rank]; }
    const std::vector<std::size_t> &counts() const noexcept { return counts_; }

    int myid() const noexcept { return myid_; }
    int nprocs() const noexcept { return nprocs_; }
    MPI_Comm comm() const noexcept { return comm_; }
    MPI_Win win() const noexcept { return win_; }

private:
    void move_from(ShmMpiHandler &other) noexcept
    {
        win_ = other.win_;
        comm_ = other.comm_;
        local_data_ = other.local_data_;
        local_count_ = other.local_count_;
        myid_ = other.myid_;
        nprocs_ = other.nprocs_;
        locked_ = other.locked_;
        ptrs_ = std::move(other.ptrs_);
        counts_ = std::move(other.counts_);

        other.win_ = MPI_WIN_NULL;
        other.comm_ = MPI_COMM_NULL;
        other.local_data_ = nullptr;
        other.local_count_ = 0;
        other.myid_ = 0;
        other.nprocs_ = 0;
        other.locked_ = false;
    }

    MPI_Win win_ = MPI_WIN_NULL;
    MPI_Comm comm_ = MPI_COMM_NULL;
    T *local_data_ = nullptr;
    std::size_t local_count_ = 0;
    int myid_ = 0;
    int nprocs_ = 0;
    bool locked_ = false;
    std::vector<T *> ptrs_;
    std::vector<std::size_t> counts_;
};

}  // namespace librpa_int
