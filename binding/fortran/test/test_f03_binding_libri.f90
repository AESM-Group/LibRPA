program test_f03_binding_libri

   use mpi
   use librpa_f03
   implicit none

   ! Example dimensions
   integer, parameter :: nspins = 1
   integer, parameter :: nk1 = 3, nk2 = 1, nk3 = 1
   integer, parameter :: nkpts = nk1 * nk2 * nk3
   integer, parameter :: nstates = 3
   integer, parameter :: nbasis = nstates
   integer, parameter :: naux = 9
   integer, parameter :: natoms = 2
   integer, parameter :: nfreq = 8
   character(len=*), parameter :: prog = "test_f03_binding"
   real(dp), parameter :: pi = atan(1.0_dp) * 4.0

   integer :: i, j, ierr, ispin, ik, ik1, ik2, ik3, istate, ibasis, ia1, ia2, r(3)
   type(LibrpaHandler) :: h
   type(LibrpaOptions) :: opts
   real(dp), allocatable :: wg(:,:,:), ekb(:,:,:), ri_coeff(:,:,:), omegas(:), df(:)
   real(dp) :: latt(3, 3), recplatt(3, 3), posi_cart(3, natoms), kpoints(3,nkpts)
   real(dp) :: kweights(nkpts)
   integer :: nbs_wfc(natoms), nbs_aux(natoms), types(natoms), map_q_ks(nkpts)
   complex(dp) :: wfc(nbasis, nstates, nkpts, nspins)
   complex(dp), allocatable :: vq(:,:), contrib_ibzk(:)
   real(dp) :: efermi, rpa_corr

   call initialize()

   write(*,*) "Version:", librpa_get_major_version(), &
                          librpa_get_minor_version(), &
                          librpa_get_patch_version()

   write(*,*) "Host code output: initialize default options"
   call opts%init()
   call opts%set_output_dir("librpa_f03.d")

   write(*,*) "Default frequency numbers: ", opts%nfreq
   write(*,*) "Output directory: ", opts%output_dir
   opts%parallel_routing = LIBRPA_ROUTING_LIBRI
   opts%nfreq = nfreq

   call h%init(MPI_COMM_WORLD)

   ! Play with the handler
   ! Initialize dimensions
   call h%set_scf_dimension(nspins, nkpts, nstates, nbasis)
   ! Parse the energy levels and occupation numbers
   allocate(wg(nstates, nkpts, nspins), ekb(nstates, nkpts, nspins))
   ! only the first state is occupied
   wg(:, :, :) = 0.0_dp
   wg(1, :, :) = 2.0_dp
   efermi = 0.0_dp
   do ispin = 1, nspins
      do ik = 1, nkpts
         do istate = 1, nstates
            ekb(istate, ik, ispin) = efermi + sign(istate + ik - 0.5_dp, istate - 1.5_dp)
         end do
      end do
   end do
   !print*, wg
   !print*, ekb
   call h%set_wg_ekb_efermi(nspins, nkpts, nstates, wg, ekb, efermi)

   ! Wave-function
   wfc(:,:,:,:) = (0.0_dp, 0.0_dp)
   do ispin = 1, nspins
      do ik = 1, nkpts
         do istate = 1, nstates
            wfc(istate, istate, ik, ispin) = (1.0_dp, 0._dp)
         end do
         call h%set_wfc(ispin, ik, nstates, nbasis, wfc(:, :, ik, ispin))
      end do
   end do

   nbs_wfc(1) = 1
   nbs_wfc(2) = 2
   if (sum(nbs_wfc) .ne. nbasis) call finalize(.false.)
   call h%set_ao_basis_wfc(natoms, nbs_wfc)
   nbs_aux(1) = 3
   nbs_aux(2) = 6
   if (sum(nbs_aux) .ne. naux) call finalize(.false.)
   call h%set_ao_basis_aux(natoms, nbs_aux)

   ! Hexagonal lattice
   latt(:,:) = 0.0_dp
   recplatt(:,:) = 0.0_dp
   latt(1, 1) = 2.0_dp
   latt(1, 2) = -1.0_dp
   latt(2, 2) = 1.0_dp * sqrt(3.0_dp)
   latt(3, 3) = 10.0_dp
   recplatt(1, 1) = 1.0_dp * pi
   recplatt(2, 1) = 1.0_dp * pi / sqrt(3.0_dp)
   recplatt(2, 2) = 2.0_dp * pi / sqrt(3.0_dp)
   recplatt(3, 3) = 0.2_dp * pi
   call h%set_latvec_and_G(latt, recplatt)

   ! K-points
   kpoints(:,:) = 0.0_dp
   kweights(:) = 1.0_dp / nkpts
   ! 1 = Gamma, 2 = b1/3, 3 = 2*b1/3
   kpoints(:,2) = recplatt(:,1) / 3.0_dp
   kpoints(:,3) = 2.0_dp * recplatt(:,1) / 3.0_dp
   call h%set_kgrids_kvec(nk1, nk2, nk3, nkpts, kpoints, kweights)
   ! SCF k-point to Coulomb q-point mapping
   map_q_ks(1) = 1
   map_q_ks(2) = 2
   map_q_ks(3) = 2
   call h%set_kq_mapping(nkpts, map_q_ks)

   ! Atoms position
   types(1) = 1
   types(2) = 2
   posi_cart(:,:) = 0.0_dp
   posi_cart(2,2) = 2.0_dp / sqrt(3.0_dp)
   posi_cart(3,2) = 5.0_dp
   call h%set_atoms(natoms, types, posi_cart)

   ! Set LRI coefficients
   do ia1 = 1, natoms
      do ia2 = 1, natoms
         allocate(ri_coeff(nbs_aux(ia1), nbs_wfc(ia2), nbs_wfc(ia1)))
         ri_coeff(:,:,:) = 0.0_dp
         do ik1 = -nint((nk1-0.1)/2.0), nint(nk1/2.0-0.6)
            do ik2 = -nint((nk2-0.1)/2.0), nint(nk2/2.0-0.6)
               do ik3 = -nint((nk3-0.1)/2.0), nint(nk3/2.0-0.6)
                  r(1) = ik1
                  r(2) = ik2
                  r(3) = ik3
                  ri_coeff(1,1,1) = 0.01_dp * real(100 * ia1 + 10 * ia2 + ik1, kind=dp)
                  call h%set_lri_coeff(opts%parallel_routing, ia1, ia2, &
                                       nbs_wfc(ia1), nbs_wfc(ia2), nbs_aux(ia1), &
                                       r, ri_coeff)
               end do
            end do
         end do
         deallocate(ri_coeff)
      end do
   end do

   ! Set Coulomb through atom pair interface
   do ia1 = 1, natoms
      do ia2 = 1, natoms
         allocate(vq(nbs_aux(ia1), nbs_aux(ia2)))
         vq(:,:) = (0.0_dp, 0.0_dp)
         do ik = 1, nkpts
            vq(1, 1) = real(10 * ik + ia1 + 0.1_dp * ia2, kind=dp)
            call h%set_aux_bare_coulomb_k_atom_pair(ik, ia1, ia2, nbs_aux(ia1), nbs_aux(ia2), vq, 0.0_dp)
            vq(1, 1) = -vq(1, 1)
            call h%set_aux_cut_coulomb_k_atom_pair(ik, ia1, ia2, nbs_aux(ia1), nbs_aux(ia2), vq, 0.0_dp)
         end do
         deallocate(vq)
      end do
   end do

   allocate(omegas(nfreq), df(nfreq))
   omegas = [(real(i, kind=dp), i = 1, nfreq)]
   df = 1.0_dp
   call h%set_dielect_func_imagfreq(nfreq, omegas, df)
   deallocate(omegas, df)

   allocate(contrib_ibzk(2))
   opts%tfgrids_type = LIBRPA_TFGRID_MINIMAX
   rpa_corr = h%get_rpa_correlation_energy(opts, 2, contrib_ibzk)
   deallocate(contrib_ibzk)

   call h%free()

   deallocate(wg, ekb)

   call finalize(.true.)

contains

   subroutine initialize
      implicit none
      integer :: provided
      call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, ierr)
      if (MPI_THREAD_MULTIPLE .ne. provided) then
         write(*,*) "Warning: MPI_Init_thread provide ", provided, " != required", MPI_THREAD_MULTIPLE
      endif
      call librpa_init_global(.true., "output.txt", .true.)
   end subroutine initialize

   subroutine finalize(success)
      logical, intent(in) :: success

      if (success) then
         write(*,*) prog, " succeeded"
      else
         write(*,*) prog, " failed"
      end if

      call librpa_finalize_global
      call MPI_Finalize(ierr)
   end subroutine finalize

end program
