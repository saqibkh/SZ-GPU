! Fortran interface for the SZ-GPU error-bounded lossy compressor.
!
! Host arrays. Generic interfaces for real(4) and real(8) arrays of rank one
! to four; pass the total element count. All device work happens inside the
! library, so callers never touch CUDA.
!
!     use szgpu
!     call szgpu_compress(field, n, 1.0e-3, cmp, cmp_size, ierr)
!     call szgpu_decompress(field, cmp, cmp_size, n, 1.0e-3, ierr)
!
! Data already on the GPU. Pass device addresses as type(c_ptr), which is what
! CUDA Fortran gives for a device array through c_devloc, and what OpenACC and
! OpenMP give inside host_data use_device / target data use_device_ptr. The
! element type is selected by the kind of the error bound.
!
!     type(c_ptr) :: ws
!     type(szgpu_result) :: res
!     call szgpu_workspace_create(ws, n, ierr)
!     call szgpu_compress_device(d_in, d_cmp, n, 1.0e-3, ws, res, ierr)
!     call szgpu_decompress_device(d_out, d_cmp, n, 1.0e-3, res, ws, ierr)
!     call szgpu_workspace_destroy(ws)
!
! A workspace may be c_null_ptr, in which case the call allocates and frees a
! temporary one; reuse a workspace when compressing repeatedly. The optional
! stream argument is a CUDA stream handle, defaulting to the default stream.
!
! ierr is SZGPU_OK (0) on success; szgpu_status_message(ierr) describes a
! failure. cmp must provide at least szgpu_max_compressed_bytes(n) bytes.

module szgpu
    use iso_c_binding, only: c_float, c_double, c_size_t, c_int, c_int8_t, &
                             c_char, c_ptr, c_loc, c_f_pointer, c_associated, &
                             c_null_char, c_null_ptr
    implicit none
    private

    public :: szgpu_compress, szgpu_decompress
    public :: szgpu_compress_device, szgpu_decompress_device
    public :: szgpu_workspace_create, szgpu_workspace_destroy, szgpu_workspace_capacity
    public :: szgpu_make_result
    public :: szgpu_max_compressed_bytes, szgpu_num_tiles
    public :: szgpu_version, szgpu_status_message
    public :: szgpu_result, SZGPU_OK

    integer(c_int), parameter :: SZGPU_OK = 0_c_int

    type, bind(c) :: szgpu_result
        integer(c_size_t) :: cmp_size    = 0_c_size_t
        integer(c_size_t) :: num_tiles   = 0_c_size_t
        integer(c_size_t) :: data_offset = 0_c_size_t
    end type szgpu_result

    interface
        function c_max_bytes(n) bind(c, name='szgpu_max_compressed_bytes') result(r)
            import :: c_size_t
            implicit none
            integer(c_size_t), value :: n
            integer(c_size_t) :: r
        end function c_max_bytes

        function c_num_tiles(n) bind(c, name='szgpu_num_tiles') result(r)
            import :: c_size_t
            implicit none
            integer(c_size_t), value :: n
            integer(c_size_t) :: r
        end function c_num_tiles

        function c_version() bind(c, name='szgpu_version') result(p)
            import :: c_ptr
            implicit none
            type(c_ptr) :: p
        end function c_version

        function c_status_string(s) bind(c, name='szgpu_status_string') result(p)
            import :: c_int, c_ptr
            implicit none
            integer(c_int), value :: s
            type(c_ptr) :: p
        end function c_status_string

        function c_compress_r4(values, cmp, n, eb, out_size) &
                bind(c, name='szgpu_compress_hostptr') result(st)
            import :: c_float, c_size_t, c_int, c_int8_t, c_ptr
            implicit none
            type(c_ptr), value :: values
            integer(c_int8_t), intent(out) :: cmp(*)
            integer(c_size_t), value :: n
            real(c_float), value :: eb
            integer(c_size_t), intent(out) :: out_size
            integer(c_int) :: st
        end function c_compress_r4

        function c_decompress_r4(values, cmp, cmp_size, n, eb) &
                bind(c, name='szgpu_decompress_hostptr') result(st)
            import :: c_float, c_size_t, c_int, c_int8_t, c_ptr
            implicit none
            type(c_ptr), value :: values
            integer(c_int8_t), intent(in) :: cmp(*)
            integer(c_size_t), value :: cmp_size
            integer(c_size_t), value :: n
            real(c_float), value :: eb
            integer(c_int) :: st
        end function c_decompress_r4

        function c_compress_r8(values, cmp, n, eb, out_size) &
                bind(c, name='szgpu_compress_hostptr_f64') result(st)
            import :: c_double, c_size_t, c_int, c_int8_t, c_ptr
            implicit none
            type(c_ptr), value :: values
            integer(c_int8_t), intent(out) :: cmp(*)
            integer(c_size_t), value :: n
            real(c_double), value :: eb
            integer(c_size_t), intent(out) :: out_size
            integer(c_int) :: st
        end function c_compress_r8

        function c_decompress_r8(values, cmp, cmp_size, n, eb) &
                bind(c, name='szgpu_decompress_hostptr_f64') result(st)
            import :: c_double, c_size_t, c_int, c_int8_t, c_ptr
            implicit none
            type(c_ptr), value :: values
            integer(c_int8_t), intent(in) :: cmp(*)
            integer(c_size_t), value :: cmp_size
            integer(c_size_t), value :: n
            real(c_double), value :: eb
            integer(c_int) :: st
        end function c_decompress_r8

        function c_ws_create(ws, max_n) bind(c, name='szgpu_workspace_create') result(st)
            import :: c_ptr, c_size_t, c_int
            implicit none
            type(c_ptr), intent(out) :: ws
            integer(c_size_t), value :: max_n
            integer(c_int) :: st
        end function c_ws_create

        subroutine c_ws_destroy(ws) bind(c, name='szgpu_workspace_destroy')
            import :: c_ptr
            implicit none
            type(c_ptr), value :: ws
        end subroutine c_ws_destroy

        function c_ws_capacity(ws) bind(c, name='szgpu_workspace_capacity') result(r)
            import :: c_ptr, c_size_t
            implicit none
            type(c_ptr), value :: ws
            integer(c_size_t) :: r
        end function c_ws_capacity

        function c_make_result(n, cmp_size) bind(c, name='szgpu_make_result') result(r)
            import :: c_size_t
            import :: szgpu_result
            implicit none
            integer(c_size_t), value :: n
            integer(c_size_t), value :: cmp_size
            type(szgpu_result) :: r
        end function c_make_result

        function c_dev_compress_r4(d_in, d_cmp, n, eb, ws, stream, res) &
                bind(c, name='szgpu_compress') result(st)
            import :: c_ptr, c_size_t, c_float, c_int
            import :: szgpu_result
            implicit none
            type(c_ptr), value :: d_in
            type(c_ptr), value :: d_cmp
            integer(c_size_t), value :: n
            real(c_float), value :: eb
            type(c_ptr), value :: ws
            type(c_ptr), value :: stream
            type(szgpu_result), intent(out) :: res
            integer(c_int) :: st
        end function c_dev_compress_r4

        function c_dev_compress_r8(d_in, d_cmp, n, eb, ws, stream, res) &
                bind(c, name='szgpu_compress_f64') result(st)
            import :: c_ptr, c_size_t, c_double, c_int
            import :: szgpu_result
            implicit none
            type(c_ptr), value :: d_in
            type(c_ptr), value :: d_cmp
            integer(c_size_t), value :: n
            real(c_double), value :: eb
            type(c_ptr), value :: ws
            type(c_ptr), value :: stream
            type(szgpu_result), intent(out) :: res
            integer(c_int) :: st
        end function c_dev_compress_r8

        function c_dev_decompress_r4(d_out, d_cmp, n, eb, res, ws, stream) &
                bind(c, name='szgpu_decompress') result(st)
            import :: c_ptr, c_size_t, c_float, c_int
            import :: szgpu_result
            implicit none
            type(c_ptr), value :: d_out
            type(c_ptr), value :: d_cmp
            integer(c_size_t), value :: n
            real(c_float), value :: eb
            type(szgpu_result), intent(in) :: res
            type(c_ptr), value :: ws
            type(c_ptr), value :: stream
            integer(c_int) :: st
        end function c_dev_decompress_r4

        function c_dev_decompress_r8(d_out, d_cmp, n, eb, res, ws, stream) &
                bind(c, name='szgpu_decompress_f64') result(st)
            import :: c_ptr, c_size_t, c_double, c_int
            import :: szgpu_result
            implicit none
            type(c_ptr), value :: d_out
            type(c_ptr), value :: d_cmp
            integer(c_size_t), value :: n
            real(c_double), value :: eb
            type(szgpu_result), intent(in) :: res
            type(c_ptr), value :: ws
            type(c_ptr), value :: stream
            integer(c_int) :: st
        end function c_dev_decompress_r8
    end interface

    interface szgpu_compress_device
        module procedure szgpu_compress_device_r4, szgpu_compress_device_r8
    end interface szgpu_compress_device

    interface szgpu_decompress_device
        module procedure szgpu_decompress_device_r4, szgpu_decompress_device_r8
    end interface szgpu_decompress_device

    interface szgpu_compress
        module procedure szgpu_compress_r4_1, szgpu_compress_r4_2, &
                         szgpu_compress_r4_3, szgpu_compress_r4_4, &
                         szgpu_compress_r8_1, szgpu_compress_r8_2, &
                         szgpu_compress_r8_3, szgpu_compress_r8_4
    end interface szgpu_compress

    interface szgpu_decompress
        module procedure szgpu_decompress_r4_1, szgpu_decompress_r4_2, &
                         szgpu_decompress_r4_3, szgpu_decompress_r4_4, &
                         szgpu_decompress_r8_1, szgpu_decompress_r8_2, &
                         szgpu_decompress_r8_3, szgpu_decompress_r8_4
    end interface szgpu_decompress

contains

    subroutine szgpu_compress_r4_1(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_float), intent(in), contiguous, target :: values(:)
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r4(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r4_1

    subroutine szgpu_compress_r4_2(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_float), intent(in), contiguous, target :: values(:, :)
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r4(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r4_2

    subroutine szgpu_compress_r4_3(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_float), intent(in), contiguous, target :: values(:, :, :)
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r4(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r4_3

    subroutine szgpu_compress_r4_4(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_float), intent(in), contiguous, target :: values(:, :, :, :)
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r4(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r4_4

    subroutine szgpu_decompress_r4_1(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_float), intent(out), contiguous, target :: values(:)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r4(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r4_1

    subroutine szgpu_decompress_r4_2(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_float), intent(out), contiguous, target :: values(:, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r4(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r4_2

    subroutine szgpu_decompress_r4_3(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_float), intent(out), contiguous, target :: values(:, :, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r4(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r4_3

    subroutine szgpu_decompress_r4_4(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_float), intent(out), contiguous, target :: values(:, :, :, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r4(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r4_4

    subroutine szgpu_compress_r8_1(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_double), intent(in), contiguous, target :: values(:)
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r8(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r8_1

    subroutine szgpu_compress_r8_2(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_double), intent(in), contiguous, target :: values(:, :)
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r8(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r8_2

    subroutine szgpu_compress_r8_3(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_double), intent(in), contiguous, target :: values(:, :, :)
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r8(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r8_3

    subroutine szgpu_compress_r8_4(values, n, eb_abs, cmp, cmp_size, ierr)
        real(c_double), intent(in), contiguous, target :: values(:, :, :, :)
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int8_t), intent(out) :: cmp(*)
        integer(c_size_t), intent(out) :: cmp_size
        integer(c_int), intent(out) :: ierr
        ierr = c_compress_r8(c_loc(values), cmp, n, eb_abs, cmp_size)
    end subroutine szgpu_compress_r8_4

    subroutine szgpu_decompress_r8_1(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_double), intent(out), contiguous, target :: values(:)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r8(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r8_1

    subroutine szgpu_decompress_r8_2(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_double), intent(out), contiguous, target :: values(:, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r8(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r8_2

    subroutine szgpu_decompress_r8_3(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_double), intent(out), contiguous, target :: values(:, :, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r8(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r8_3

    subroutine szgpu_decompress_r8_4(values, cmp, cmp_size, n, eb_abs, ierr)
        real(c_double), intent(out), contiguous, target :: values(:, :, :, :)
        integer(c_int8_t), intent(in) :: cmp(*)
        integer(c_size_t), intent(in) :: cmp_size
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        integer(c_int), intent(out) :: ierr
        ierr = c_decompress_r8(c_loc(values), cmp, cmp_size, n, eb_abs)
    end subroutine szgpu_decompress_r8_4

    subroutine szgpu_workspace_create(ws, max_n, ierr)
        type(c_ptr), intent(out) :: ws
        integer(c_size_t), intent(in) :: max_n
        integer(c_int), intent(out) :: ierr
        ierr = c_ws_create(ws, max_n)
    end subroutine szgpu_workspace_create

    subroutine szgpu_workspace_destroy(ws)
        type(c_ptr), intent(inout) :: ws
        call c_ws_destroy(ws)
        ws = c_null_ptr
    end subroutine szgpu_workspace_destroy

    function szgpu_workspace_capacity(ws) result(r)
        type(c_ptr), intent(in) :: ws
        integer(c_size_t) :: r
        r = c_ws_capacity(ws)
    end function szgpu_workspace_capacity

    ! Rebuilds the descriptor of a stream known only by its element count and
    ! byte size, for decompressing data read back from storage.
    function szgpu_make_result(n, cmp_size) result(r)
        integer(c_size_t), intent(in) :: n
        integer(c_size_t), intent(in) :: cmp_size
        type(szgpu_result) :: r
        r = c_make_result(n, cmp_size)
    end function szgpu_make_result

    subroutine szgpu_compress_device_r4(d_in, d_cmp, n, eb_abs, ws, res, ierr, stream)
        type(c_ptr), intent(in) :: d_in
        type(c_ptr), intent(in) :: d_cmp
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        type(c_ptr), intent(in) :: ws
        type(szgpu_result), intent(out) :: res
        integer(c_int), intent(out) :: ierr
        type(c_ptr), intent(in), optional :: stream
        type(c_ptr) :: s
        s = c_null_ptr
        if (present(stream)) s = stream
        ierr = c_dev_compress_r4(d_in, d_cmp, n, eb_abs, ws, s, res)
    end subroutine szgpu_compress_device_r4

    subroutine szgpu_compress_device_r8(d_in, d_cmp, n, eb_abs, ws, res, ierr, stream)
        type(c_ptr), intent(in) :: d_in
        type(c_ptr), intent(in) :: d_cmp
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        type(c_ptr), intent(in) :: ws
        type(szgpu_result), intent(out) :: res
        integer(c_int), intent(out) :: ierr
        type(c_ptr), intent(in), optional :: stream
        type(c_ptr) :: s
        s = c_null_ptr
        if (present(stream)) s = stream
        ierr = c_dev_compress_r8(d_in, d_cmp, n, eb_abs, ws, s, res)
    end subroutine szgpu_compress_device_r8

    subroutine szgpu_decompress_device_r4(d_out, d_cmp, n, eb_abs, res, ws, ierr, stream)
        type(c_ptr), intent(in) :: d_out
        type(c_ptr), intent(in) :: d_cmp
        integer(c_size_t), intent(in) :: n
        real(c_float), intent(in) :: eb_abs
        type(szgpu_result), intent(in) :: res
        type(c_ptr), intent(in) :: ws
        integer(c_int), intent(out) :: ierr
        type(c_ptr), intent(in), optional :: stream
        type(c_ptr) :: s
        s = c_null_ptr
        if (present(stream)) s = stream
        ierr = c_dev_decompress_r4(d_out, d_cmp, n, eb_abs, res, ws, s)
    end subroutine szgpu_decompress_device_r4

    subroutine szgpu_decompress_device_r8(d_out, d_cmp, n, eb_abs, res, ws, ierr, stream)
        type(c_ptr), intent(in) :: d_out
        type(c_ptr), intent(in) :: d_cmp
        integer(c_size_t), intent(in) :: n
        real(c_double), intent(in) :: eb_abs
        type(szgpu_result), intent(in) :: res
        type(c_ptr), intent(in) :: ws
        integer(c_int), intent(out) :: ierr
        type(c_ptr), intent(in), optional :: stream
        type(c_ptr) :: s
        s = c_null_ptr
        if (present(stream)) s = stream
        ierr = c_dev_decompress_r8(d_out, d_cmp, n, eb_abs, res, ws, s)
    end subroutine szgpu_decompress_device_r8

    function szgpu_max_compressed_bytes(n) result(r)
        integer(c_size_t), intent(in) :: n
        integer(c_size_t) :: r
        r = c_max_bytes(n)
    end function szgpu_max_compressed_bytes

    function szgpu_num_tiles(n) result(r)
        integer(c_size_t), intent(in) :: n
        integer(c_size_t) :: r
        r = c_num_tiles(n)
    end function szgpu_num_tiles

    function szgpu_version() result(v)
        character(len=:), allocatable :: v
        v = to_f_string(c_version())
    end function szgpu_version

    function szgpu_status_message(s) result(v)
        integer(c_int), intent(in) :: s
        character(len=:), allocatable :: v
        v = to_f_string(c_status_string(s))
    end function szgpu_status_message

    function to_f_string(p) result(v)
        type(c_ptr), intent(in) :: p
        character(len=:), allocatable :: v
        character(kind=c_char), pointer :: chars(:)
        integer :: i, j
        v = ''
        if (.not. c_associated(p)) return
        call c_f_pointer(p, chars, [256])
        do i = 1, 256
            if (chars(i) == c_null_char) exit
        end do
        if (i <= 1) return
        v = repeat(' ', i - 1)
        do j = 1, i - 1
            v(j:j) = chars(j)
        end do
    end function to_f_string

end module szgpu
