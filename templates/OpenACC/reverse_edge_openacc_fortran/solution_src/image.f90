!
! This is an OpenACC code that performs an iterative reverse edge 
! detection algorithm.
! 
! Training material developed by Alan Gray
! Copyright EPCC, The University of Edinburgh, 2012
!



program casestudy

  implicit none


  ! dimensions of image
  integer, parameter :: M=600, N=840

  ! number of iterations to run
  integer, parameter :: ITERATIONS   = 100

  real, dimension(0:M+1, 0:N+1) :: new, old, edge

  real, dimension(M,N) :: buf

  integer, parameter :: maxlen = 64

  character*(maxlen) :: filename

  integer :: i, j, iter

  real etime
  real tarray(2), time0, time1, time2

! Variables for subroutine system_clock

  integer count_0, count_1, count_rate, count_max


  write(*,*) 'Number of iterations = ', ITERATIONS
  
  filename = '../../input_files/edge600x840.dat'


! read in edge data
  write(*,*)
  write(*,*) 'Reading ', filename
  write(*,*)

  call datread(filename, buf, M, N)



! initialise variables
  do j = 0, N+1
     do i = 0, M+1
        
        !if halo
        if ((i .eq. 0) .or. (j .eq. 0) &
             .or. (i .eq. M+1) .or. (j .eq. N+1) ) then
           edge(i,j) = 0.0
        else
           edge(i,j) = buf(i,j)
        end if
        
        old(i,j) = edge(i,j)  
        
     end do
  end do
  

  !$acc data copy(old) copyin(edge) create(new)


  call system_clock(count_0, count_rate, count_max)

  do iter = 1, ITERATIONS !start main loop
     
     ! perform stencil operation

     !$acc parallel vector_length(256)
     !$acc loop
     do j = 1, N
        do i = 1, M
           
           new(i,j) = 0.25*(old(i+1,j)+old(i-1,j)+old(i,j+1)+old(i,j-1) &
                - edge(i,j))
           
        end do
     end do
     
     ! copy output back to input buffer

     !$acc loop
     do j = 1, N
        do i = 1, M
           
           old(i,j) = new(i,j)
           
        end do
     end do
     
     !$acc end parallel
     
  end do !end main loop
  
  !$acc end data
  
  call system_clock(count_1, count_rate, count_max)
  
  write(*,*) "time: ", (count_1-count_0)*1.0/count_rate, "s"
  
  !  Gather data

  do j = 1, N
    do i = 1, M

      buf(i,j) = old(i,j)

    end do
  end do



  filename='image600x840.pgm'
  write(*,*)
  write(*,*) 'Writing ', filename
  call pgmwrite(filename, buf, M, N)
  


end program casestudy
