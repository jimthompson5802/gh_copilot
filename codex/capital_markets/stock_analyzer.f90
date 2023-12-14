program stock_analzyer
    implicit none
    ! Declare variables
    character(len=10) :: date
    character(len=100) :: line
    real, dimension(:), allocatable :: close_prices, close_prices_new
    real :: open, high, low, close, adj_close, sum_val, mean_val, min_val, max_val
    real :: strike_price, risk_free_rate, volatility, time_to_maturity
    real :: d1, d2, call_option, put_option, stock_price
    integer :: volume, io, count

    ! Open the CSV file for reading
    open(unit=10, file='AAPL.csv', status='old', action='read')

    ! Skip the header line
    read(10, '(a)') line

    count = 0
    do
        ! Read each line of the file
        read(10, *, iostat=io) date, open, high, low, close, adj_close, volume
        if (io /= 0) exit  ! Exit the loop if the end of the file is reached
        count = count + 1
        if (allocated(close_prices)) then
            ! If close_prices is already allocated, move the allocation to close_prices_new
            call move_alloc(close_prices, close_prices_new)
            allocate(close_prices(count))
            close_prices(1:count-1) = close_prices_new
            close_prices(count) = close
        else
            ! If close_prices is not allocated, allocate it and store the close price
            allocate(close_prices(1))
            close_prices(1) = close
        end if
    end do

    if (count > 0) then
        ! Calculate the sum, mean, min, and max of the close prices
        sum_val = sum(close_prices)
        mean_val = sum_val / count
        min_val = minval(close_prices)
        max_val = maxval(close_prices)
        print*, 'Mean closing price: ', mean_val
        print*, 'Min closing price: ', min_val
        print*, 'Max closing price: ', max_val

        ! Initialize these variables with appropriate values
        stock_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        volatility = 0.2
        time_to_maturity = 1.0    

        ! Calculate the values of d1 and d2 for the Black-Scholes formula
        d1 = (log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * &
             time_to_maturity) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        ! Calculate the prices of the call and put options using the Black-Scholes formula
        call_option = stock_price * cdf_normal(d1) - strike_price * &
             exp(-risk_free_rate * time_to_maturity) * cdf_normal(d2)
        put_option = strike_price * exp(-risk_free_rate * time_to_maturity) * cdf_normal(-d2) - &
             stock_price * cdf_normal(-d1)

        print*, 'Call option price: ', call_option
        print*, 'Put option price: ', put_option
    end if

    ! Close the file
    close(10)

    contains

    ! Define the cdf_normal function, which calculates the cumulative distribution function for a normal distribution
    function cdf_normal(x)
        implicit none
        real, intent(in) :: x
        real :: cdf_normal
        cdf_normal = 0.5 * (1.0 + erf(x / sqrt(2.0)))
    end function cdf_normal

end program stock_analzyer