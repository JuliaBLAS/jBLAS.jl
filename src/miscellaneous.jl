

function divide_evenly(n, d)
    num_splits = cld(n, d)
    num_splits, divrem(n, num_splits)
end

round_x_to_nearest_y(x::Int, y::Int, ::RoundingMode{:Up}) = cld(x,y)*y
round_x_to_nearest_y(x::Int, y::Int, ::RoundingMode{:Down}) = fld(x,y)*y
round_x_to_nearest_y(x::Int, y::Int, ::RoundingMode{:Nearest}) = div(x,y)*y

round_x_to_nearest_y(x, y) = round(Int, x / y) * y
round_x_to_nearest_y(x, y::Int, rm::RoundingMode) = round(Int, x / y, rm) * y
round_x_to_nearest_y(x, y, ::RoundingMode{:Nearest}) = round(Int, x / y) * y
