using Random
using Zygote: @adjoint
using Zygote
using DSP

Random.seed!(1)

function dconv(x, a, b)
    n = length(size(a))
    x = reshape(vec(x), (size(a) .+ size(b) .- ones(Int, n))...)
    r = (
        DSP.conv(x, reverse(b))[(
            # DSP.xcorr(x, b, padmode = :none)[(
            i:j for (i, j) in zip(size(b), size(x))
        )...],
        DSP.conv(x, reverse(a))[(
            # DSP.xcorr(x, a, padmode = :none)[(
            i:j for (i, j) in zip(size(a), size(x))
        )...],
    )
    return r
end
# @adjoint DSP.conv(a, b) = DSP.conv(a, b), x -> dconv(x, a, b)
@adjoint DSP.conv(a, b) = DSP.conv(float.(a), float.(b)),
x ->
    dconv(float.(x), float.(a), float.(b))
# function frule(
#     (_, ΔA, ΔB),
#     ::typeof(DPS.conv),
#     A,
#     B,
# )
#     Ω = conv(A , B)
#     ∂Ω = conv(ΔA * B )+ A * ΔB
#     return (Ω, ∂Ω)
# end
@show jacobian(DSP.conv, [1, 2], [3, 4])

const tol = 1e-9

mutable struct LinearOperator
    i::Int
    o::Int
    li::Int
    lo::Int
    lr::Int

    dx::Float64
    rmax::Float64
    sz::Any

    params::AbstractVector

    radfunc::Function
    conv::Function

    H::AbstractArray
end

function remake!(m::LinearOperator; dx = m.dx, rmax = m.rmax, sz = m.sz)
    m.H = makefilters(m.lr, dx, rmax, sz, m.radfunc, m.params)
end

invr =
    (r; p = 1, rmax = 1.0e20, kwargs...) ->
        abs(r) < tol || r > rmax + tol ? 0.0 : 1 / r^p

const FD = [[1.0], [0.0, 0.5]]
function δ(x, dx; n = 0)
    i = round(Int, x / dx)
    if abs(x - i) < tol
        return i < length(FD[n+1]) ? FD[n+1][i+1] : 0.0
    end
    return 0.0
end

function LinearOperator(name, dx; kwargs...)
    LinearOperator(dx; name = name, kwargs...)
end
function LinearOperator(
    dx;
    i::Int = -1,
    o::Int = -1,
    li = -1,
    lo = -1,
    lr = -1,
    name = nothing,
    radfunc = g,
    params = ones(8),
    rmax = 1 / tol,
    sz = nothing,
)
    dV = dx^3

    if name == :potential
        li = lo = lr = 0
        params = []
        radfunc = invr
    elseif name == :field
        li = 0
        lr = lo = 1
        params = []
        radfunc = (r; kwargs...) -> invr(r; p = 2)
    elseif name == :grad
        li = 0
        lr = lo = 1
        params = []
        rmax = dx
        radfunc = (r; kwargs...) -> δ(r, dx; n = 1) / dV
    end

    if rmax !== nothing
        nrmax = round(Int, rmax / dx)
    end
    if sz === nothing
        sz = (1 + 2nrmax) * ones(Int, 3)
    end

    conv = Conv(li, lr, lo)

    m = LinearOperator(
        i,
        o,
        li,
        lo,
        lr,
        dx,
        rmax,
        sz,
        params,
        radfunc,
        conv,
        [],
    )
    remake!(m)
    return m
end

function (m::LinearOperator)(X; remake = false)
    if remake
        remake!(m)
    end
    m.conv(X, m.H)
end

"""
radial function
"""
function g(r; params = ones(16), dx = 1.0)
    k, σ = params[end-2+1:end]
    dV = dx^3
    funcs = [
        abs(r) < tol ? 1 / dV : 0.0,
        -invr(r; rmax = dx) / dV,
        invr(r),
        invr(r; p = 2),
        exp(-abs(k) * r),
        exp(-(r / σ)^2),
    ]
    dot(params[1:length(funcs)], funcs)
end

const s3 = sqrt(3)
function makefilters(lr, dx, rmax, sz, radfunc, params;)
    dV = dx^3
    nrmax = round(Int, rmax / dx)
    center = (sz .+ 1) .÷ 2


    V = 0
    Zygote.ignore() do
        V = [[x, y, z] .- center for x = 1:sz[1], y = 1:sz[2], z = 1:sz[3]] * dx
    end
    R0 = norm.(V)

    radfunc1 =
        r -> r > rmax + tol ? 0.0 : radfunc(r; params = params, dx = dx)
    R = radfunc1.(R0) * dV

    if lr == 0
        ret = [R]
    else
        Vhat = V ./ broadcast(x -> abs(x) < tol ? 1.0 : x, R0)
        X = getindex.(Vhat, 1)
        Y = getindex.(Vhat, 2)
        Z = getindex.(Vhat, 3)
        # println(X)
        # println(R)
        if lr == 1
            ret = [R .* X, R .* Y, R .* Z]
        elseif lr == 2
            ret = broadcast(
                x -> R .* x,
                [
                    s3 * X .* Z,
                    s3 * Y .* X,
                    Y .^ 2 - (X .^ 2 + Z .^ 2) / 2,
                    s3 * Y .* Z,
                    (Z .^ 2 - X .^ 2) * s3 / 2,
                ],
            )
        end
    end
    # [x ./ sum(abs.(x)) for x in ret]
    ret
end

function conv(x, f; crop = 1, periodic = false)
    sx = 0
    sf = 0
    ix = 0
    # starts = 0
    Zygote.ignore() do
        sx = size(x)
        sf = size(f)
        if crop==2
            sx,sf=sf,sx
        end
        ix = [((i-1)÷2+1):((i-1)÷2+j) for (i, j) in zip(sf, sx)]
    end

    y = DSP.conv(x, f)

    if periodic
        for i = 1:3

        end
    else
        return y[ix...]
    end
end

c0__ = (X, F; kwargs...) -> [conv(X[1], f; kwargs...) for f in F]
c111 =
    (X, Y; kwargs...) -> [
        conv(X[2], Y[3]; kwargs...) - conv(X[3], Y[2]; kwargs...),
        conv(X[3], Y[1]; kwargs...) - conv(X[1], Y[3]; kwargs...),
        conv(X[1], Y[2]; kwargs...) - conv(X[2], Y[1]; kwargs...),
    ]
c__0 = (X, F; kwargs...) -> sum(conv.(X, F; kwargs...))
c121 =
    (X, Y; kwargs...) -> [
        conv(X[1], -Y[3] / s3 .- Y[5]; kwargs...) +
        conv(X[2], Y[2]; kwargs...) +
        conv(X[3], Y[1]; kwargs...),
        conv(X[1], Y[2]; kwargs...) +
        conv(X[2], 2 / s3 * Y[3]; kwargs...) +
        conv(X[3], Y[4]; kwargs...),
        conv(X[1], Y[1]; kwargs...) +
        conv(X[2], Y[4]; kwargs...) +
        conv(X[3], -Y[3] / s3 .+ Y[5]; kwargs...),
    ]

function Conv(l1, l2, l; crop = 1, kwargs...)
    if l1 <= l2
        kwargs = (crop = crop,)
        pre = f -> ((X, Y) -> f(X, Y; kwargs...))
    else
        l1, l2 = l2, l1
        crop = [2, 1][crop]
        kwargs = (crop = crop,)
        pre = f -> ((X, Y) -> f(Y, X; kwargs...))
    end

    if l1 == 0
        return pre(c0__)
    elseif l1 == l2 && l == 0
        return pre(c__0)
    elseif l1 == l2 == l == 1
        return pre(c111)
    elseif (l1, l2, l) == (1, 2, 1)
        return pre(c121)
    end
end



function conv(x, y, l)
    Conv(length(x), length(y), l)(x, y)
end

function toArray(x::Vector)
    vcat(vec.(x)...)
    # cat(x...; dims = 4)
    # cat([reshape(x,(size(x)...,1)) for x in x]...;dims=4)
end

using GLMakie
# using CairoMakie

function tfplot(X)
    # fig = Figure()
    fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
    resolution = (1000, 700))
 for i = 1:length(X)
    volume(fig[1,i],X[i])
end
    # plots = [volume(X[i]) for i = 1:length(X)]
    # plot(plots..., layout = length(X))
end
# p
