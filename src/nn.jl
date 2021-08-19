# module EquivariantNeuralOperators

using LinearAlgebra
# using Rotations
using Zygote
using Random
using Flux



include("./utils.jl")
Random.seed!(1)

"""
equivariant convolution params
"""
# struct EquivConv{F,S<:AbstractArray,T<:Union{Zeros, AbstractVector}}
struct EquivConv{T<:AbstractFloat}
    paths::AbstractVector{LinearOperator}
    pathsmap::AbstractVector
    dx::T
    rmax::T
    sz
end

Flux.trainable(m::EquivConv) = [x.params for x in m.paths]

"""
initialize params
"""

function EquivConv(
    inranks::AbstractVector,
    outranks::AbstractVector,
    dx::Float64;
    rmax = 1/SMALL,
    sz=nothing,
    lmax = 1,
    paths = Vector{LinearOperator}(),
)
    Zygote.ignore() do
        nin = length(inranks)
        nout = length(outranks)
        inparities = (-1) .^ inranks
        outparities = (-1) .^ outranks

        # if sz!==nothing
        #     sz=2 .*sz .-1
        # end

        # iterate all tensor product paths
        if isempty(paths)
            for lr = 0:lmax
                for (i, li, si) in zip(1:nin, inranks, inparities)
                    loutrange = abs(li - lr):min(li + lr, lmax)
                    parity = si * (-1)^lr
                    for (o, lo) in enumerate(outranks)
                        if lo in loutrange && outparities[o] == parity
                            path = LinearOperator(
                                dx;
                                li = li,
                                lo = lo,
                                lr = lr,
                                i = i,
                                o = o,
                                rmax = rmax,
                                sz=sz
                            )
                            push!(paths, path)
                        end
                    end
                end
            end
        end


        pathsmap = [[] for i = 1:nout]
        for (i, path) in enumerate(paths)
            push!(pathsmap[path.o], i)
        end

        return EquivConv(paths, pathsmap, dx, rmax,sz)
    end
end

"""
layer function
"""
function (f::EquivConv)(
    X::AbstractVector;
    remake = true,
) where {T<:AbstractFloat}
    paths, pathsmap, dx, rmax,sz = f.paths, f.pathsmap, f.dx, f.rmax,f.sz
    sx = size(X[1][1])

if remake
    for x in paths
        remake!(x;dx=dx,rmax=rmax)
    end
    end

    Y = [
        dropdims(
            sum(
                hcat([paths[i](X[paths[i].i]) for i in pathsmap[o]]...),
                dims = 2,
            ),
            dims = 2,
        ) for o in eachindex(pathsmap)
    ]
    return Y
end

##
mutable struct Prod
    i1::Int
    i2::Int
    o::Int
    li1::Int
    li2::Int
    lo::Int
    prod::Function
    params::Any
end

function Prod(l1, l2, l)
    if l1 == 0
        return (x, y) -> [x[1] .* y for y in y]
    elseif l2 == 0
        return (x, y) -> [x .* y[1] for x in x]
    elseif l1 == l2 && l == 0
        return (x, y) -> [sum([x .* y for (x, y) in zip(x, y)])]
    end
end

"""
equivariant params
"""
struct EquivAttn
    paths::Vector{Prod}
    pathsmap::AbstractVector
end
Flux.trainable(m::EquivAttn) = [x.params for x in m.paths]


"""
initialize params
"""

function EquivAttn(
    inranks::AbstractVector{Int},
    outranks::AbstractVector{Int},
    ;
    lmax = 1,
    paths = Vector{Prod}(),
)
    Zygote.ignore() do
        nin = length(inranks)
        nout = length(outranks)
        println(inranks)
        inparities = (-1) .^ inranks
        outparities = (-1) .^ outranks

        # iterate all tensor product paths
        if isempty(paths)
                for (i1, li1, si1) in zip(1:nin, inranks, inparities)
                    for (i2, li2, si2) in
                        zip(1:i1, inranks[1:i1], inparities[1:i1])
                        # for (i2, li2, si2) in zip(
                        #     0:nin,
                        #     vcat([0], inranks[1:i1]),
                        #     vcat([1], inparities[1:i1]),
                        # )
                        loutrange = abs(li1 - li2):min(li1 + li2, lmax)
                        parity = si1 * si2
                        for (o, lo) in enumerate(outranks)
                            if lo in loutrange && outparities[o] == parity
                                path = Prod(
                                    i1,
                                    i2,
                                    o,
                                    li1,
                                    li2,
                                    lo,
                                    Prod(li1, li2, lo),
                                    ones(4, 1),
                                )
                                push!(paths, path)
                        end
                    end
                end
            end
        end


        pathsmap = [[] for i = 1:nout]
        for (i, path) in enumerate(paths)
            push!(pathsmap[path.o], i)
        end

        return EquivAttn(paths, pathsmap)
    end
end

"""
layer function
"""
function (f::EquivAttn)(X::AbstractVector;) where {T<:AbstractFloat}
    paths, pathsmap = f.paths, f.pathsmap
    sz = size(X[1][1])

    Y = [
        dropdims(
            sum(
                hcat(
                    [
                        paths[i].params[1] * paths[i].prod(
                            X[paths[i].i1],
                            paths[i].i2 === 0 ? [ones(sz...)] : X[paths[i].i2],
                        ) for i in pathsmap[o]
                    ]...,
                ),
                dims = 2,
            ),
            dims = 2,
        ) for o in eachindex(pathsmap)
    ]
    return Y
end


# end # module
