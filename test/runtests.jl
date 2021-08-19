using Flux
using Random
using Test

Random.seed!(1)
include("../src/EquivariantNeuralOperators.jl")



"""
charge distribution -> potential (Poisson's eqn), electric field (Gauss's law)
"""

# input Scalar field
inranks = [0]
# output scalar field, vector field
outranks = [0, 1]
sz=(3,3,3)
ctr=(2,2,2)
dx = 0.1
# max convolution radius
rmax = dx

# charge distribution
X = [zeros(sz...)]
X[1][ctr...] = 1.0

# generate data
# Green's fn for Poisson, Gauss
f1 = LinearOperator(:potential,dx;rmax=rmax)
f2 = LinearOperator(:field,dx;rmax=rmax)

Y1 = f1(X)
Y2 = f2(X)

# check
ix = ctr.+[1,1,0]
@show Y1[1][ix...]
@show [Y2[i][ix...] for i = 1:3]

# train
# linear layer: tensor field convolution
L = EquivConv(inranks, outranks, dx; rmax = rmax)
##
function nn(X)
    L(X)
end


function loss()
    Y1hat, Y2hat = nn([X])
    l1 = Flux.mae(toArray(Y1), toArray(Y1hat))
    l2 = Flux.mae(toArray(Y2), toArray(Y2hat))
    l = l1 + l2
    println(l)
    l
end
loss()
##
ps = Flux.params(L)
data = [()]
opt = ADAM(0.1)

println("===\nTraining")
for i = 1:5
    # global doplot = i % 50 == 0
    Flux.train!(loss, ps, data, opt)
end

##
Random.seed!(1)
n=4
inranks=[0,0]
outranks=[0]
X=[[rand(n,n,n)],[rand(n,n,n)]]
Y=[[X[1][1].*X[2][1]]]

# train
# linear layer: tensor field convolution

A = EquivAttn(inranks, outranks)

function nn(X)
    A(X)
end


function loss()
    Yhat = nn(X)
    l = Flux.mae(toArray(Y[1]), toArray(Yhat[1]))
    println(l)
    l
end
loss()

ps = Flux.params(A)
data = [()]
opt = ADAM(0.1)

println("===\nTraining")
for i = 1:10
    # global doplot = i % 50 == 0
    Flux.train!(loss, ps, data, opt)
end
