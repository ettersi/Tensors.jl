# Tensors

Tensors in the sense of this package are multidimensional arrays with the additional twist that indices are distinguished based on labels rather than their position in some linear order. For example, the line 
```julia
x = zeros([Mode(:apple,3), Mode(:orange,4)])
``` 
creates an all-zeros tensor with two *modes* (aka dimensions or indices) of size 3 and 4, respectively. This tensor is equivalent to a 3x4 matrix except that we do not refer to the indices as "first" and "second" but rather as "apple" and "orange". We believe that labelling modes in this manner is both more natural as well as much simpler than imposing an arbitrary order convention on the indices.


## Basic Usage

Create tensor with modes `:a`, `:b`, `:c` of size 4x3x2 and random entries.
```julia
julia> x = rand([Mode(:a,4), Mode(:b,3), Mode(:c,2)])
Tensor{Float64}([Mode(:a,4), Mode(:b,3), Mode(:c,2)])
```
Unfold tensor to a multi-dimensional array.
```julia
julia> x[[:a],[:b],[:c]]
4x3x2 Array{Float64,3}:
[:, :, 1] =
 0.412986   0.286382  0.774888 
 0.0398485  0.85685   0.152157 
 0.360799   0.57589   0.0533906
 0.404407   0.560899  0.24279  

[:, :, 2] =
 0.697664  0.23961   0.741541 
 0.217322  0.343162  0.0774094
 0.514375  0.958773  0.320539 
 0.16762   0.408356  0.484142 
```
The brackets and the order of the arguments are important! Compare the above with 
```julia
julia> x[[:a],[:b,:c]]
4x6 Array{Float64,2}:
 0.412986   0.286382  0.774888   0.697664  0.23961   0.741541 
 0.0398485  0.85685   0.152157   0.217322  0.343162  0.0774094
 0.360799   0.57589   0.0533906  0.514375  0.958773  0.320539 
 0.404407   0.560899  0.24279    0.16762   0.408356  0.484142 

julia> x[[:b,:c],[:a]]
6x4 Array{Float64,2}:
 0.412986  0.0398485  0.360799   0.404407
 0.286382  0.85685    0.57589    0.560899
 0.774888  0.152157   0.0533906  0.24279 
 0.697664  0.217322   0.514375   0.16762 
 0.23961   0.343162   0.958773   0.408356
 0.741541  0.0774094  0.320539   0.484142
```
Access a particular element.
```julia
julia> x[:a => 3, :b => 2, :c => 1]
0.57589
```
Advanced initialisation.
```julia
julia> x = init(i-> i[:a] + 3*(i[:b]-1), Int, [Mode(:a,3), Mode(:b,4)]);
julia> x[[:a],[:b]]
3x4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```

Please note how in the above code snippets we represent modes using different objects depending on the context. To create a tensor, we pass a `Mode` object, that is a pair of a mode label and a mode size. 
```julia
immutable Mode
    mlabel::Any
    msize::Int
end
```
Once we have a tensor, it becomes redundant to specify the mode sizes again hence we only mention the mode labels from then on.


## Mode Product

The higher-dimensional analogue of the matrix product is the *mode product* defined as follows. Let `x`, `y` be two tensors with mode labels `[M;K]` and `[K;N]` where `M`, `K` and `N` are disjoint mode sets. Then, `z = x*y` is a tensor with mode set `[M;N]` defined through `z[M,N] = x[M,K]*y[K,N]` where here the `*` stands for the standard matrix product of the respective unfoldings. 

**Examples**
 - Vector inner product.
```julia
julia> x = mod(rand(Int, [Mode(:a,3)]), 6); 
julia> y = mod(rand(Int, [Mode(:a,3)]), 6); 
julia> println("x = ", x[[:a]])
x = [1,5,4]
julia> println("y = ", y[[:a]])
y = [3,0,0]
julia> println("x*y = ",scalar(x*y))
x*y = 3
```
 - Vector outer product.
```julia
julia> x = mod(rand(Int, [Mode(:a,3)]), 6); 
julia> y = mod(rand(Int, [Mode(:b,3)]), 6);
julia> println("x = ", x[[:a]])
x = [4,5,2]
julia> println("y = ", y[[:b]])
y = [5,0,3]
julia> println("x*y = \n", (x*y)[[:a],[:b]])
x*y = 
[20 0 12
 25 0 15
 10 0 6]
```
 - Matrix vector product.
```julia
julia> A = mod(rand(Int, [Mode(:a,3), Mode(:b,3)]), 6); 
julia> x = mod(rand(Int, [Mode(:b,3)]), 6); 
julia> println("A = \n", A[[:a],[:b]])
A = 
[0 3 0
 2 3 2
 4 0 2]
julia> println("x = ", x[[:b]])
x = [3,5,0]
julia> println("A*x = ", (A*x)[[:a]])
A*x = [15,21,12]
```
 - Frobenius inner product.
```julia
julia> X = mod(rand(Int, [Mode(:a,2), Mode(:b,3)]),6); 
julia> Y = mod(rand(Int, [Mode(:a,2), Mode(:b,3)]),6);
julia> println("X = \n", X[[:a],[:b]])
X = 
[5 0 1
 5 1 4]
julia> println("Y = \n", Y[[:a],[:b]])
Y = 
[0 0 1
 2 5 4]
julia> println("X*Y = ", (X*Y)[])
X*Y = 32
```

The above definition of the mode product involved a little white lie as it suggested that the mode product `x*y` runs over all common modes of `x` and `y`. The actual truth is that a mode `k` of `x` is contracted with a mode `l` of `y` if the predicate `multiplies(k,l)` returns `true`. In most cases, contracting equal modes is the behaviour you want, therefore the default definition is `multiplies(k,l) = (k == l)`. There are situations, however, where different rules are more suitable. 

The particular situation we have in mind are linear operators `A` from a tensor space with mode set `D` onto itself. These operators are naturally tensors with two modes for each mode `k in D`, namely one which is to be contracted with the input and one delivering the mode for the output. In the notation of this package, we distinguish these modes by *tagging* them with a `:C` (for column) or `:R` (for row) tag, respectively. Given a mode symbol `k`, this is done by writing `tag(:C,k)` which wraps `k` in a `Tag{:C}` object. 
```julia
immutable Tag{L} mlabel::Any end
tag(L,k) = Tag{L}(k)
```
For convenience, the `tag()` function is overloaded to work on both `Mode` objects as well as `Vector{Any}` and `Vector{Mode}`. 

The natural rules for matching row and column modes in the mode product are different from the above default. We would like the expression `A*x` to indicate the application of an operator `A` to a tensor `x`, i.e. the column modes of `A` should be multiplied with the corresponding mode of `x` despite the fact that they do not have equal mode labels. Similarly, we want to allow chaining of operators as in `A*B` and right-sided application to vectors as in `x*A`. We thus add the following methods to `multiplies`.
```julia
multiplies(k::Tag{:C}, l::Tag{:R}) = multiplies(k.mlabel, l.mlabel)
multiplies(k::Any    , l::Tag{:R}) = multiplies(k       , l.mlabel)
multiplies(k::Tag{:C}, l::Any    ) = multiplies(k.mlabel, l       )
```
At this point, the expression `y = A*x` involving tensors `A` with modes `[C(D); R(D)]` and `x` with modes `D` would result in a tensor `y` with modes `R(D)` instead of `D`. To resolve this issue, we add the rule that if only either the `R(k)` or `C(k)` mode of a tensor is multiplied, the remaining mode gets renamed to `k`. 

If these rules confuse you at first, do not worry! The key point is that row and column modes behave exactly as you would expect them to, as illustrated in the following example. 
```julia
julia> A = mod(rand(Int, [Mode(k,2) for k in (tag(:R,:a), tag(:C,:a))]), 6); 
julia> b = mod(rand(Int, [Mode(:a,2)]),6); 
julia> println("A = \n", A[[tag(:R,:a)],[tag(:C,:a)]])
A = 
[0 1
 0 5]
julia> println("b = ", b[[:a]])
b = [4,5]
julia> println("A*b = ", (A*b)[:a])
A*b = [5,25]
julia> println("b*A = ", (b*A)[:a])
b*A = [0,29]
```

We so far silently assumed that in a mode product `x*y` there is at most one mode `k` of `x` for every mode `l` of `y` such that `multiplies(k,l)` is true, and vice versa. It is hard to imagine a situation where this rule would not be naturally satisfied, but we would like to warn users that its violation results in undefined behaviour. 


## Tensor Factorisations

This package provides tensor analogues for the QR decomposition and the SVD. As for the mode product, the general pattern of these functions is
 - unfold the tensor into a matrix,
 - compute its matrix decomposition,
 - reshape the results into tensors.

**Tensor QR Decomposition**

Let `x` be a tensor with modes `D`, `M` a subset of `D` and `k` a mode label not in `D`. The expression `q,r = qr(x,M,k)` is defined through `q[setdiff(D,M),[k]], r[[k],M] = qr(x[setdiff(D,M),M])`. 

**Tensor SVD**

Let `x` be a tensor with modes `D`, `M` a subset of `D`, `k` a mode label not in `D` and `rfunc` a function `(::Vector{Real}) -> ::Int`. The expression `u,s,v = svd(x,M,k,rfunc)` is defined through
```julia
U,S,V = svd(x[setdiff(D,M),M])
r = rfunc(S)
u[setdiff(D,M),[k]] = U[:,1:r]
s[[k]] = S[1:r]
v[M,[k]] = V[:,1:r]
```

The following generators for `rfunc` are provided:
 - `fixed(r) = (S) -> r`.
 - `maxrank() = (s) -> length(s)`.
 - `adaptive(eps; rel = true) = (S) -> [ smallest r such that norm(S[r+1:end])/(rel ? norm(S) : 1) <= eps ]`. 


## Examples

### Higher-Order SVD

**References:** 
 - Lieven De Lathauwer, Bart De Moor and Joos Vandewalle. 'A multilinear singular value decomposition'. In: SIAM. J. Matrix Anal. & Appl., 21(4), 1253–1278. URL: <http://dx.doi.org/10.1137/S0895479896305696>

**Definition**
```julia
function hosvd(x, eps)
    eps = eps*norm(x)/sqrt(ndims(x))
    core = x
    factors = Dict{Any,Tensor{eltype(x)}}()
    for k in mlabel(x)
        u,s,v = svd(core, [k], tag(:Rank,k), adaptive(eps, rel=false))
        core = scale(u,s)
        factors[k] = v
    end
    return core,factors
end
```

**Test**

Get a tensor and compute its HOSVD.
```julia
x = rand([Mode(k,4) for k in 1:10])
core,factors = hosvd(x, 0.8)
```
Reassemble the tensor and check accuracy.
```julia
xx = core; for f in values(factors) xx *= f; end 
julia> norm(x - xx)/norm(x)
0.49961117095943813
```
Monitor ranks.
```julia
julia> for k = 1:10 println(k, " => ", msize(core,tag(:Rank,k))); end
1 => 3
2 => 3
3 => 3
4 => 2
5 => 1
6 => 1
7 => 1
8 => 1
9 => 1
10 => 1
```


### Tensor Train Decomposition

**References:**
 - I. V. Oseledets and E. E. Tyrtyshnikov. 'Breaking the curse of dimensionality, or how to use SVD in many dimensions'. In: SIAM J. Sci. Comput., 31(5), 3744–3759. URL: <http://dx.doi.org/10.1137/090748330>
 - I. V. Oseledets. 'Tensor-Train Decomposition'. In: SIAM J. Sci. Comput., 33(5), 2295–2317. URL: <http://dx.doi.org/10.1137/090752286>


**Definition**
```julia
function tt_tensor(x, order, eps)
    @assert Set(mlabel(x)) == Set(order)
    d = ndims(x)
    eps = eps*norm(x)/sqrt(d-1)
    tt = Vector{Tensor{eltype(x)}}(d)
    u,s,v = svd(x, [order[d]], tag(:Rank,d-1), adaptive(eps; rel=false))
    x = scale(u,s); tt[d] = v
    for k in d-1:-1:2
        u,s,v = svd(x, [tag(:Rank,k),order[k]], tag(:Rank,k-1), adaptive(eps; rel=false))
        x = scale(u,s); tt[k] = v
    end
    tt[1] = x
    return tt
end
```

**Test**

Get a tensor and compute its TT decomposition.
```julia
x = rand([Mode(k,4) for k in 1:10])
tt = tt_tensor(x, 1:10, 0.8)
```
Reassemble the tensor and check accuracy.
```julia
julia> norm(x - prod(tt))/norm(x)
0.4995203414908285
```
Monitor ranks.
```julia
julia>  println([msize(tt[k], tag(:Rank,k)) for k in 1:9])
[1,1,1,1,1,1,14,8,3]
```

### Cyclic vs. Tree Structured Tensor Networks

It is known that tensor network formats based on cyclic graphs are in general not closed (see <http://arxiv.org/abs/1105.4449>) and therefore appear to be unsuitable for numerical computations. An immediate follow-up question is whether there is any reason to consider cyclic graphs at all, i.e. whether there are tensors which can be represented more efficiently in cyclic rather than tree-based formats. We next construct a three-dimensional tensor and verify numerically that it can be compressed more efficiently on a triangle than a tree. 

The construction is fairly simple: take a triangle, set all ranks equal to `r` and fill the vertex tensors with random entries. 
```julia
function triangle(r)
    n = r^2
    return [
        rand([Mode(1,n), Mode((1,2),r), Mode((1,3),r)]),
        rand([Mode(2,n), Mode((1,2),r), Mode((2,3),r)]),
        rand([Mode(3,n), Mode((1,3),r), Mode((2,3),r)]),
    ]
end
```
Converting this tensor to a tree requires separating single modes. We expect these separations to have rank `r^2` with probability 1, yet proving this conjecture would require showing linear independence of all slices 
```julia
[
    triangle(r)[1][(1,2) => r12, (1,3) => r13] 
    for r12 = 1:r, r13 = 1:r
]
```
(obvious) and 
```julia
[
    (triangle(r)[2]*triangle(r)[3])[(1,2) => r12, (1,3) => r13] 
    for r12 = 1:r, r13 = 1:r
]
``` 
(not obvious). The second part is easily investigated numerically. 
```julia
conjecture_valid = true
for r = 1:10
    n = r^2
    c2 = rand([Mode(2,n), Mode((1,2),r), Mode((2,3),r)])
    c3 = rand([Mode(3,n), Mode((1,3),r), Mode((2,3),r)])
    u,s,v = svd(c2*c3, [(1,2),(1,3)], tag(:Rank,1), adaptive(1e-3))
    conjecture_valid &= (length(s) == r^2)
end
if conjecture_valid println("The conjecture appears to be valid.")
else println("THE CONJECTURE IS NOT VALID!!!")
end
```
We run this code several times and always obtain the answer `The conjecture appears to be valid.`. We therefore conclude that `prod(triangle(r))` can be represented with `n*r^2` floats in a triangle compared to `2*n*r^2 + n*r^4` in TT or `3*n*r^2 + r^6` in a star. 
