module Tensors

using Base.Cartesian
import Base.==

export 
    Mode, msize, mlabel, Index, index,
    Virtual, splitm, splitm!, mergem, mergem!, resize,
    Tensor, scalartype, mode, msize,
    empty, init, 
    padcat, Tag, tag!, tag, untag!, untag, adaptive, fixed, maxrank


# Typedefs

immutable Mode
    mlabel::Any
    msize::Int
end

immutable Tag{T,U} mlabel::U end

immutable Virtual{T}
    mlabel::T
    scale::Int
end

typealias Index Dict{Any,Int}

type Tensor{T}
    modes::Vector{Mode}
    data::Vector{T}
end


# Output formatting

Base.show(io::IO, k::Mode   ) = (print(io,"Mode("); show(io,mlabel(k)); print(io,",",msize(k),")"))
Base.show{T}(io::IO, k::Tag{T}) = (print(io, T,"("); show(io,k.mlabel); print(io,")"))
Base.show(io::IO, k::Virtual) = (print(io,"Virtual("); show(io,k.mlabel); print(io,",",k.scale")"))
Base.show(io::IO, t::Tensor ) = print(io,"Tensor{",eltype(t),"}([",join(map(string, mode(t)), ", "),"])")


# Mode

msize(k::Mode) = k.msize
msize(K::AbstractVector{Mode}) = length(K) > 0 ? prod(map(msize,K)) : 1
mlabel(k::Mode) = k.mlabel
mlabel(K::AbstractVector{Mode}) = Any[k.mlabel for k in K]
==(k::Mode, l::Mode) = mlabel(k) == mlabel(l) && msize(k) == msize(l)
multiplies(k::Any,l::Any) = k == l
multiplies(k::Mode, l::Mode) = multiplies(mlabel(k), mlabel(l))


# Mode tags

tag(T, k) = Tag{T,typeof(k)}(k)
tag(T, k::Mode) = Mode(tag(T,mlabel(k)),msize(k))
tag{U}(T, K::Vector{U}) = U[tag(T,k) for k in K]
untag(k) = k
untag(k::Tag) = k.mlabel
untag(k::Mode) = Mode(untag(k), msize(k))
untag{U}(K::Vector{U}) = U[untag(T,k) for k in K]
=={T}(k::Tag{T}, l::Tag{T}) = untag(k) == untag(l)

function tag!(t::Tensor, T, modes)
    for k in modes
        i = findfirst(mlabel(t), k)
        if i == 0 error("Could not find mode $k!"); end
        t.modes[i] = tag(T, t.modes[i])
    end
    return t
end
tag(t::Tensor, T, modes) = tag!(copy(t), T, modes)

function untag!(t::Tensor, modes)
    for k in modes
        i = findfirst(untag(t), k)
        if i == 0 error("Could not find mode $k!"); end
        t.modes[i] = untag(t.modes[i])
    end
    return t
end
untag(t::Tensor, modes) = untag!(copy(t), modes)


# Row / column mode tags

multiplies(k::Tag{:C}, l::Tag{:R}) = multiplies(k.mlabel, l.mlabel)
multiplies(k::Any    , l::Tag{:R}) = multiplies(k       , l.mlabel)
multiplies(k::Tag{:C}, l::Any    ) = multiplies(k.mlabel, l       )


# Basic functions

scalartype(x) = scalartype(typeof(x))
scalartype{T}(::Type{Tensor{T}}) = T
Base.eltype{T}(::Type{Tensor{T}}) = T
Base.length(t::Tensor) = length(t.data)
Base.ndims(t::Tensor) = length(t.modes)
Base.size(t::Tensor) = map(msize,t.modes)
Base.size(t::Tensor,k) = msize(t,k)
mode(t::Tensor) = t.modes
mode(t::Tensor, k::Any) = (i = findfirst(mlabel(t),k); i > 0 ? t.modes[i] : throw(ArgumentError("Tensor has no mode $k")))
mode(t::Tensor, K::AbstractVector) = Mode[mode(t,k) for k in K]
mlabel(t::Tensor) = mlabel(t.modes)
msize(t::Tensor) = size(t)
msize(t::Tensor, k) = msize(mode(t,k))
Base.copy(t::Tensor) = Tensor(copy(t.modes), copy(t.data))


# Indexing

function linearindex(D::AbstractVector{Mode}, i::Index)
    n = 1; j = 0
    for k in D
        if !( 1 <= i[mlabel(k)] <= msize(k)) 
            throw(BoundsError(k,i[mlabel(k)]))
        end
        j = j + n*(i[mlabel(k)]-1)
        n *= msize(k)
    end
    return j+1
end
Base.getindex( t::Tensor, i::Index) = t.data[linearindex(t.modes,i)]
Base.setindex!(t::Tensor, ti, i::Index) = (t.data[linearindex(t.modes,i)] = ti; t)
Base.getindex( t::Tensor, i::Pair...) = t[Index(i...)]
Base.setindex!(t::Tensor, ti, i::Pair...) = t[Index(i...)] = ti


# Index iteration

immutable IndexSet D::Vector{Mode} end
index(D::Vector{Mode}) = IndexSet(D)
index(t::Tensor) = index(mode(t))

Base.start(idx::IndexSet) = Index(map(m -> (mlabel(m),1), idx.D))
function Base.next(idx::IndexSet, i) 
    ip = copy(i)
    done = true
    for k in idx.D
        if ip[mlabel(k)] < msize(k) 
            ip[mlabel(k)] += 1
            done = false
            break
        end
        ip[mlabel(k)] = 1
    end
    if done return (i,Index()) end
    return (i,ip)
end
Base.done(idx::IndexSet, i) = isempty(i)
Base.eltype(::Type{IndexSet}) = Index
Base.length(idx::IndexSet) = msize(idx.D)


# Construction and Initialization

empty{T}(::Type{T}, D::Vector{Mode}) = Tensor(D, Array{T}(msize(D)))
empty(D::Vector{Mode}) = empty(Float64, D)
for f in (:zeros, :ones, :rand, :randn)
    @eval begin
        Base.$f{T}(::Type{T}, D::Vector{Mode}) = Tensor(D, $f(T, msize(D)))
        Base.$f(D::Vector{Mode}) = $f(Float64, D)
    end
end
Base.eye{T}(::Type{T}, D::Vector{Mode}) = Tensor(Square(D), vec(eye(T, msize(D))))
Base.eye(D::Vector{Mode}) = eye(Float64, D)
Base.one{T}(::Type{Tensor{T}}) = Tensor(Mode[],ones(T,1))
Base.one(t::Tensor) = one(scalartype(t))

function init{T}(f::Function, ::Type{T}, D::Vector{Mode})
    x = empty(T,D)
    for (il,i) in enumerate(index(D))
        x.data[il] = f(i)
    end
    return x
end
init(f::Function, D::Vector{Mode}) = init(f, Float64, D)


# Storage order permutations

function permutedimsI!(t::Tensor, perm::Vector{Int})
    if length(perm) == 0 return t end
    t.data = vec(permutedims(reshape(t.data,size(t)...),perm))
    t.modes = t.modes[perm]
    return t
end
Base.permutedims!(t::Tensor, newmodes::AbstractVector) = permutedimsI!(t, findpermutation(mlabel(t.modes), newmodes))


# Unfoldings

function Base.getindex(t::Tensor, Ms...)
    permutedims!(t,vcat(Ms...))
    return reshape(t.data, [msize(t,M) for M in Ms]...)
end


# Reshaping

splitm(t::Tensor, s#=::Dict{Any,Vector{Mode}}=#) = splitm!(copy(t),s)
function splitm!(t::Tensor, s#=::Dict{Any,Vector{Mode}}=#) 
    for k in mode(t)
        if msize(k) != msize(s[mlabel(k)]) 
            throw(ArgumentError("Invalid splitting for mode ", mlabel(k), "! (", msize(k), " != ", join("*",[msize(kk) for kk in s[mlabel(k)]]),")"))
        end
    end
    t.modes = vcat([s[mlabel(k)] for k in mode(t)]...)
    return t
end

mergem(t::Tensor, m#=::Dict{Vector, Any}=#) = mergem!(copy(t),m)
function mergem!(t::Tensor, m#=::Dict{Vector, Any}=#) 
    permutedims!(t,vcat([K for K in keys(m)]...))
    n = (k = 1; [msize(t.modes[k:(k+=length(K))-1]) for K in keys(m)])
    t.modes = Mode[Mode(k,nk) for (k,nk) in zip(values(m),n)]
    return t
end

function resize(t::Tensor, n)
    modes = [Mode(mlabel(k), get(n, mlabel(k), msize(k))) for k in t.modes]
    return Tensor(modes, vec(reshape(t.data, [nk for nk in size(t)]...)[[1:msize(k) for k in modes]...]))
end


# Concatenation and padding (for TN sum)

@generated function padcat_{T,N}(x::Array{T,N}, y::Array{T,N}, extendmode::Vector{Bool})
    quote
        z = zeros(T, (collect(size(x)) + extendmode.*collect(size(y)))...)

        stridez_1 = 1; @nexprs $N d->(stridez_{d+1} = stridez_d*size(z, d))
        stridex_1 = 1; @nexprs $N d->(stridex_{d+1} = stridex_d*size(x, d))
        stridey_1 = 1; @nexprs $N d->(stridey_{d+1} = stridey_d*size(y, d))

        $(symbol(:offsetz_, N)) = 1
        $(symbol(:offsetx_, N)) = 1
        $(symbol(:offsety_, N)) = 1

        @nexprs $N d->(nx_d = extendmode[d] ? size(x,d) : 0)

        @nloops $N i x d->(
            offsetz_{d-1} = offsetz_d + (i_d-1)*stridez_d;
            offsetx_{d-1} = offsetx_d + (i_d-1)*stridex_d
        ) begin
            @inbounds z[offsetz_0] = x[offsetx_0]
        end

        @nloops $N i y d->(
            offsetz_{d-1} = offsetz_d + (nx_d+i_d-1)*stridez_d;
            offsety_{d-1} = offsety_d + (i_d-1)*stridey_d
        ) begin
            @inbounds z[offsetz_0] = y[offsety_0]
        end

        return z
    end
end

function padcat(x::Tensor, y::Tensor, extendmodes)
    @assert length(extendmodes) > 0 "Must extend over at least one mode!"
    permutedims!(x, mlabel(y))
    zdata = padcat_(
        reshape(x.data,size(x)...), reshape(y.data,size(y)...),
        [k in extendmodes for k in mlabel(x)]
    )
    return Tensor(
        [Mode(k,n) for (k,n) in zip(mlabel(x), size(zdata))], 
        vec(zdata)
    )
end


# Transposition and conjugation

Base.conj!(t::Tensor) = (conj!(t.data); return t)
Base.conj(t::Tensor) = conj!(copy(t))
function Base.transpose!(t::Tensor)
    for (i,k) in enumerate(t.modes)
        if isa(mlabel(k), Tag{:R}) t.modes[i] = Mode(tag(:C,mlabel(k).mlabel), msize(k))
        elseif isa(mlabel(k), Tag{:C}) t.modes[i] = Mode(tag(:R,mlabel(k).mlabel), msize(k))
        end
    end
    return t
end
Base.transpose(t::Tensor) = transpose!(copy(t))


# Vector arithmetic

for f in (:+,:-)
    @eval begin
        function Base.$f(t1::Tensor, t2::Tensor)
            permutedims!(t2, mlabel(t1))
            return Tensor(copy(t1.modes), $f(t1.data, t2.data))
        end
        Base.$f(t::Tensor) = Tensor(copy(t.modes), $f(t.data))
    end
end

import Base: *, /
*(a::Number, t::Tensor) = Tensor(copy(t.modes), a*t.data)
*(t::Tensor, a::Number) = a*t
/(t::Tensor, a::Number) = Tensor(copy(t.modes), t.data/a)

function Base.dot(t1::Tensor,t2::Tensor) 
    permutedims!(t2, mlabel(t1))
    return dot(t1.data, t2.data)
end
Base.norm(t::Tensor) = vecnorm(t.data)


# Unary elementwise arithmetic

for f in (:abs, :sin, :cos, :tan, :sqrt, :exp, :log)
    @eval (Base.$f)(t::Tensor, args...) = Tensor(copy(t.modes), $f(t.data, args...))
end
for f in (:mod, :mod1)
    @eval (Base.$f)(t::Tensor, m) = Tensor(copy(t.modes), $f(t.data, m))
end
for f in (:floor, :ceil, :base)
    @eval (Base.$f)(t::Tensor, opt::Integer...) = Tensor(copy(t.modes), $f(t.data, opt...))
end


# Binary elementwise arithmetic

for f in (:.*,:./)
    @eval begin
        Base.$f(t1::Tensor, t2::Tensor) = Tensor(copy(t1.modes), $f(t1.data, t2[mlabel(t1)]))
        Base.$f(a::Number, t::Tensor) = Tensor(copy(t.modes), $f(a, t.data))
        Base.$f(t::Tensor, a::Number) = Tensor(copy(t.modes), $f(t.data, a))
    end
end


# Mode product

function matchmodes(M1::Vector{Mode},M2::Vector{Mode})
    deleteindex! = (v::Vector, i) -> (v[i] = v[end]; pop!(v))
    M = Array(Mode,0)
    K1 = Array(Mode,0)
    K2 = Array(Mode,0)
    N = copy(M2)
    for m1 in M1
        i2 = findfirst(m2->multiplies(mlabel(m1),mlabel(m2)), N)
        if i2 > 0
            push!(K1, m1)
            push!(K2, N[i2])
            deleteindex!(N, i2)
        else
            push!(M, m1)
        end
    end
    return M,K1,K2,N
end

function mergemodes(M, K1, K2, N)
    M = copy(M)
    N = copy(N)
    for (k1,k2) in zip(K1,K2)
        if isa(mlabel(k1), Tag{:C}) && !isa(mlabel(k2), Tag{:R})
            l = mlabel(k1).mlabel
            i = findfirst(m -> mlabel(m) == tag(:R,l), M)
            if i > 0 M[i] = Mode(l, msize(M[i])) end
        elseif !isa(mlabel(k1), Tag{:C}) && isa(mlabel(k2), Tag{:R})
            l = mlabel(k2).mlabel
            i = findfirst(m -> mlabel(m) == tag(:C,l), N)
            if i > 0 N[i] = Mode(l, msize(N[i])) end
        end
    end
    return [M;N]
end

function *(t1::Tensor, t2::Tensor)
    M,K1,K2,N = matchmodes(t1.modes,t2.modes)
    return Tensor(mergemodes(M,K1,K2,N), vec(t1[mlabel(M),mlabel(K1)] * t2[mlabel(K2),mlabel(N)]))
end


# Scaling by diagonal matrices

@generated function Base.scale!(R::Array, A::Array, b::Vector...)
    N = length(b)
    quote
        @assert size(R) == size(A)

        stride_1 = 1
        @nexprs $N d->(stride_{d+1} = stride_d*size(A, d))
        $(symbol(:offset_, N)) = 1
        $(symbol(:b_, N)) = 1
        
        @nloops $N i A d->(
            offset_{d-1} = offset_d + (i_d-1)*stride_d;
            b_{d-1} = b[d][i_d]*b_d
        ) begin
            @inbounds R[offset_0] = b_0*A[offset_0]
        end
        return A
    end
end
Base.scale!(A::Array, b::Vector...) = scale!(A,A,b...)
Base.scale(A::Array, b::Vector...) = scale!(Array{eltype(A)}(size(A)),A,b)

function splitAb(t::Tensor...)
    i = max(findfirst(t -> ndims(t) > 1, t), 1)
    A = t[i]
    b = [
        (j = findfirst(t -> multiplies(mlabel(t)[1], mlabel(k)), t[1:i-1])) > 0 ? t[j].data : 
            (j = findfirst(t -> multiplies(mlabel(k), mlabel(t)[1]), t[i+1:end])) > 0 ? t[i+j].data : 
                ones(msize(k))
        for k in mode(A)
    ]
    return A,b
end

function Base.scale!(t::Tensor...)
    A,b = splitAb(t...)
    scale!(reshape(A.data, size(A)...), b...)
    return A
end
function Base.scale(t::Tensor...)
    A,b = splitAb(t...)
    R = Tensor(copy(A.modes), Array{eltype(A)}(length(A)))
    scale!(reshape(R.data, size(A)...), reshape(A.data, size(A)...), b...)
    return R
end


# QR decomposition

function Base.qr(t::Tensor, k::Any, r = maxrank(t,k)) 
    Dmk = setdiff(mlabel(t),[k])
    F = qrfact(t[Dmk,k])
    kk = Mode(k,r)
    return Tensor([mode(t,Dmk);kk], vec(F[:Q]*eye(eltype(t), msize(t,Dmk),r))), 
           Tensor([tag(:R,kk),tag(:C,mode(t,k))], vec(F[:R][1:r,:]))
end
function Base.qr(t::Tensor, K::AbstractVector, k::Any, r = maxrank(t,K))
    DmK = setdiff(mlabel(t),K)
    F = qrfact(t[DmK,K])
    k = Mode(k,r)
    return Tensor([mode(t,DmK);k], vec(F[:Q]*eye(eltype(t), msize(t,DmK),r))), 
           Tensor([k;mode(t,K)], vec(F[:R][1:r,:]))
end


# SVD decomposition

function Base.svd(t::Tensor, k::Any, rank = maxrank())
    Kc = setdiff(mlabel(t), [k])
    F = svdfact(t[Kc,k])
    kk = Mode(k, rank(F[:S]))
    return Tensor([mode(t,Kc); kk], vec(F[:U][:,1:msize(kk)])), 
           Tensor([kk], F[:S][1:msize(kk)]), 
           Tensor([tag(:R,kk);tag(:C,mode(t,k))], vec(F[:Vt][1:msize(kk), :]))
end
function Base.svd(t::Tensor, K::AbstractVector, k::Any, rank = maxrank())
    Kc = setdiff(mlabel(t), K)
    F = svdfact(t[Kc,K])
    k = Mode(k, rank(F[:S]))
    return Tensor([mode(t,Kc); k], vec(F[:U][:,1:msize(k)])), 
           Tensor([k], F[:S][1:msize(k)]), 
           Tensor([k;mode(t,K)], vec(F[:Vt][1:msize(k), :]))
end


# Common rank functions

adaptive(eps; rel = true) = (s) -> max(length(s) - searchsortedlast(cumsum(reverse(s).^2), (eps*(rel ? vecnorm(s) : 1))^2),1)
fixed(r) = (s) -> r
maxrank() = (s) -> length(s)


# Helpers

# Determine perm such that after == before[perm]
function findpermutation(before, after)
    perm = Array(Int,length(after))
    for i in 1:length(after)
        perm[i] = findfirst(before, after[i]) 
    end
    return perm
end

maxrank(t::Tensor, K) = min(div(length(t),msize(t,K)), msize(t,K))

end # module
