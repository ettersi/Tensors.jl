using Base.Test
using Tensors


# Single mode QR

t = rand([Mode(k,2) for k in 1:4])
q,r = qr(t, 3)
@test t ≈ q*r


# Multiple modes QR

t = rand([Mode(k,2) for k in 1:4])
q,r = qr(t, Any[3,1], 5)
@test t ≈ q*r


# Single mode SVD

t = rand([Mode(k,2) for k in 1:4])
u,s,v = svd(t, 3)
@test t ≈ u*scale(s,v)


# Multiple modes SVD

t = rand([Mode(k,2) for k in 1:4])
u,s,v = svd(t, Any[4,2], 5)
@test t ≈ scale(u,s)*v
