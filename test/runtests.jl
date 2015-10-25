using Base.Test
using Tensors


# Single mode QR

t = rand([Mode(k,2) for k in 1:4])
q,r = qr(t, 3)
@test_approx_eq_eps(norm(t - q*r), 0, 1e-12)


# Multiple modes QR

t = rand([Mode(k,2) for k in 1:4])
q,r = qr(t, Any[3,1], 5)
@test_approx_eq_eps(norm(t - q*r), 0, 1e-12)


# Single mode SVD

t = rand([Mode(k,2) for k in 1:4])
u,s,v = svd(t, 3)
@test_approx_eq_eps(norm(t - u*scale(s,v)), 0, 1e-12)


# Multiple modes SVD

t = rand([Mode(k,2) for k in 1:4])
u,s,v = svd(t, Any[4,2], 5)
@test_approx_eq_eps(norm(t - scale(u,s)*v), 0, 1e-12)
