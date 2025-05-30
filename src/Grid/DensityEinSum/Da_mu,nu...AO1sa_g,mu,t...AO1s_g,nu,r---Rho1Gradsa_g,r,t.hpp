/*
Generated by EigenEinSum

Recommended filename:
Da_mu,nu...AO1sa_g,mu,t...AO1s_g,nu,r---Rho1Gradsa_g,r,t.hpp

Einsum expression:
Da(mu, nu), AO1sa(g, mu, t), AO1s(g, nu, r) -> Rho1Gradsa(g, r, t)

The einsum expression is decomposed into:
Da(mu, nu), AO1sa(g, mu, t) -> DaAO1sa(g, t, nu)
DaAO1sa(g, t, nu), AO1s(g, nu, r) -> Rho1Gradsa(g, r, t)

The index paths are derived to be:
TOP
└── nu
    ├── t
    │   └── mu
    │       └── g
    └── r
        └── t
            └── g
*/
{
	[[maybe_unused]] const int g_len = AO1sa.dimension(0);
	assert( g_len == AO1s.dimension(0) );
	assert( g_len == Rho1Gradsa.dimension(0) );
	[[maybe_unused]] const int r_len = AO1s.dimension(2);
	assert( r_len == Rho1Gradsa.dimension(1) );
	[[maybe_unused]] const int t_len = AO1sa.dimension(2);
	assert( t_len == Rho1Gradsa.dimension(2) );
	[[maybe_unused]] const int mu_len = Da.dimension(0);
	assert( mu_len == AO1sa.dimension(1) );
	[[maybe_unused]] const int nu_len = Da.dimension(1);
	assert( nu_len == AO1s.dimension(1) );
	Eigen::Tensor<double, 2> DaAO1sanu(g_len, t_len);
	for ( int nu = 0; nu < nu_len; nu++ ){
		DaAO1sanu.setZero();
		for ( int t = 0; t < t_len; t++ ){
			for ( int mu = 0; mu < mu_len; mu++ ){
				for ( int g = 0; g < g_len; g++ ){
					DaAO1sanu(g, t) += Da(mu, nu) * AO1sa(g, mu, t);
				}
			}
		}
		for ( int r = 0; r < r_len; r++ ){
			for ( int t = 0; t < t_len; t++ ){
				for ( int g = 0; g < g_len; g++ ){
					Rho1Gradsa(g, r, t) += DaAO1sanu(g, t) * AO1s(g, nu, r);
				}
			}
		}
	}
}
