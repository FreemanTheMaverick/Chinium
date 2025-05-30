/*
Generated by EigenEinSum

Recommended filename:
Daa_mu,nu...AO1sa_g,mu,t...AO1sa_g,nu,s---RhoHesssaa_g,t,s.hpp

Einsum expression:
Daa(mu, nu), AO1sa(g, mu, t), AO1sa(g, nu, s) -> RhoHesssaa(g, t, s)

The einsum expression is decomposed into:
Daa(mu, nu), AO1sa(g, mu, t) -> DaaAO1sa(g, t, nu)
DaaAO1sa(g, t, nu), AO1sa(g, nu, s) -> RhoHesssaa(g, t, s)

The index paths are derived to be:
TOP
└── nu
    ├── mu
    │   └── t
    │       └── g
    └── t
        └── s
            └── g
*/
{
	const int g_len = AO1sa.dimensions()[0];
	assert( g_len == AO1sa.dimensions()[0] );
	assert( g_len == RhoHesssaa.dimensions()[0] );
	const int t_len = AO1sa.dimensions()[2];
	assert( t_len == RhoHesssaa.dimensions()[1] );
	const int s_len = AO1sa.dimensions()[2];
	assert( s_len == RhoHesssaa.dimensions()[2] );
	const int mu_len = Daa.dimensions()[0];
	assert( mu_len == AO1sa.dimensions()[1] );
	const int nu_len = Daa.dimensions()[1];
	assert( nu_len == AO1sa.dimensions()[1] );
	Eigen::Tensor<double, 2> DaaAO1sanu(g_len, t_len);
	for ( int nu = 0; nu < nu_len; nu++ ){
		DaaAO1sanu.setZero();
		for ( int mu = 0; mu < mu_len; mu++ ){
			for ( int t = 0; t < t_len; t++ ){
				for ( int g = 0; g < g_len; g++ ){
					DaaAO1sanu(g, t) += Daa(mu, nu) * AO1sa(g, mu, t);
				}
			}
		}
		for ( int t = 0; t < t_len; t++ ){
			for ( int s = 0; s < s_len; s++ ){
				for ( int g = 0; g < g_len; g++ ){
					RhoHesssaa(g, t, s) += DaaAO1sanu(g, t) * AO1sa(g, nu, s);
				}
			}
		}
	}
}
