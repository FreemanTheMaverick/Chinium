/*
Generated by EigenEinSum

Recommended filename:
Dab_mu,nu...AO1sa_g,mu,t...AO1sb_g,nu,s---RhoHesssab_g,t,s.hpp

Einsum expression:
Dab(mu, nu), AO1sa(g, mu, t), AO1sb(g, nu, s) -> RhoHesssab(g, t, s)

The einsum expression is decomposed into:
Dab(mu, nu), AO1sa(g, mu, t) -> DabAO1sa(g, t, nu)
DabAO1sa(g, t, nu), AO1sb(g, nu, s) -> RhoHesssab(g, t, s)

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
	assert( g_len == AO1sb.dimensions()[0] );
	assert( g_len == RhoHesssab.dimensions()[0] );
	const int t_len = AO1sa.dimensions()[2];
	assert( t_len == RhoHesssab.dimensions()[1] );
	const int s_len = AO1sb.dimensions()[2];
	assert( s_len == RhoHesssab.dimensions()[2] );
	const int mu_len = Dab.dimensions()[0];
	assert( mu_len == AO1sa.dimensions()[1] );
	const int nu_len = Dab.dimensions()[1];
	assert( nu_len == AO1sb.dimensions()[1] );
	Eigen::Tensor<double, 2> DabAO1sanu(g_len, t_len);
	for ( int nu = 0; nu < nu_len; nu++ ){
		DabAO1sanu.setZero();
		for ( int mu = 0; mu < mu_len; mu++ ){
			for ( int t = 0; t < t_len; t++ ){
				for ( int g = 0; g < g_len; g++ ){
					DabAO1sanu(g, t) += Dab(mu, nu) * AO1sa(g, mu, t);
				}
			}
		}
		for ( int t = 0; t < t_len; t++ ){
			for ( int s = 0; s < s_len; s++ ){
				for ( int g = 0; g < g_len; g++ ){
					RhoHesssab(g, t, s) += DabAO1sanu(g, t) * AO1sb(g, nu, s);
				}
			}
		}
	}
}
