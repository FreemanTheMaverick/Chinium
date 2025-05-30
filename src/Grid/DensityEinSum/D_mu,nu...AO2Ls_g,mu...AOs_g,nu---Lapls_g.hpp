/*
Generated by EigenEinSum

Recommended filename:
D_mu,nu...AO2Ls_g,mu...AOs_g,nu---Lapls_g.hpp

Einsum expression:
D(mu, nu), AO2Ls(g, mu), AOs(g, nu) -> Lapls(g)

The einsum expression is decomposed into:
D(mu, nu), AO2Ls(g, mu) -> DAO2Ls(g, nu)
DAO2Ls(g, nu), AOs(g, nu) -> Lapls(g)

The index paths are derived to be:
TOP
└── nu
    ├── mu
    │   └── g
    └── g
*/
{
	[[maybe_unused]] const int g_len = AO2Ls.dimension(0);
	assert( g_len == AOs.dimension(0) );
	assert( g_len == Lapls.dimension(0) );
	[[maybe_unused]] const int mu_len = D.dimension(0);
	assert( mu_len == AO2Ls.dimension(1) );
	[[maybe_unused]] const int nu_len = D.dimension(1);
	assert( nu_len == AOs.dimension(1) );
	Eigen::Tensor<double, 1> DAO2Lsnu(g_len);
	for ( int nu = 0; nu < nu_len; nu++ ){
		DAO2Lsnu.setZero();
		for ( int mu = 0; mu < mu_len; mu++ ){
			for ( int g = 0; g < g_len; g++ ){
				DAO2Lsnu(g) += D(mu, nu) * AO2Ls(g, mu);
			}
		}
		for ( int g = 0; g < g_len; g++ ){
			Lapls(g) += DAO2Lsnu(g) * AOs(g, nu);
		}
	}
}
