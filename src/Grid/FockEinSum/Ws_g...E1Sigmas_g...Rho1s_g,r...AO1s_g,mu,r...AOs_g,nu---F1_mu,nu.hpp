/*
Generated by EigenEinSum

Recommended filename:
Ws_g...E1Sigmas_g...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu.hpp

Einsum expression:
Ws(g), E1Sigmas(g), Rho1s(g, r), AO1s(g, mu, r), AOs(g, nu) -> F1(mu, nu)

The einsum expression is decomposed into:
Ws(g), E1Sigmas(g) -> WsE1Sigmas(g)
WsE1Sigmas(g), Rho1s(g, r) -> WsE1SigmasRho1s(g, r)
WsE1SigmasRho1s(g, r), AO1s(g, mu, r) -> WsE1SigmasRho1sAO1s(g, mu)
WsE1SigmasRho1sAO1s(g, mu), AOs(g, nu) -> F1(mu, nu)

The index paths are derived to be:
TOP
├── g
├── r
│   └── g
└── mu
    ├── r
    │   └── g
    └── nu
        └── g
*/
{
	[[maybe_unused]] const int mu_len = AO1s.dimension(1);
	assert( mu_len == F1.dimension(0) );
	[[maybe_unused]] const int nu_len = AOs.dimension(1);
	assert( nu_len == F1.dimension(1) );
	[[maybe_unused]] const int g_len = Ws.dimension(0);
	assert( g_len == E1Sigmas.dimension(0) );
	assert( g_len == Rho1s.dimension(0) );
	assert( g_len == AO1s.dimension(0) );
	assert( g_len == AOs.dimension(0) );
	[[maybe_unused]] const int r_len = Rho1s.dimension(1);
	assert( r_len == AO1s.dimension(2) );
	Eigen::Tensor<double, 1> WsE1Sigmas(g_len);
	Eigen::Tensor<double, 2> WsE1SigmasRho1s(g_len, r_len);
	WsE1Sigmas.setZero();
	WsE1SigmasRho1s.setZero();
	Eigen::Tensor<double, 1> WsE1SigmasRho1sAO1smu(g_len);
	for ( int g = 0; g < g_len; g++ ){
		WsE1Sigmas(g) += Ws(g) * E1Sigmas(g);
	}
	for ( int r = 0; r < r_len; r++ ){
		for ( int g = 0; g < g_len; g++ ){
			WsE1SigmasRho1s(g, r) += WsE1Sigmas(g) * Rho1s(g, r);
		}
	}
	for ( int mu = 0; mu < mu_len; mu++ ){
		WsE1SigmasRho1sAO1smu.setZero();
		for ( int r = 0; r < r_len; r++ ){
			for ( int g = 0; g < g_len; g++ ){
				WsE1SigmasRho1sAO1smu(g) += WsE1SigmasRho1s(g, r) * AO1s(g, mu, r);
			}
		}
		for ( int nu = 0; nu < nu_len; nu++ ){
			for ( int g = 0; g < g_len; g++ ){
				F1(mu, nu) += WsE1SigmasRho1sAO1smu(g) * AOs(g, nu);
			}
		}
	}
}
