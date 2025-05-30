/*
Generated by EigenEinSum

Recommended filename:
Ws_g...E2Sigma2s_g...SigmaGrads_g,t,a...SigmaGrads_g,s,b---H_t,a,s,b.hpp

Einsum expression:
Ws(g), E2Sigma2s(g), SigmaGrads(g, t, a), SigmaGrads(g, s, b) -> H(t, a, s, b)

The einsum expression is decomposed into:
Ws(g), E2Sigma2s(g) -> WsE2Sigma2s(g)
WsE2Sigma2s(g), SigmaGrads(g, t, a) -> WsE2Sigma2sSigmaGrads(g, t, a)
WsE2Sigma2sSigmaGrads(g, t, a), SigmaGrads(g, s, b) -> H(t, a, s, b)

The index paths are derived to be:
TOP
├── g
└── a
    └── t
        ├── g
        └── b
            └── s
                └── g
*/
{
	const int t_len = SigmaGrads.dimensions()[1];
	assert( t_len == H.dimensions()[0] );
	const int a_len = SigmaGrads.dimensions()[2];
	assert( a_len == H.dimensions()[1] );
	const int s_len = SigmaGrads.dimensions()[1];
	assert( s_len == H.dimensions()[2] );
	const int b_len = SigmaGrads.dimensions()[2];
	assert( b_len == H.dimensions()[3] );
	const int g_len = Ws.dimensions()[0];
	assert( g_len == E2Sigma2s.dimensions()[0] );
	assert( g_len == SigmaGrads.dimensions()[0] );
	assert( g_len == SigmaGrads.dimensions()[0] );
	Eigen::Tensor<double, 1> WsE2Sigma2s(g_len);
	WsE2Sigma2s.setZero();
	Eigen::Tensor<double, 1> WsE2Sigma2sSigmaGradsat(g_len);
	for ( int g = 0; g < g_len; g++ ){
		WsE2Sigma2s(g) += Ws(g) * E2Sigma2s(g);
	}
	for ( int a = 0; a < a_len; a++ ){
		for ( int t = 0; t < t_len; t++ ){
			WsE2Sigma2sSigmaGradsat.setZero();
			for ( int g = 0; g < g_len; g++ ){
				WsE2Sigma2sSigmaGradsat(g) += WsE2Sigma2s(g) * SigmaGrads(g, t, a);
			}
			for ( int b = 0; b < b_len; b++ ){
				for ( int s = 0; s < s_len; s++ ){
					for ( int g = 0; g < g_len; g++ ){
						H(t, a, s, b) += WsE2Sigma2sSigmaGradsat(g) * SigmaGrads(g, s, b);
					}
				}
			}
		}
	}
}
