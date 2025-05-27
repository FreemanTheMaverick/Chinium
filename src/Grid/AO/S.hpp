Q[0] = 1;
if (ao1xs){ // (uv)' = u'v + uv'
	Qx[0] = 0;
	Qy[0] = 0;
	Qz[0] = 0;
	if (ao2ls || ao2xxs){
		Qxx[0] = 0;
		Qyy[0] = 0;
		Qzz[0] = 0;
		if (ao2xxs){
			Qxy[0] = 0;
			Qxz[0] = 0;
			Qyz[0] = 0;
			if (ao3xxxs){
				Qxxx[0] = 0;
				Qxxy[0] = 0;
				Qxxz[0] = 0;
				Qxyy[0] = 0;
				Qxyz[0] = 0;
				Qxzz[0] = 0;
				Qyyy[0] = 0;
				Qyyz[0] = 0;
				Qyzz[0] = 0;
				Qzzz[0] = 0;
			}
		}
	}
}
