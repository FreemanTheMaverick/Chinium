Q[0] = x;
Q[1] = y;
Q[2] = z;
if (ao1xs){
	Qx[0] = 1;
	Qy[0] = 0;
	Qz[0] = 0;
	Qx[1] = 0;
	Qy[1] = 1;
	Qz[1] = 0;
	Qx[2] = 0;
	Qy[2] = 0;
	Qz[2] = 1;
	if (ao2ls || ao2xxs){
		Qxx[0] = 0;
		Qyy[0] = 0;
		Qzz[0] = 0;
		Qxx[1] = 0;
		Qyy[1] = 0;
		Qzz[1] = 0;
		Qxx[2] = 0;
		Qyy[2] = 0;
		Qzz[2] = 0;
		if (ao2xxs){
			Qxy[0] = 0;
			Qxz[0] = 0;
			Qyz[0] = 0;
			Qxy[1] = 0;
			Qxz[1] = 0;
			Qyz[1] = 0;
			Qxy[2] = 0;
			Qxz[2] = 0;
			Qyz[2] = 0;
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
				Qxxx[1] = 0;
				Qxxy[1] = 0;
				Qxxz[1] = 0;
				Qxyy[1] = 0;
				Qxyz[1] = 0;
				Qxzz[1] = 0;
				Qyyy[1] = 0;
				Qyyz[1] = 0;
				Qyzz[1] = 0;
				Qzzz[1] = 0;
				Qxxx[2] = 0;
				Qxxy[2] = 0;
				Qxxz[2] = 0;
				Qxyy[2] = 0;
				Qxyz[2] = 0;
				Qxzz[2] = 0;
				Qyyy[2] = 0;
				Qyyz[2] = 0;
				Qyzz[2] = 0;
				Qzzz[2] = 0;
			}
		}
	}
}
