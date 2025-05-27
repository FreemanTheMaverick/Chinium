#ifdef EIGEN_CXX11_TENSOR_MODULE_H
template<int ndim> inline Eigen::Tensor<double, ndim> SliceTensor(
		const Eigen::Tensor<double, ndim>& tensor,
		const int (&offsets_)[ndim], const int (&extents_)[ndim]){
	Eigen::array<Eigen::Index, ndim> offsets;
	Eigen::array<Eigen::Index, ndim> extents;
	for ( int i = 0; i < ndim; i++ ){
		offsets[i] = offsets_[i];
		extents[i] = extents_[i];
	}
	return tensor.slice(offsets, extents);
}

#define ScaleTensor(tensor, scalar){\
	for ( long int _kgrid_ = 0; _kgrid_ < tensor.size(); _kgrid_++ )\
		tensor.data()[_kgrid_] *= (scalar);\
}

#endif
