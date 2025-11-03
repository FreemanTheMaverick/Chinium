template<int ndim>
using EigenTensor = Eigen::Tensor<double, ndim>;

template<typename Derived, int ndim> inline EigenTensor<ndim> SliceTensor(
		const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& tensor,
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

template<typename Derived, int ndim> inline EigenTensor<ndim> PadTensor(
		const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& tensor,
		const int (&pairs)[ndim][2]){
	Eigen::array<std::pair<int, int>, ndim> paddings;
	for ( int i = 0; i < ndim; i++ ){
		paddings[i] = std::make_pair(pairs[i][0], pairs[i][1]);
	}
	return tensor.pad(paddings);
}
