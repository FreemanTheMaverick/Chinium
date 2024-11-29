#define EigenArray Eigen::ArrayXd
#define EigenVector Eigen::VectorXd
#define EigenDiagonal Eigen::DiagonalMatrix<double,-1,-1>
#define EigenMatrix Eigen::MatrixXd
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define Diag(X) (X).diagonal().asDiagonal()
#define Dot(X, Y) ( (X).transpose() * (Y) ).trace()
