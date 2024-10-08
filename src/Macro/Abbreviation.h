#define EigenVector Eigen::VectorXd
#define EigenMatrix Eigen::MatrixXd
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define Diag(X) (EigenMatrix)(X).diagonal().asDiagonal()
