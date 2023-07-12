#include <Eigen/Dense>
#include "Aliases.h"

EigenMatrix GramSchmidt(EigenMatrix target){
	EigenMatrix result=target;
	for (int i=0;i<target.cols();i++){
		for (int j=0;j<i;j++)
			result.col(i)-=target.col(i).transpose()*result.col(j)*result.col(j)/result.col(j).norm()/result.col(j).norm();
		result.col(i)/=result.col(i).norm();
	}
	return result;
}

/*
#include <iostream>
int main(){
	EigenMatrix target(3,3);target<<1,3,5,7,9,2,4,6,8;
	std::cout<<GramSchmidt(target)<<std::endl;
}
*/
