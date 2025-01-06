#include <vector>

class Mwfn{
	public:
		// Field 1
		int Wfntype=-114514;
		int Charge=-114514;
		int Naelec=-114514;
		int Nbelec=-114514;
		double E_tot=-114514;
		double VT_ratio=-114514;
		
		// Field 2
		int Ncenter=-114514;
		std::vector<double> Centers;

		// Field 3
		int Nbasis=-114514;
		int Nindbasis=-114514;
		int Nprims=-114514;
		int Nshell=-114514;
		int Nprimshell=-114514;
		std::vector<int> Shell_types;
		std::vector<int> Shell_centers;
		std::vector<int> Shell_contraction_degrees;
		std::vector<double> Primitive_exponents;
		std::vector<double> Contraction_coefficients;

		// Field 4
		std::vector<int> Type;
		EigenVector Energy;
		EigenVector Occ;
		std::vector<std::string> Sym;
		EigenMatrix Coeff;

		// Field 5
		EigenMatrix Total_density_matrix;
		EigenMatrix Hamiltonian_matrix;
		EigenMatrix Overlap_matrix;
		EigenMatrix Kinetic_energy_matrix;
		EigenMatrix Potential_energy_matrix;

		void Export(std::string filename,const bool output);
		Mwfn(std::string filename,const bool output);
};

