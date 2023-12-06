#include <vector>

class Mwfn{
	public:
		// Field 1
		int Wfntype;
		int Charge;
		int Naelec;
		int Nbelec;
		double E_tot;
		
		// Field 2
		int Ncenter;
		double * Centers;

		// Field 3
		int Nbasis;
		int Nindbasis;

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

		void Save(std::string filename,const bool output);
		Mwfn(std::string filename,const bool output);
};

