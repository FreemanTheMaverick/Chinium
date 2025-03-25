#############################################################
# gau_cnm.sh is an interface to Gaussian geometry optimizer #
#############################################################


#########
# Usage #
#########

# 1. Create an EMPTY folder for this job and cd into it.
# 2. Create a "template" to tell Chinium which model to use. This template file resembles a normal Chinium input file except for the absense of xyz, guess, charge, spin multiplicity and derivative sections. The missing coordinate information and others will be obtained from Gaussian input file.
# 3. Create a Gaussian input file with the keyword external. For example, an optimization job input file should look like:
# %nproc=1 ! Only 1 thread is needed, because intensive computation is done by Chinium instead of Gaussian.
# %chk=job.chk ! The final geometry will be stored in this file, but the wavefunction will not.
# # opt(calcfc,micro) freq nosym external='sh $CHINIUM_PATH/tools/gau_cnm.sh template job.mwfn' ! The option micro and the keyword nosym are NECESSARY. "job.mwfn" will be used to store the final wavefunction. If "job.mwfn" exists when this job starts, it will be read as initial guess for the first frame.
# [ Blank line ]
# Job title
# [ Blank line ]
# 0 1
# [ Molecular Coordinates ]
# [ ... ]
# An example on an SN2 reaction can be found at $CHINIUM_PATH/tools/.


#######################
# Location of Chinium #
#######################

Chinium=$CHINIUM_PATH/Chinium


##############
# Parameters #
##############

# Wavefunction file name
template=$1
mwfn=$2

# Files interfacing to Gaussian
layer=$3
input_file=$4
output_file=$5
msg_file=$6

# Retrieving information from the first line of input_file
natoms=`awk 'NR==1{print $1}' $input_file`
derivative=`awk 'NR==1{print $2}' $input_file`
charge=`awk 'NR==1{print $3}' $input_file`
spin=`awk 'NR==1{print $4}' $input_file`

# Atomic coordinates
elements=(fuck_index_from_zero H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra)
atoms=""
for i in `seq 2 $(($natoms+1))`; do
	Z=`awk 'NR=='$i'{print $1}' $input_file`
	x=`awk 'NR=='$i'{print $2}' $input_file`
	y=`awk 'NR=='$i'{print $3}' $input_file`
	z=`awk 'NR=='$i'{print $4}' $input_file`
	x=`awk -v x=$x 'BEGIN {printf "%.8f",x/1.8897259886}'` 
	y=`awk -v y=$y 'BEGIN {printf "%.8f",y/1.8897259886}'` 
	z=`awk -v z=$z 'BEGIN {printf "%.8f",z/1.8897259886}'` 
	atoms="$atoms${elements[$Z]} $x $y $z\n"
done


############################################
# Finding current step index and step name #
############################################

step_index=""
step_name=""
dir=`ls -F | grep "/$" | grep step_ | sed 's/\///g'`
if [ -z "$dir" ]; then
	step_index=0
	step_name=step_0000
else
	step_index=`ls -F | grep "/$" | grep step_ | sed 's/\///g' | sort -r | sed -n 1p | cut -d '_' -f 2 | sed -r 's/0*([0-9])/\1/'`
	step_index=$(($step_index+1))
	step_name=step_`printf '%04d\n' $step_index`
fi
mkdir $step_name
cd $step_name


#############################
# Writing input for Chinium #
#############################

cp ../$template ${step_name}.inp
printf "\n" >> ${step_name}.inp

printf "xyz\n" >> ${step_name}.inp
printf "%d\n" $natoms >> ${step_name}.inp
printf "$atoms" >> ${step_name}.inp
printf "\n" >> ${step_name}.inp

printf "derivative\n" >> ${step_name}.inp
printf "%d\n" $derivative >> ${step_name}.inp
printf "\n" >> ${step_name}.inp

printf "charge\n" >> ${step_name}.inp
printf "%d\n" $charge >> ${step_name}.inp
printf "\n" >> ${step_name}.inp

printf "spin\n" >> ${step_name}.inp
printf "%d\n" $spin >> ${step_name}.inp
printf "\n" >> ${step_name}.inp

printf "guess\n" >> ${step_name}.inp
if [ $step_index -eq 0 ]&&[ ! -f ../$mwfn ]; then
	printf "sap\n" >> ${step_name}.inp
else
	printf "read\n" >> ${step_name}.inp
	cp ../$mwfn ${step_name}.mwfn
fi


###################
# Running Chinium #
###################

$Chinium ${step_name}.inp > ${step_name}.out 2>&1
cp ${step_name}.mwfn ../$mwfn


###############################
# Writing output for Gaussian #
###############################

# Energy
grep 'Total energy:' ${step_name}.out | awk '{printf "%20.12e",$3}' > $output_file

# Dipole moment, which is not given
printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file

# Gradient
line=`grep -n 'Total nuclear gradient:' ${step_name}.out | cut -d ':' -f 1`
for i in `seq 1 $natoms`; do
        awk 'NR=='$(($line+$i))'{printf "%20.12e%20.12e%20.12e\n",$4,$5,$6}' ${step_name}.out >> $output_file
done

# Polarizability, which is not given
printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file
printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file

# Dipole derivatives, which are not given
for i in `seq 1 $natoms`; do
	printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file
	printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file
	printf "%20.12e%20.12e%20.12e\n" 0 0 0 >> $output_file
done

# Hessian
if [ $derivative -eq 2 ]; then
	line=`grep -n 'Total nuclear hessian:' ${step_name}.out | cut -d ':' -f 1`
	xypert=1
	for xpert in `seq 1 $((3*$natoms))`; do
		thisline=`sed -n $(($line+$xpert))p ${step_name}.out`
		for ypert in `seq 1 $xpert`; do
			word=`echo $thisline | tr -s ' ' | cut -d ' ' -f $(($ypert+1))`
			printf "%20.12e" $word >> $output_file
			if [ $(($xypert%3)) -eq 0 ]; then
				printf "\n" >> $output_file
			fi
			xypert=$(($xypert+1))
		done
	done
fi
