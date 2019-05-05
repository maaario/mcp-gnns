# Download benchmark datasets DIMACS and BHOSLIB and convert their graphs from dimacs format
# to plain .in format.

DIMACS_PATH=../data/dimacs

mkdir $DIMACS_PATH

wget -P $DIMACS_PATH http://iridia.ulb.ac.be/~fmascia/files/DIMACS_all_ascii.tar.bz2
tar xjC $DIMACS_PATH -f $DIMACS_PATH/DIMACS_all_ascii.tar.bz2

mkdir $DIMACS_PATH/converted

for i in `ls $DIMACS_PATH/DIMACS_all_ascii/`; do 
    ./convert_dimacs_to_in.sh $DIMACS_PATH/DIMACS_all_ascii/$i $DIMACS_PATH/converted/$i.in
done;



BHOSLIB_PATH=../data/bhoslib

mkdir $BHOSLIB_PATH

wget -P $BHOSLIB_PATH http://iridia.ulb.ac.be/~fmascia/files/BHOSLIB_ascii.tar.bz2
tar xjC $BHOSLIB_PATH -f $BHOSLIB_PATH/BHOSLIB_ascii.tar.bz2

mkdir $BHOSLIB_PATH/converted

for i in `ls $BHOSLIB_PATH/BHOSLIB_ascii/`; do 
    ./convert_dimacs_to_in.sh $BHOSLIB_PATH/BHOSLIB_ascii/$i $BHOSLIB_PATH/converted/$i.in
done;
