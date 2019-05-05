mkdir third_party

mkdir tmp

# wget -P tmp http://iridia.ulb.ac.be/~fmascia/files/mannino.tar.bz2
# tar xjC third_party -f tmp/mannino.tar.bz2 

wget -P tmp http://iridia.ulb.ac.be/~fmascia/files/brockington.tar.bz2
tar xjC third_party -f tmp/brockington.tar.bz2 
gcc third_party/brockington/graphgen.c -o third_party/brockington/graphgen -DSUN -lm

wget -P tmp http://iridia.ulb.ac.be/~fmascia/files/ANSI.tar.bz2
tar xjC third_party -f tmp/ANSI.tar.bz2
cd third_party/ANSI
make
cd ../..

# wget -P tmp http://iridia.ulb.ac.be/~fmascia/files/shor.tar.bz2
# tar xjC third_party -f tmp/shor.tar.bz2
# gcc third_party/shor/shor.c -o third_party/shor/shor

cd third_party
git clone https://github.com/notbad/RB.git
cd ..

rm -r tmp