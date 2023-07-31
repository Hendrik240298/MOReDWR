

for i in {0..6}
do
  sed -i "778s/-0/-${i}/" main.cc
  sed -i "858s/-0/-${i}/" main.cc
  sed -i "1017s/-0/-${i}/" main.cc
  sed -i "1213s/-0/-${i}/" main.cc
  sed -i "1326s/= 0;/= ${i};/" main.cc

  make 
  make run 

  sed -i "778s/-${i}/-0/" main.cc
  sed -i "858s/-${i}/-0/" main.cc
  sed -i "1017s/-${i}/-0/" main.cc
  sed -i "1213s/-${i}/-0/" main.cc
  sed -i "1326s/= ${i};/= 0;/" main.cc
done
