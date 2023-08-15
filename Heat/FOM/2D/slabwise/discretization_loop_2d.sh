
for i in {0..5}
do
  sed -i "840s/-0/-${i}/" main.cc
  sed -i "920s/-0/-${i}/" main.cc
  sed -i "1079s/-0/-${i}/" main.cc
  sed -i "1109s/-0/-${i}/" main.cc
  sed -i "1305s/-0/-${i}/" main.cc
  sed -i "1416s/= 0;/= ${i};/" main.cc

  make 
  make run &

  sleep 15

  sed -i "840s/-${i}/-0/" main.cc
  sed -i "920s/-${i}/-0/" main.cc
  sed -i "1079s/-${i}/-0/" main.cc
  sed -i "1109s/-${i}/-0/" main.cc
  sed -i "1305s/-${i}/-0/" main.cc
  sed -i "1416s/= ${i};/= 0;/" main.cc
done
