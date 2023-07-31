
for i in {0..4}
do
  sed -i "845s/-0/-${i}/" main.cc
  sed -i "935s/-0/-${i}/" main.cc
  sed -i "953s/-0/-${i}/" main.cc
  sed -i "1026s/-0/-${i}/" main.cc
  sed -i "1097s/-0/-${i}/" main.cc
  sed -i "1172s/-0/-${i}/" main.cc
  sed -i "1449s/-0/-${i}/" main.cc
  sed -i "1588s/= 0;/= ${i};/" main.cc

  make 
  make run &

  sleep 15

  sed -i "845s/-${i}/-0/" main.cc
  sed -i "935s/-${i}/-0/" main.cc
  sed -i "953s/-${i}/-0/" main.cc
  sed -i "1026s/-${i}/-0/" main.cc
  sed -i "1097s/-${i}/-0/" main.cc
  sed -i "1172s/-${i}/-0/" main.cc
  sed -i "1449s/-${i}/-0/" main.cc
  sed -i "1588s/= ${i};/= 0;/" main.cc
done

wait
echo "All processes have finished."