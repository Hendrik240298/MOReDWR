
for i in {0..6}
do
  sed -i "842s/-0/-${i}/" main.cc
  sed -i "922s/-0/-${i}/" main.cc
  sed -i "1081s/-0/-${i}/" main.cc
  sed -i "1111s/-0/-${i}/" main.cc
  sed -i "1307s/-0/-${i}/" main.cc
  sed -i "1418s/= 0;/= ${i};/" main.cc

  make 
  make run &

  sleep 15

  sed -i "842s/-${i}/-0/" main.cc
  sed -i "922s/-${i}/-0/" main.cc
  sed -i "1081s/-${i}/-0/" main.cc
  sed -i "1111s/-${i}/-0/" main.cc
  sed -i "1307s/-${i}/-0/" main.cc
  sed -i "1418s/= ${i};/= 0;/" main.cc
done

wait
echo "All processes have finished."