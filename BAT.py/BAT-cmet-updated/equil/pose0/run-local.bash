#!/bin/bash


pmemd.cuda -O -i mini.in -p full.hmr.prmtop -c full.inpcrd -o mini.out -r mini.rst7 -x mini.nc -ref full.inpcrd
pmemd.cuda -O -i therm1.in -p full.hmr.prmtop -c mini.rst7 -o therm1.out -r therm1.rst7 -x therm1.nc -ref full.inpcrd
pmemd.cuda -O -i therm2.in -p full.hmr.prmtop -c therm1.rst7 -o therm2.out -r therm2.rst7 -x therm2.nc -ref full.inpcrd
pmemd.cuda -O -i eqnpt.in -p full.hmr.prmtop -c therm2.rst7 -o eqnpt.out -r eqnpt.rst7 -x traj000.nc -ref full.inpcrd
pmemd.cuda -O -i eqnpt.in -p full.hmr.prmtop -c eqnpt.rst7 -o eqnpt01.out -r eqnpt01.rst7 -x traj001.nc -ref full.inpcrd
pmemd.cuda -O -i eqnpt.in -p full.hmr.prmtop -c eqnpt01.rst7 -o eqnpt02.out -r eqnpt02.rst7 -x traj002.nc -ref full.inpcrd
pmemd.cuda -O -i eqnpt.in -p full.hmr.prmtop -c eqnpt02.rst7 -o eqnpt03.out -r eqnpt03.rst7 -x traj003.nc -ref full.inpcrd
pmemd.cuda -O -i eqnpt.in -p full.hmr.prmtop -c eqnpt03.rst7 -o eqnpt04.out -r eqnpt04.rst7 -x traj004.nc -ref full.inpcrd
pmemd.cuda -O -i mdin-00 -p full.hmr.prmtop -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref full.inpcrd
i=1
while [ $i -le 8 ] ;
do
j=$(($i - 1))
x=`printf "%02.0f" $i`
y=`printf "%02.0f" $j`
echo "pmemd.cuda -O -i mdin-$x -p full.hmr.prmtop -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref full.inpcrd"
pmemd.cuda -O -i mdin-$x -p full.hmr.prmtop -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref full.inpcrd
let i=$i+1
done
