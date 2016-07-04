#!/bin/bash
x=0
while [ $x -le 5 ]
do
	y=0
	while [ $y -le 99 ]
	do
	if [ -d "$x.$y" ]; then
		rm -r g*/"$x.$y"/imp/nohup*
		rm -r g*/"$x.$y"/imp/Delta.dat.*
		rm -r g*/"$x.$y"/imp/Gf.out.*
	fi
	y=$(($y+1))
	done
x=$(($x+1))
done
