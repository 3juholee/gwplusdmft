#!/bin/bash
x=0
while [ $x -le 5 ]
do
	y=0
	while [ $y -le 99 ]
	do
	if [ -d "$x.$y" ]; then
		rm "$x.$y"/imp/nohup*
		rm "$x.$y"/imp/ctqmc
		rm "$x.$y"/imp/broad
		rm "$x.$y"/imp/Gf.out.*
		rm "$x.$y"/imp/Sig.out.*
		rm "$x.$y"/imp/Gloc.dat.*
		rm "$x.$y"/imp/Delta.dat.*
		rm "$x.$y"/imp/status*
	fi
	y=$(($y+1))
	done
x=$(($x+1))
done
