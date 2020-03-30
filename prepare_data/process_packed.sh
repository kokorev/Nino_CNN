#!/bin/sh
# This script prepare ORAS5 data for use ENSO prediction model
# Depends on CDO https://code.mpimet.mpg.de/projects/cdo

for var in {sosstsst,sohtc300}; do
for i in {0..4}; do
	opa=opa$i
	p=$var/$opa
	for a in `ls -1 $p/*.tar.gz`; do tar -xzf $a -C $p; done
	cdo genbil,n16 $(ls -t $p/*.nc | head -n 1) weights.tmp
	mkdir ./$p/rg
	for f in ./$p/*.nc
	    do cdo remap,n16,weights.tmp $f ./$p/rg/$(basename $f)
	done
	f=$var\_$opa.nc
	cdo mergetime ./$p/rg/*.nc ./$f
	cdo sub $f -ymonmean $f anom_$f
	cdo div -sub anom_$f -timmin anom_$f -sub -timmax anom_$f -timmin anom_$f norm_anom_$f
	cdo detrend $f dt_$f
	cdo div -sub dt_$f -timmin dt_$f -sub -timmax dt_$f -timmin dt_$f norm_dt_$f
	rm $p/*.nc
	rm -rf $p/rg/
done
done
