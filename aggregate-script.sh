#!/bin/bash
declare -a arr=("Comedy" "Drama" "Action" "Documentary" "Adult" "Romance" "Thriller" "Animation" "Family" "Horror" "Music" "Crime" "Adventure" "Fantasy" "Sci-Fi" "Mystery" "Biography" "History" "Sport" "Musical" "War" "Western" "News" "Reality-TV")

mkdir rescollected

for i in `seq 0 5 100`;
do
	cat res$i/* > res$i.txt
done

for i in "${arr[@]}"
do
	grep $i, res*.txt | sed "s/,/;/g" > rescollected/$i.tmp.csv
	./transpose.py < rescollected/$i.tmp.csv | sed -e 's/^/'"$i"';/' | sed '1d' > rescollected/$i.csv
	rm rescollected/$i.tmp.csv
done

cat rescollected/* > rescollected.csv
