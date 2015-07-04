#!/bin/bash
declare -a arr=("Comedy" "Drama" "Action" "Documentary" "Adult" "Romance" "Thriller" "Animation" "Family" "Horror" "Music" "Crime" "Adventure" "Fantasy" "Sci-Fi" "Mystery" "Biography" "History" "Sport" "Musical" "War" "Western" "News" "Reality-TV")

for i in "${arr[@]}"
#for i in `seq 0 5 100`;
do
	#cat res$i/* > res$i.txt
	grep $i res*.txt | sed "s/,/;/g" > rescollected/tmp.csv
	./transpose.py < rescollected/tmp.csv | sed -e 's/^/'"$i"';/' | sed '1d' > rescollected/$i.csv
done

cat rescollected/* > rescollected.csv
