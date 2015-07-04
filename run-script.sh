#!/bin/bash
export MAVEN_OPTS="-Xmx2g"

for i in `seq 10 5 95`;
do
	mvn exec:java -Dexec.mainClass=de.tuberlin.dima.aim3.project.IMDbMovieClassification -Dexec.args="-i $i -o res$i --csv -k"
done
