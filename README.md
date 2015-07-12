# IMDb Movie Classification

Study project by Florian Gößler (@FGoessler) & Xi Yang(@cici-debug) @ [TU Berlin](https://github.com/TU-Berlin-DIMA) for the course ["Advanced Information Management 3 - Scaleable Data Analytics and Data Mining"](http://www.dima.tu-berlin.de/menue/studium_und_lehre/masterstudium/aim-3/)


## What is this about?

This is a small study project to classify movies based on their plot, keywords and actors.
The detailed paper can be found here: (IN PROGRESS!)
We're using [Apache Spark](https://spark.apache.org) and data from the [International Movie Database (IMDb)](http://www.imdb.com).


## Getting started

1. Clone this project and build via Maven or open it with your favorite IDE (IntelliJ Idea right ;-))
2. Download the IMDb datasets. You have two options:
    - Download from [IMDb](http://www.imdb.com/interfaces) directly and preprocess it yourself (see convert the data below).
    OR:
    - Download already preprocessed data from our public Dropbox folder (Link comes soon)
3. Run the program via maven exec ```mvn exec:java -Dexec.mainClass=de.tuberlin.dima.aim3.project.IMDbMovieClassification```. For available command line options see below or run it with ```--help```.
 
 
## Data Preprocessing
 
**Genres and plot files:**
Use the preprocessing.py script.
It downloads the two files, stripes headers and converts the plot file in a Spark compatible one-line-per-movie format.

**Keywords file:**
Download the keywords.list file from the FTP server and just delete the very long header and list of included keywords.

**Actors file:**
Use the ```ActorsConverter.scala``` class to convert your downloaded actors.list file.

 
## Command Line Options
 ```
   -s <value> | --sampling <value>
         Input data sampling rate - default 1.0 (all). Please note that when using multiple features the sampling rate is applied per feature. Since only movies with valid vectors for all features are evaluated the total amount of sampled movies that get evaluated might be smaller.
   -i <value> | --svmIterations <value>
         Number of training iterations - default 100.
   -t <value> | --trainingPercentage <value>
         Percentage of data that should be used for training. Remaining part will be used for testing. Default 0.7.
   -d | --printDetailed
         Outputs classification results of every tested movie otherwise only outputs error stats.
   --csv
         Emits the results in a CSV format
   -p <value> | --plotsInputFile <value>
         File with all plot descriptions - default 'plot_normalized.list'.
   --genresInputFile <value>
         File with all movie-genre mappings - default 'genres.list'.
   --keywordsInputFile <value>
         File with all movie-keywords mappings - default 'keywords.list'.
   --actorsInputFile <value>
         File with all movie-actor mappings - default 'actors.list'.
   -o <value> | --outputFile <value>
         Output file path - default stdout.
   --features <feature1>,<feature2>...
         Features to use for classification - default all. Possible values: plot,keywords,actors
   --genres <genre1>,<genre2>...
         Genres to classify - default all.
   --help
         Prints this usage text.
 ```
 
## Other scripts

**run-script.sh:**
This script can be used to execute the program via maven with different number of training iterations one after the other. This was used to generate the data in the Results folder to construct the graphs and analysis in the paper.

**aggregate-script.sh:**
This script can be used to aggregate the output of multiple runs of the program (e.g. via the run-script.sh) into one CSV file (rescollected) so that one can easily import this data into your favorite spreadsheet tool and generate graphs.
Please note that the order of the columns follow the standard order used by cmd line programs like 'ls' (e.g.: ``` 0 1 10 100 15 20 25 30 35 40 45 5 50 55 ```).
The order of the rows per genre is: number of false-positives, percentage of false-positives, number of false-negatives, percentage of false-negatives, number of wrong classifications, percentage of wrong classifications, number of right classifications, percentage of right classifications

**transpose.py**
Python script to transpose a CSV file. Used by the aggregate-script.sh.
