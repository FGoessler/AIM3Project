package de.tuberlin.dima.aim3.project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

case class Config(sampling: Double = 1.0, svmIterations: Int = 100,
                  trainingPercentage: Double = 0.7,
                  printDetailed: Boolean = false,
                  csvOutput: Boolean = false,
                  plotsInputFile: String = "plot_normalized.list",
                  genresInputFile: String = "genres.list",
                  keywordsInputFile: String = "keywords.list",
                  outputFile: Option[String] = None,
                  features: Seq[String] = Seq("plot", "keywords"),
                  genres: Seq[String] = Seq("Comedy", "Drama", "Action", "Documentary", "Adult",
                    "Romance", "Thriller", "Animation", "Family", "Horror", "Music", "Crime",
                    "Adventure", "Fantasy", "Sci-Fi", "Mystery", "Biography", "History", "Sport",
                    "Musical", "War", "Western", "News", "Reality-TV"))

object IMDbMovieClassification {

  val logger = Logger.getLogger("## IMDb Movie Classification ##")

  def main(args: Array[String]) {

    val parser = new scopt.OptionParser[Config]("") {
      head("IMDb Movie Classification")
      opt[Double]('s', "sampling") optional() action { (x, c) =>
        c.copy(sampling = x)
      } validate { x =>
        if (x > 0.0 && x <= 1.0) success else failure("Value <sampling> must be between 0.0 and 1.0")
      } text "Input data sampling rate - default 1.0 (all)."
      opt[Int]('i', "svmIterations") optional() action { (x, c) =>
        c.copy(svmIterations = x)
      } text "Number of training iterations - default 100."
      opt[Double]('t', "trainingPercentage") optional() action { (x, c) =>
        c.copy(trainingPercentage = x)
      } validate { x =>
        if (x > 0.0 && x < 1.0) success else failure("Value <trainingPercentage> must be between 0.0 and 1.0")
      } text "Percentage of data that should be used for training. Remaining part will be used for testing. Default 0.7."
      opt[Unit]('d', "printDetailed") optional() action { (_, c) =>
        c.copy(printDetailed = true)
      } text "Outputs classification results of every tested movie otherwise only outputs error stats."
      opt[Unit]("csv") optional() action { (_, c) =>
        c.copy(csvOutput = true)
      } text "Emits the results in a CSV format"
      opt[String]('p', "plotsInputFile") optional() action { (x, c) =>
        c.copy(plotsInputFile = x)
      } text "File with all plot descriptions - default 'plot_normalized.list'."
      opt[String]('g', "genresInputFile") optional() action { (x, c) =>
        c.copy(genresInputFile = x)
      } text "File with all movie-genre mappings - default 'genres.list'."
      opt[String]('g', "keywordsInputFile") optional() action { (x, c) =>
        c.copy(keywordsInputFile = x)
      } text "File with all movie-keywords mappings - default 'keywords.list'."
      opt[String]('o', "outputFile") optional() action { (x, c) =>
        c.copy(outputFile = Some(x))
      } text "Output file path - default stdout."
      opt[Seq[String]]("features") valueName "<feature1>,<feature2>..." optional() action { (x, c) =>
        c.copy(features = x)
      } text "Features to use for classification - default all. Possible values: plot,keywords"
      opt[Seq[String]]("genres") valueName "<genre1>,<genre2>..." optional() action { (x, c) =>
        c.copy(genres = x)
      } text "Genres to classify - default all."
      help("help") text "Prints this usage text."
    }

    parser.parse(args, Config()) match {
      case Some(config) =>
        exec(config)
      case None =>
      // arguments are bad, error message will have been displayed
    }

    System.exit(0)
  }

  def exec(config: Config): Unit = {
    val conf = new SparkConf()
      .setAppName("SparkGrep")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    /* ####################### PREPROCESSING ####################### */

    logger.log(Level.INFO, "Starting IMDb Movie Classification Task")
    logger.log(Level.INFO, "Config: %s".format(config))

    /* predefined list of all genres that we analyze */
    val genreList = sc.broadcast(config.genres.toArray)

    var movieTitles: RDD[String] = null
    var featureVectorsWithLabel: Seq[RDD[(String, Vector)]] = Seq()
    var numFeatureVectorsPerMovie = 0
    if (config.features.contains("plot")) {
      /* read in plot descriptions */
      var documents = sc.textFile(config.plotsInputFile).cache()
      if (config.sampling < 1.0) documents = documents.sample(withReplacement = false, config.sampling, 5)
      val plotsWithLabel = documents.map(line => {
        val s = line.split(":::")
        (s(0), s(1).split(" ").toSeq)
      })

      logger.log(Level.INFO, "Finished read plot descriptions from file")

      /* create a list of all available movie titles */
      movieTitles = plotsWithLabel.map(_._1).distinct()
      logger.log(Level.INFO, "Building features for %d movies".format(movieTitles.count()))

      /* calculate TF-IDF vectors */
      logger.log(Level.INFO, "Starting calculating TF")
      val hashingTF = new HashingTF()
      featureVectorsWithLabel = featureVectorsWithLabel :+ plotsWithLabel.map(m => (m._1, hashingTF.transform(m._2)))
      numFeatureVectorsPerMovie = numFeatureVectorsPerMovie + 1
    }

    if (config.features.contains("keywords")) {
      /* load keywords */
      val keywordsInputFile = sc.textFile(config.genresInputFile, 2).cache()
      val moviesWithKeywords = keywordsInputFile.map(line => {
        val s = line.split("\t+")
        (s(0), Seq(s(1)))
      }).reduceByKey((k1, k2) => k1 ++ k2)

      /* create a list of all available movie titles */
      movieTitles = moviesWithKeywords.map(_._1).distinct()
      logger.log(Level.INFO, "Building features for %d movies".format(movieTitles.count()))

      /* construct feature vector */
      val keywordsTF = new HashingTF()
      featureVectorsWithLabel = featureVectorsWithLabel :+ moviesWithKeywords.map(m => (m._1, keywordsTF.transform(m._2)))
      numFeatureVectorsPerMovie = numFeatureVectorsPerMovie + 1
    }
    // TODO: build feature vector based on actors

    /* load movie to genre mapping */
    logger.log(Level.INFO, "Starting loading genres")
    val inputFile = sc.textFile(config.genresInputFile, 2).cache()
    var moviesWithGenres = inputFile.flatMap(line => {
      val s = line.split("\t+")
      val genreVector = Vectors.dense(genreList.value.map(f => if (f == s(1)) 1.0 else 0.0))
      // filter out movies with no genre info
      if (genreVector.numNonzeros > 0) {
        Seq((s(0), genreVector))
      } else {
        Seq()
      }
    }).reduceByKey((v1, v2) => Vectors.dense(v1.toArray.toList.zip(v2.toArray).map(w => Math.min(1.0, w._1 + w._2)).toArray))
    if (config.sampling < 1.0) {
      // filter to only use the sampled movies from the plot description file
      moviesWithGenres = moviesWithGenres.join(movieTitles.map((_, 0))).map(v => (v._1, v._2._1))
    }

    logger.log(Level.INFO, "Using %d movies with genre data".format(moviesWithGenres.count()))
    logger.log(Level.INFO, "Finished loading genres")

    /* join feature vectors with genres to get labeled data */
    var moviesWithGenresAndFeatureVector = moviesWithGenres.map(x => (x._1, (x._2, Seq[Vector]())))
    for (featureVectors <- featureVectorsWithLabel) {
      val joinedVectors = moviesWithGenresAndFeatureVector.join(featureVectors)
      moviesWithGenresAndFeatureVector = joinedVectors.map(x => (x._1, (x._2._1._1, x._2._1._2 :+ x._2._2)))
    }

    /* split data into training and test */
    val splits = moviesWithGenresAndFeatureVector.randomSplit(Array(config.trainingPercentage, 1.0 - config.trainingPercentage), seed = 11L)
    val moviesWithGenresAndFeatureVector_training = splits(0).cache()
    val moviesWithGenresAndFeatureVector_test = splits(1).cache()

    logger.log(Level.INFO, "Training on %d movies | Testing on %d movies".format(
      moviesWithGenresAndFeatureVector_training.count(),
      moviesWithGenresAndFeatureVector_test.count())
    )

    /* ####################### TRAINING ####################### */

    /* train SVMs for genres */
    logger.log(Level.INFO, "Starting training SVMs")
    var modelsLocal: Seq[Seq[SVMModel]] = Seq()
    for (i <- genreList.value.indices) {
      modelsLocal = modelsLocal :+ trainModelsForGenre(moviesWithGenresAndFeatureVector_training, numFeatureVectorsPerMovie, i, config.svmIterations)
      logger.log(Level.INFO, "Finished training SVM for '%s'".format(genreList.value(i)))
    }
    val models = sc.broadcast(modelsLocal.toArray)
    logger.log(Level.INFO, "Finished training SVMs")


    /* ####################### TESTING ####################### */

    /* use trained models to predict genres of the test data */
    logger.log(Level.INFO, "Starting testing based on SVMs")
    val res = moviesWithGenresAndFeatureVector_test
      .map(m => (m._1, m._2._1, generatePredictedGenreVector(m._2._2, models.value)))
    logger.log(Level.INFO, "Finished testing based on SVMs")


    /* ####################### EVALUATION ####################### */

    logger.log(Level.INFO, "Starting evaluation")

    /* compute some textual info about predicted and expected genres for the movies */
    var textualResult = sc.makeRDD(Seq[String]())
    if (config.printDetailed) {
      textualResult = res.map(m => {
        val genres = genreList.value
        val expectedGenreVector = m._2
        val predictedGenreVector = m._3

        val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray).zip(genres)
        val notPredicatedGenres = zipped.flatMap(v => if (v._1._1 > v._1._2) Seq(v._2) else Seq()).mkString(",")
        val tooMuchPredicatedGenres = zipped.flatMap(v => if (v._1._1 < v._1._2) Seq(v._2) else Seq()).mkString(",")
        val correctlyPredictedGenres = zipped.flatMap(v => if ((v._1._1 == 1) && (v._1._2 == 1)) Seq(v._2) else Seq()).mkString(",")

        "%s: correctly classified: [%s] | not classified: [%s] | too much classified: [%s]"
          .format(m._1, correctlyPredictedGenres, notPredicatedGenres, tooMuchPredicatedGenres)
      })
    }

    /* calculate error rate */
    val errorStats = res.map(m => {
      val expectedGenreVector = m._2
      val predictedGenreVector: Vector = m._3
      val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray)

      val errStats = zipped.map(v => {
        val falsePositive = if (v._1 == 0 && v._2 == 1) 1 else 0
        val falseNegative = if (v._1 == 1 && v._2 == 0) 1 else 0
        (falsePositive, falseNegative)
      })
      val completelyCorrectlyClassified = if (errStats.map(v => v._1 + v._2).sum == 0) 1 else 0
      //TODO: collect additional error classes: only-1-false-negative, only-1-false-positive, only-2-false-negative, only-2-false-positive, only-1-false-negative-&-1-false-positive, ...


      (errStats, 1, completelyCorrectlyClassified)
    }).reduce((v1, v2) => {
      (v1._1.zip(v2._1).map(x => (x._1._1 + x._2._1, x._1._2 + x._2._2)), v1._2 + v2._2, v1._3 + v2._3)
    })

    val numClassifiedMovies = errorStats._2
    val numCompletelyCorrectlyClassifiedMovies = errorStats._3

    val precision = errorStats._1.reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
    val numTotalFalsePositives = precision._1
    val numTotalFalseNegatives = precision._2
    val numTotalErrors = precision._1 + precision._2
    val numClassifications = errorStats._1.size * numClassifiedMovies

    /* output results and error rate */
    val errorPerGenreTextualResult = errorStats._1.zip(genreList.value).map(e => {
      val formatString = if (config.csvOutput) "%s,%d,%f,%d,%f,%d,%f,%d,%f" else "%s: false-positives: %d (%f) | false-negatives: %d (%f) | %d wrong classifications (%f) | right classifications: %d (%f)"
      formatString.format(e._2,
        e._1._1, e._1._1.toDouble / numClassifiedMovies.toDouble,
        e._1._2, e._1._2.toDouble / numClassifiedMovies.toDouble,
        e._1._1 + e._1._2, (e._1._1.toDouble + e._1._2.toDouble) / numClassifiedMovies.toDouble,
        numClassifiedMovies - e._1._2 - e._1._2, (numClassifiedMovies - e._1._2 - e._1._2).toDouble / numClassifiedMovies.toDouble)
    })

    val completelyCorrectlyClassifiedMoviesString = "## Total: %d of %d (%f) movies completely correctly classified ##"
      .format(numCompletelyCorrectlyClassifiedMovies, numClassifiedMovies, numCompletelyCorrectlyClassifiedMovies.toDouble / numClassifiedMovies.toDouble)

    val formatString = if (config.csvOutput) "Total,%d,%f,%d,%f,%d,%f,%d,%f" else "## Total: false-positives: %d (%f) | false-negatives: %d (%f) | %d wrong classifications (%f) | right classifications: %d (%f) ##"
    val totalErrorString = formatString.format(
      numTotalFalsePositives, numTotalFalsePositives.toDouble / numClassifications.toDouble,
      numTotalFalseNegatives, numTotalFalseNegatives / numClassifications.toDouble,
      numTotalErrors, numTotalErrors.toDouble / numClassifications.toDouble,
      numClassifications - numTotalErrors, (numClassifications - numTotalErrors).toDouble / numClassifications.toDouble)

    val errorRatesStrings = errorPerGenreTextualResult ++ Seq(completelyCorrectlyClassifiedMoviesString) ++ Seq(totalErrorString)

    val joinedTextualResult = textualResult ++ sc.makeRDD(errorRatesStrings)

    config.outputFile match {
      case Some(fname) =>
        joinedTextualResult.saveAsTextFile(fname)
      case None =>
        joinedTextualResult.foreach(println(_))
    }
  }

  def trainModelsForGenre(trainingData: RDD[(String, (Vector, Seq[Vector]))], numModels: Int, genreIndex: Int, numIterations: Int): Seq[SVMModel] = {
    var models = Seq[SVMModel]()
    for (i <- 0 to numModels - 1) {
      val trainingDataForGenre = trainingData.map(m => LabeledPoint(m._2._1(genreIndex), m._2._2(i))).cache()
      val model = SVMWithSGD.train(trainingDataForGenre, numIterations)
      models = models :+ model
    }
    models
  }

  def generatePredictedGenreVector(featureVectors: Seq[Vector], genreModels: Array[Seq[SVMModel]]): Vector = {
    val prediction = genreModels.map(models => {
      val modelsWithFetaureVectors = models.zip(featureVectors)
      modelsWithFetaureVectors.map(x => x._1.predict(x._2)).sum / featureVectors.length.toDouble
    })
    Vectors.dense(prediction)
  }
}
