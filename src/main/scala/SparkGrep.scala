package spark.example

/* SimpleApp.scala */

import org.apache.log4j.{Level, Priority, Logger}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


case class Config(sampling: Double = 1.0, svmIterations: Int = 100,
                  plotsInputFile: String = "plot_normalized.list",
                  genresInputFile : String = "genres.list",
                  outputFile: String = "",
                  genres: Seq[String] = Seq("Comedy", "Drama"))   // TODO: add more genres

object SparkGrep {

  val logger = Logger.getLogger("## IMDb Movie Classification ##")

  def main(args: Array[String]) {

    val parser = new scopt.OptionParser[Config]("") {
      head("IMDb Movie Classification")
      opt[Double]('s', "sampling") optional() action { (x, c) =>
        c.copy(sampling = x) } text "input data sampling rate - default 1.0 (all)"
      opt[Int]('i', "svmIterations") optional() action { (x, c) =>
        c.copy(svmIterations = x) } text "number of training iterations - default 100"
      opt[String]('p', "plotsInputFile") optional() action { (x, c) =>
        c.copy(plotsInputFile = x) } text "file with all plot descriptions - default plot_normalized.list"
      opt[String]('g', "genresInputFile") optional() action { (x, c) =>
        c.copy(genresInputFile = x) } text "file with all movie-genre mappings - default genres.list"
      opt[String]('o', "outputFile") optional() action { (x, c) =>
        c.copy(outputFile = x) } text "output file path - default stdout"
      opt[Seq[String]]("genres") valueName "<genre1>,<genre2>..." optional() action { (x,c) =>
        c.copy(genres = x) } text "genres to classify - default all"
      help("help") text "prints this usage text"
    }

    parser.parse(args, Config()) match {
      case Some(config) =>

    val conf = new SparkConf()
      .setAppName("SparkGrep")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    /* ####################### PREPROCESSING ####################### */

    logger.log(Level.INFO, "Starting IMDb Movie Classification Task")

    /* predefined list of all genres that we analyze */
    val genreList = sc.broadcast(config.genres.toArray)

    /* read in plot descriptions */
    var documents = sc.textFile(config.plotsInputFile).cache()
    if(config.sampling < 1.0) documents = documents.sample(withReplacement = false, config.sampling, 5)
    val plotsWithLabel = documents.map(line => {
        val s = line.split(":::")
        (s(0), s(1).split(" ").toSeq)
      })

    logger.log(Level.INFO, "Finished read plot descriptions from file")

    /* create a list of all available movie titles */
    val movieTitles = plotsWithLabel.map(_._1).distinct()
    logger.log(Level.INFO, "Using %d movies".format(movieTitles.count()))

    /* calculate TF-IDF vectors */
    logger.log(Level.INFO, "Starting calculating TF")
    val hashingTF = new HashingTF()
    val tfWithLabel = plotsWithLabel.map(m => (m._1, hashingTF.transform(m._2)))

    logger.log(Level.INFO, "Starting calculating IDF")
    val tf = tfWithLabel.map(m => m._2)
    val idf = new IDF().fit(tf) //TODO: transmit as a broadcast variable?

    logger.log(Level.INFO, "Starting calculating TF-IDF")
    val moviesWithTFIDFVectors = tfWithLabel.map(m => (m._1, idf.transform(m._2)))

    logger.log(Level.INFO, "Finished with TF-IDF calculation")

    /* load movie to genre mapping */
    logger.log(Level.INFO, "Starting loading genres")
    val inputFile = sc.textFile(config.genresInputFile, 2).cache()
    var moviesWithGenres = inputFile.map(line => {
      val s = line.split("\t+")
      val genreVector = Vectors.dense(genreList.value.map(f => if (f == s(1)) 1.0 else 0.0))
      (s(0), genreVector)
    }).reduceByKey((v1, v2) => Vectors.dense(v1.toArray.toList.zip(v2.toArray).map(w => Math.min(1.0, w._1 + w._2)).toArray))
    if(config.sampling < 1.0) {
      // filter to only use the sampled movies from the plot description file
      moviesWithGenres = moviesWithGenres.join(movieTitles.map((_, 0))).map(v => (v._1, v._2._1))
    }

    logger.log(Level.INFO, "Finished loading genres")

    /* join TF-IDF vectors with genres to get labeled data */
    val moviesWithGenresAndTFIDFVector = moviesWithGenres.join(moviesWithTFIDFVectors)

     /* split data into training (70%) and test (30%) */
    val splits = moviesWithGenresAndTFIDFVector.randomSplit(Array(0.7, 0.3), seed = 11L)
    val moviesWithGenresAndTFIDFVector_training = splits(0).cache()
    val moviesWithGenresAndTFIDFVector_test = splits(1).cache()


    /* ####################### TRAINING ####################### */

    /* train SVMs for genres */
    logger.log(Level.INFO, "Starting training SVMs")
    var modelsLocal: Seq[SVMModel] = Seq()
    for(i <- 0 to genreList.value.length - 1) {
      modelsLocal = modelsLocal :+ trainModelForGenre(moviesWithGenresAndTFIDFVector_training, i, config.svmIterations)
      logger.log(Level.INFO, "Finished training SVM for '%s'".format(genreList.value(i)))
    }
    val models = sc.broadcast(modelsLocal.toArray)
    logger.log(Level.INFO, "Finished training SVMs")


    /* ####################### TESTING ####################### */

    /* use trained models to predict genres of the test data */
    logger.log(Level.INFO, "Starting testing based on SVMs")
    val res = moviesWithGenresAndTFIDFVector_test
      .map(m => (m._1, m._2._1, generatePredictedGenreVector(m._2._2, models.value)))
    logger.log(Level.INFO, "Finished testing based on SVMs")


    /* ####################### EVALUATION ####################### */

    logger.log(Level.INFO, "Starting evaluation")

    /* output some textual info about predicted and expected genres for the movies */
    res.foreach(m => {
      val genres = genreList.value
      val expectedGenreVector = m._2
      val predictedGenreVector = m._3

      val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray).zip(genres)
      val notPredicatedGenres = zipped.flatMap(v => if (v._1._1 > v._1._2) Seq(v._2) else Seq()).mkString(",")
      val tooMuchPredicatedGenres = zipped.flatMap(v => if (v._1._1 < v._1._2) Seq(v._2) else Seq()).mkString(",")
      val correctlyPredictedGenres = zipped.flatMap(v => if ((v._1._1 == 1) && (v._1._2 == 1)) Seq(v._2) else Seq()).mkString(",")

      logger.log(Level.INFO, "%s: correctly classified: [%s] | not classified: [%s] | too much classified: [%s]"
        .format(m._1, correctlyPredictedGenres, notPredicatedGenres, tooMuchPredicatedGenres))
    })

    /* calculate error rate */
    val precision = res.map(m => {
      val expectedGenreVector = m._2
      val predictedGenreVector: Vector = m._3
      val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray)

      val errors = zipped.map(v => if (v._1 != v._2) 1 else 0).sum
      (errors.toLong, zipped.size.toLong)
    }).reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))

    val errorPercentage = precision._1.toDouble / precision._2.toDouble

    logger.log(Level.INFO, "%d wrong classifications, %d right classifications -> error rate of %f"
      .format(precision._1, precision._2 - precision._1, errorPercentage))

      case None =>
      // arguments are bad, error message will have been displayed
    }

    System.exit(0)
  }

  def trainModelForGenre(trainingData: RDD[(String, (Vector, Vector))], genreIndex: Int, numIterations: Int): SVMModel = {
    val trainingDataForGenre = trainingData.map(m => LabeledPoint(m._2._1(genreIndex), m._2._2)).cache()
    val model = SVMWithSGD.train(trainingDataForGenre, numIterations)
    model
  }

  def generatePredictedGenreVector(tfIDFVector: Vector, genreModels: Array[SVMModel]): Vector = {
    val prediction = genreModels.map(model => model.predict(tfIDFVector))
    Vectors.dense(prediction)
  }

}